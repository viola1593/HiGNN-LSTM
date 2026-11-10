"""Hyperparameter sweep for the anomaly ConvLSTM: learning rate x hidden_dim x num_layers, one seed.

Mirrors hparam_sweep.py (HiGNN) with the same grid and the cosine schedule, so both models
are tuned alike. Runs are ranked by the best-checkpoint val_rmse (best_model_score stored in
the checkpoint; eval_on_val.json holds the last-epoch score and only marks a run as finished)
into <exp_dir>/hparam_sweep.csv. The winner must be re-run (seed study) with the same --epochs.

    PYTHONPATH=src/ python scripts/hparam_sweep_convlstm.py --config_path configs/convlstm.yaml --gpu 1

Resumable: grid points that already have a best checkpoint in exp_dir are not retrained.
"""

import argparse
import copy
import glob
import itertools
import os
import re
import subprocess
import sys

import pandas as pd

from hparam_sweep import best_ckpt, best_score
from utils.utils import read_config, save_config


def build_configs(base, lrs, hidden_dims, depths, epochs, gpu, exp_dir):
    """One config per (lr, hidden_dim, num_layers) grid point; base is not modified."""
    configs = []
    for lr, hidden, depth in itertools.product(lrs, hidden_dims, depths):
        config = copy.deepcopy(base)
        config["model"].update(learning_rate=lr, hidden_dim=hidden, num_layers=depth, lr_scheduler="cosine")
        config["pl"]["max_epochs"] = epochs
        config["pl"]["devices"] = [gpu]
        config["experiment"]["exp_dir"] = exp_dir
        config["experiment"]["experiment_name"] += f"_h{hidden}_d{depth}_lr{lr}_ep{epochs}"
        configs.append(config)
    return configs


def collect_results(exp_dir):
    """Best-checkpoint val_rmse and best epoch of every finished run in exp_dir, best first."""
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(exp_dir, "*_ep*_*"))):
        ckpt = best_ckpt(run_dir)
        if not os.path.exists(os.path.join(run_dir, "eval_on_val.json")) or ckpt is None:
            continue
        rows.append({"run": re.match(r"^(.*_ep\d+)_", os.path.basename(run_dir)).group(1),
                     "val_rmse": best_score(ckpt),
                     "best_epoch": int(re.match(r"epoch=(\d+)", os.path.basename(ckpt)).group(1)),
                     "run_dir": run_dir})
    return pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--lrs", type=float, nargs="+", default=[0.0005, 0.001, 0.002])
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--exp_dir", default="experiments/baseline2D/anom/hparam_sweep")
    parser.add_argument("--no_train", action="store_true", help="only write the configs")
    args = parser.parse_args()

    cfg_dir = os.path.join(args.exp_dir, "hparam_sweep_configs")
    os.makedirs(cfg_dir, exist_ok=True)
    configs = build_configs(read_config(args.config_path), args.lrs, args.hidden_dims, args.depths,
                            args.epochs, args.gpu, args.exp_dir)

    for config in configs:
        name = config["experiment"]["experiment_name"]
        cfg_path = os.path.join(cfg_dir, f"{name}.yaml")
        save_config(config, cfg_path)
        if args.no_train:
            continue
        if any(best_ckpt(d) for d in glob.glob(os.path.join(args.exp_dir, f"{name}_*"))):
            print(f"[hparam sweep] {name} already trained, skipping")
            continue

        print(f"[hparam sweep] training {name} ({cfg_path})")
        proc = subprocess.run([sys.executable, "scripts/train_ConvLSTM2D.py",
                               "--config_path", cfg_path, "--seed", str(args.seed)])
        if proc.returncode != 0:
            print(f"[hparam sweep] WARNING: {name} failed, continuing")

    if not args.no_train:
        df = collect_results(args.exp_dir)
        df.to_csv(os.path.join(args.exp_dir, "hparam_sweep.csv"), index=False)
        print(df.to_string())


if __name__ == "__main__":
    main()

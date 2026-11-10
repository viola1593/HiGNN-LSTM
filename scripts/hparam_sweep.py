"""Hyperparameter sweep for the anomaly HiGNN: learning rate x hidden_dim x depth, one seed.

Depth sets num_processing_gnns and rnn_layers together. Runs are ranked by the
best-checkpoint val_rmse (best_model_score stored in the checkpoint, normalised, summed over
leads; metrics_eval.json holds the last-epoch score and only marks a run as finished) into
<exp_dir>/hparam_sweep.csv. The cosine schedule anneals over max_epochs, so the winner
must be re-run (seed study) with the same --epochs.

    PYTHONPATH=src/ python scripts/hparam_sweep.py --config_path configs/hignn_lstm.yaml --gpu 1

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
import torch

from utils.utils import read_config, save_config


def build_configs(base, lrs, hidden_dims, depths, epochs, gpu, exp_dir):
    """One config per (lr, hidden_dim, depth) grid point; base is not modified."""
    configs = []
    for lr, hidden, depth in itertools.product(lrs, hidden_dims, depths):
        config = copy.deepcopy(base)
        config["model"]["learning_rate"] = lr
        config["model"]["hidden_dim"] = hidden
        config["model"]["num_processing_gnns"] = depth
        config["model"]["rnn_layers"] = depth
        config["model"]["skip_gnn"] = False
        config["pl"]["max_epochs"] = epochs
        config["pl"]["devices"] = [gpu]
        config["experiment"]["exp_dir"] = exp_dir
        config["experiment"]["experiment_name"] += f"_h{hidden}_d{depth}_lr{lr}_ep{epochs}"
        configs.append(config)
    return configs


def best_ckpt(run_dir):
    """The best-val_rmse checkpoint of a run dir (not last.ckpt), or None."""
    ckpts = [p for p in glob.glob(os.path.join(run_dir, "*.ckpt")) if os.path.basename(p) != "last.ckpt"]
    return ckpts[0] if ckpts else None


def best_score(ckpt):
    """The monitored val_rmse of a checkpoint, from the ModelCheckpoint state saved in it."""
    callbacks = torch.load(ckpt, map_location="cpu", weights_only=False)["callbacks"]
    return next(v["best_model_score"].item() for k, v in callbacks.items() if k.startswith("ModelCheckpoint"))


def collect_results(exp_dir):
    """Best-checkpoint val_rmse and best epoch of every finished run in exp_dir, best first."""
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(exp_dir, "*_seed*"))):
        ckpt = best_ckpt(run_dir)
        if not os.path.exists(os.path.join(run_dir, "metrics_eval.json")) or ckpt is None:
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
    parser.add_argument("--exp_dir", default="experiments/HiGNN/V0anom/hparam_sweep")
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
        if any(best_ckpt(d) for d in glob.glob(os.path.join(args.exp_dir, f"{name}_*_seed{args.seed}_*"))):
            print(f"[hparam sweep] {name} already trained, skipping")
            continue

        print(f"[hparam sweep] training {name} ({cfg_path})")
        proc = subprocess.run([sys.executable, "scripts/train_HiGNN.py",
                               "--config_path", cfg_path, "--seed", str(args.seed)])
        if proc.returncode != 0:
            print(f"[hparam sweep] WARNING: {name} failed, continuing")

    if not args.no_train:
        df = collect_results(args.exp_dir)
        df.to_csv(os.path.join(args.exp_dir, "hparam_sweep.csv"), index=False)
        print(df.to_string())


if __name__ == "__main__":
    main()


# get eval from best checkpoint for all runs in exp_dir, and write to hparam_sweep.csv
# PYTHONPATH=src:scripts python -c "import hparam_sweep as h; d='experiments/HiGNN/V0anom/hparam_sweep'; h.collect_results(d).to_csv(d+'/hparam_sweep.csv', index=False)"

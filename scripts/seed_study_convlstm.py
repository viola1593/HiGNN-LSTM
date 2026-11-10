"""Seed study harness for the ConvLSTM baseline, mirroring run_seed_study.py (HiGNN).

Trains one run per seed with the given config, evaluates each run's best-val_rmse
checkpoint on the validation and test periods, and writes the per-lead grid-cell
RMSE table. The across-seed max-min of the lead-mean on test is the noise floor.

Run from the repo root:

    PYTHONPATH=src/ python scripts/seed_study_convlstm.py --config_path configs/convlstm.yaml \
      --seeds 42 43 44 45 --gpu 0

Resumable: seeds already in the manifest with a checkpoint are not retrained, and
evaluations whose mean_timestep_scores.json exists are not re-run.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import pandas as pd

from run_seed_study import DATASETS, has_best_ckpt, write_report
from utils.utils import read_config


# label of the runs in the report; the HiGNN study labels its runs by graph variant
VARIANT = "convlstm"


def train_seed(config_path, seed, exp_dir, run_prefix):
    """Train one seed. Returns its experiment dir, or None if training failed."""
    t0 = time.time()
    proc = subprocess.run([sys.executable, "scripts/train_ConvLSTM2D.py",
                           "--config_path", config_path, "--seed", str(seed)])
    if proc.returncode != 0:
        return None

    # create_experiment_dir neither returns nor prints its path, nor puts the seed in it;
    # sequential execution means the only dir newer than t0 is the one this subprocess made.
    candidates = [d for d in glob.glob(os.path.join(exp_dir, f"{run_prefix}_*"))
                  if os.path.isdir(d) and os.path.getmtime(d) > t0]
    assert len(candidates) == 1, f"expected one new run dir for seed {seed}, got {candidates}"
    return candidates[0]


def eval_dir_for(run_dir, dataset):
    """Where evaluation_ConvLSTM writes the outputs of one run; mirrors its create_experiment_dir."""
    return os.path.join(run_dir, f"eval_on_{dataset}")


def evaluate_run(run_dir, dataset, gpu):
    """Evaluate one run on one dataset. Returns True if the scores json is there afterwards."""
    eval_dir = eval_dir_for(run_dir, dataset)
    scores_path = os.path.join(eval_dir, "mean_timestep_scores.json")
    if os.path.exists(scores_path):
        print(f"[seed study] {scores_path} exists, skipping evaluation")
        return True

    # predictions.pt may have been written by a different checkpoint, so never trust it
    stale = os.path.join(eval_dir, "predictions.pt")
    if os.path.exists(stale):
        os.remove(stale)

    proc = subprocess.run([sys.executable, "scripts/evaluation_ConvLSTM.py",
                           "--exp_path", run_dir, "--dataset", dataset, "--gpu", str(gpu)])
    return proc.returncode == 0 and os.path.exists(scores_path)


def collect_scores(runs):
    """Read the per-lead scores of every run into the tidy format of run_seed_study."""
    rows = []
    for seed, run_dir in runs.items():
        for dataset in DATASETS:
            path = os.path.join(eval_dir_for(run_dir, dataset), "mean_timestep_scores.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                scores = json.load(f)
            # metric names must match what run_seed_study.build_table selects
            for metric, key in (("rmse_latw", "rmse_latw"), ("clim_rmse_latw", "rmse_latw_clim")):
                for lead, value in enumerate(scores[key], start=1):
                    rows.append({"dataset": dataset, "variant": VARIANT, "seed": int(seed),
                                 "lead": lead, "metric": metric, "value": value})
    return pd.DataFrame(rows, columns=["dataset", "variant", "seed", "lead", "metric", "value"])


def main():
    parser = argparse.ArgumentParser(
        prog="seed_study_convlstm.py",
        description="Train and evaluate one ConvLSTM config over several seeds and tabulate the spread.")
    parser.add_argument("--config_path", required=True, help="Path to the config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU for the evaluations; training uses pl.devices from the config")
    parser.add_argument("--datasets", nargs="+", default=["val", "test"], choices=DATASETS,
                        help="Datasets to evaluate each seed on")
    args = parser.parse_args()

    config = read_config(args.config_path)
    exp_dir = config["experiment"]["exp_dir"]
    run_prefix = config["experiment"]["experiment_name"]
    manifest_path = os.path.join(exp_dir, f"{run_prefix}_seed_study_manifest.json")

    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    warnings = []
    for seed in args.seeds:
        key = str(seed)
        if key in manifest and has_best_ckpt(manifest[key]):
            print(f"[seed study] seed {seed} already trained in {manifest[key]}, skipping")
            continue
        print(f"[seed study] training seed {seed}")
        run_dir = train_seed(args.config_path, seed, exp_dir, run_prefix)
        if run_dir is None:
            warnings.append(f"training failed for seed {seed}; it is missing from the table")
            print(f"[seed study] WARNING: training failed for seed {seed}, continuing")
            continue
        manifest[key] = run_dir
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    for seed in args.seeds:
        key = str(seed)
        if key not in manifest:
            continue
        for dataset in args.datasets:
            print(f"[seed study] evaluating seed {seed} on {dataset}")
            if not evaluate_run(manifest[key], dataset, args.gpu):
                warnings.append(f"evaluation failed for seed {seed} on {dataset}")
                print(f"[seed study] WARNING: evaluation failed for seed {seed} on {dataset}")

    runs = {key: manifest[key] for key in map(str, args.seeds) if key in manifest}
    tidy = collect_scores(runs)
    if tidy.empty:
        print("[seed study] no scores to aggregate")
        return
    write_report(tidy,
                 os.path.join(exp_dir, f"{run_prefix}_seed_study.csv"),
                 os.path.join(exp_dir, f"{run_prefix}_seed_study.md"),
                 warnings)


if __name__ == "__main__":
    main()

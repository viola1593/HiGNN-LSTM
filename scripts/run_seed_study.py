"""Seed study harness.

Trains one run per seed with the given config, evaluates each run's best-val_rmse
checkpoint on the validation and test periods, and writes the per-lead grid-cell
RMSE table. The across-seed max-min of the lead-mean on test is the noise floor.

Run from the repo root:

    PYTHONPATH=src/ python scripts/run_seed_study.py --config_path configs/hignn_lstm.yaml \
      --seeds 42 43 44 45 --gpu 0

Resumable: seeds already in the manifest with a checkpoint are not retrained, and
evaluations whose per_timestep_scores.csv exists are not re-run.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import pandas as pd

from utils.utils import read_config


# every dataset the report picks up if its scores exist; only val and test are evaluated by default
DATASETS = ("val", "test", "grace")


def has_best_ckpt(run_dir):
    """True if the run dir holds a best-val_rmse checkpoint (i.e. one that is not last.ckpt)."""
    return any(os.path.basename(p) != "last.ckpt"
               for p in glob.glob(os.path.join(run_dir, "*.ckpt")))


def train_seed(config_path, seed, exp_dir, run_prefix):
    """Train one seed. Returns its experiment dir, or None if training failed."""
    t0 = time.time()
    proc = subprocess.run([sys.executable, "scripts/train_HiGNN.py",
                           "--config_path", config_path, "--seed", str(seed)])
    if proc.returncode != 0:
        return None

    # create_experiment_dir neither returns nor prints its path; sequential execution
    # means the only dir newer than t0 is the one this subprocess just made.
    candidates = [d for d in glob.glob(os.path.join(exp_dir, f"{run_prefix}_seed{seed}_*"))
                  if os.path.isdir(d) and os.path.getmtime(d) > t0]
    assert len(candidates) == 1, f"expected one new run dir for seed {seed}, got {candidates}"
    return candidates[0]


def eval_dir_for(run_dir, dataset):
    """Where evaluation_HiGNN writes the outputs of one run; mirrors its create_experiment_dir."""
    return os.path.join(run_dir, f"eval_on_{dataset}")


def evaluate_run(run_dir, dataset, gpu, scores_only=False):
    """Evaluate one run on one dataset. Returns True if the scores csv is there afterwards."""
    eval_dir = eval_dir_for(run_dir, dataset)
    scores_path = os.path.join(eval_dir, "per_timestep_scores.csv")
    if os.path.exists(scores_path):
        print(f"[seed study] {scores_path} exists, skipping evaluation")
        return True

    # predictions.pt may have been written by a different checkpoint, so never trust it
    stale = os.path.join(eval_dir, "predictions.pt")
    if os.path.exists(stale):
        os.remove(stale)

    proc = subprocess.run([sys.executable, "scripts/evaluation_HiGNN.py",
                           "--exp_path", run_dir, "--dataset", dataset, "--gpu", str(gpu)]
                          + (["--scores_only"] if scores_only else []))
    return proc.returncode == 0 and os.path.exists(scores_path)


def collect_scores(runs):
    """Read the per-timestep scores of every run into tidy long format."""
    rows = []
    for seed, run_dir in runs.items():
        variant = read_config(os.path.join(run_dir, "config.yaml"))["datamodule"]["graph_variant"]
        for dataset in DATASETS:
            path = os.path.join(eval_dir_for(run_dir, dataset), "per_timestep_scores.csv")
            if not os.path.exists(path):
                continue
            scores = pd.read_csv(path)
            for _, row in scores.iterrows():
                for metric in ("rmse_latw", "clim_rmse_latw"):
                    rows.append({"dataset": dataset, "variant": variant, "seed": int(seed),
                                 "lead": int(row["Timestep"]) + 1, "metric": metric,
                                 "value": row[metric]})
    return pd.DataFrame(rows, columns=["dataset", "variant", "seed", "lead", "metric", "value"])


def build_table(tidy, dataset, variant):
    """Per-lead and lead-mean RMSE, one row per run plus mean, max-min and climatology."""
    subset = tidy[(tidy["dataset"] == dataset) & (tidy["variant"] == variant)]
    runs = subset[subset["metric"] == "rmse_latw"].pivot(index="seed", columns="lead", values="value")
    runs["mean"] = runs.mean(axis=1)
    runs.index = [f"{variant}/seed{seed}" for seed in runs.index]

    climatology = subset[subset["metric"] == "clim_rmse_latw"].pivot(
        index="seed", columns="lead", values="value").mean(axis=0)
    climatology["mean"] = climatology.mean()

    table = runs.copy()
    table.loc[f"{variant} mean"] = runs.mean(axis=0)
    table.loc[f"{variant} max-min"] = runs.max(axis=0) - runs.min(axis=0)
    table.loc["climatology"] = climatology
    table.columns = [c if c == "mean" else f"L{c}" for c in table.columns]
    return table


def md_table(table):
    header = ["run"] + [str(c) for c in table.columns]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for name, row in table.iterrows():
        lines.append("| " + " | ".join([str(name)] + [f"{v:.3f}" for v in row]) + " |")
    return "\n".join(lines)


def write_report(tidy, csv_path, md_path, warnings):
    tidy.to_csv(csv_path, index=False)

    blocks = []
    if warnings:
        blocks.append("> **WARNING**\n" + "\n".join(f"> - {w}" for w in warnings))
    for dataset in DATASETS:
        for variant in sorted(tidy[tidy["dataset"] == dataset]["variant"].unique()):
            table = build_table(tidy, dataset, variant)
            blocks.append(f"### RMSE [cm] - grid cells - {dataset.upper()}\n\n" + md_table(table))
            if dataset == "test":
                floor = table.loc[f"{variant} max-min", "mean"]
                blocks.append(f'Noise floor ({variant}, test, lead-mean max-min): '
                              f'**{floor:.3f} cm** - the "~=" threshold for Decision 12.')
    report = "\n\n".join(blocks) + "\n"

    with open(md_path, "w") as f:
        f.write(report)
    print("\n" + report)
    print(f"[seed study] wrote {csv_path} and {md_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="run_seed_study.py",
        description="Train and evaluate one config over several seeds and tabulate the spread.")
    parser.add_argument("--config_path", required=True, help="Path to the config file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU for the evaluations; training uses pl.devices from the config")
    parser.add_argument("--datasets", nargs="+", default=["val", "test"], choices=DATASETS,
                        help="Datasets to evaluate each seed on")
    parser.add_argument("--scores_only", action="store_true",
                        help="Evaluate without saving predictions.pt or figures")
    args = parser.parse_args()

    config = read_config(args.config_path)
    exp_dir = config["experiment"]["exp_dir"]
    name = config["experiment"]["experiment_name"]
    run_prefix = f"{name}_{config['datamodule']['graph_variant']}"
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
            if not evaluate_run(manifest[key], dataset, args.gpu, args.scores_only):
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

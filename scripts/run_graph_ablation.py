"""Graph-parameter ablation: one seed study per variant, each changing one graph parameter of the base model.

The base model is configs/hignn_lstm.yaml, the config.yaml saved in the seed-42 run of the first study in
results_tables.MODELS.
Each variant goes through run_seed_study.py on the test set, without figures or predictions.pt. The
summary table in OUT_DIR compares the lead-mean test RMSE of every variant to all seeds of the base model:
difference in cm, its 95 % confidence interval and the p-value of a Welch t-test. ablation_graph_per_lead.tex
gives the differences and confidence intervals per lead, with the leads as columns; the p-values per lead
are in ablation_graph.md. A variant that misses a seed or its scores is left out of the tables, with a warning.

Run from the repo root, in the background:

    PYTHONPATH=src/ nohup python scripts/run_graph_ablation.py --gpu 1 \
      > experiments/HiGNN/V0anom/ablation_graph_02.log 2>&1 &

Resumable: run_seed_study.py skips the seeds and evaluations that are done, so running it again
finishes a study and rewrites the table.
"""

import argparse
import copy
import glob
import os
import subprocess
import sys

import numpy as np

from results_tables import (MODELS, OUT_DIR, ablation_latex, ablation_lead_markdown, ablation_lead_metrics,
                            ablation_markdown, ablation_rows, load_scores)
from utils.utils import read_config, save_config

BASE_LABEL, BASE_DIR = MODELS[0]
STUDY_DIR = "experiments/HiGNN/V0anom/ablation_graph_02"
# name: (table label, datamodule overrides). The cached edge indices do not encode the cluster count, so each
# cluster count needs its own graph dir; the dir comes from graph_embedding_version, which does nothing else
VARIANTS = {
    "oc10": ("10 ocean clusters", {"num_ocean_clusters": 10, "graph_embedding_version": "unitsphere_v2_oc10"}),
    "oc50": ("50 ocean clusters", {"num_ocean_clusters": 50, "graph_embedding_version": "unitsphere_v2_oc50"}),
    "m2m5": ("5 M2M edges", {"k_mesh2mesh": 5}),
    "m2m20": ("20 M2M edges", {"k_mesh2mesh": 20}),
    "m2g1": ("1 M2G edge", {"k_mesh2grid": 1}),
    "m2g5": ("5 M2G edges", {"k_mesh2grid": 5}),
}
SHORT_LABELS = {"10 ocean clusters": "OC = 10", "50 ocean clusters": "OC = 50", "5 M2M edges": "M2M = 5",
                "20 M2M edges": "M2M = 20", "1 M2G edge": "M2G = 1", "5 M2G edges": "M2G = 5"}


def base_config():
    return read_config("configs/hignn_lstm.yaml")


def build_config(base, name, overrides, gpu):
    """Config of one variant; base is not modified."""
    config = copy.deepcopy(base)
    config["datamodule"].update(overrides)
    config["pl"]["devices"] = [gpu]
    config["experiment"]["exp_dir"] = os.path.join(STUDY_DIR, name)
    config["experiment"]["experiment_name"] += f"_{name}"
    return config


def study_rmse(study_dir):
    """Test RMSE [seeds, leads] of a study, or None if it has no manifest or a seed has no scores."""
    if not glob.glob(os.path.join(study_dir, "*_manifest.json")):
        return None
    frames = load_scores(study_dir, "test")
    return None if frames is None else np.stack([f["rmse_latw"].to_numpy() for f in frames])


def lead_latex(rows, caption, label):
    """results_tables.ablation_lead_latex without the Metric column and the p rows, and with SHORT_LABELS, to fit the page."""
    leads = len(rows[0][1])
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{l{'c' * leads}}}", "\\tophline",
             "Model & " + " & ".join(f"L{i + 1}" for i in range(leads)) + " \\\\"]
    for model, metric_rows in ablation_lead_metrics(rows, latex=True):
        model, metric_rows = SHORT_LABELS.get(model, model), metric_rows[:2]
        lines.append("\\middlehline")
        for i, (_, cells) in enumerate(metric_rows):
            name = "" if i else (model if len(metric_rows) == 1 else f"\\multirow{{{len(metric_rows)}}}{{*}}{{{model}}}")
            lines.append(" & ".join([name] + cells) + " \\\\")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def write_tables(n_seeds):
    base = study_rmse(BASE_DIR)
    rmse = {BASE_LABEL: base}
    for name, (label, _) in VARIANTS.items():
        x = study_rmse(os.path.join(STUDY_DIR, name))
        if x is None or len(x) != n_seeds:
            print(f"WARNING: {label} does not have test scores for all {n_seeds} seeds, left out")
            continue
        rmse[label] = x

    rows = ablation_rows({m: {"test": x.mean(1)} for m, x in rmse.items()})
    caption = (f"Graph parameters: area-weighted test RMSE averaged over the leads (mean $\\pm$ standard deviation "
               f"over {n_seeds} seeds, {len(base)} seeds for {BASE_LABEL}), and the difference to {BASE_LABEL} in cm, "
               "with the 95\\,\\% confidence interval and $p$-value of a Welch $t$-test over the seeds. Each variant "
               "changes one parameter of 20 ocean clusters, 10 M2M edges and 3 M2G edges.")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "ablation_graph.tex"), "w") as f:
        f.write(ablation_latex(rows, ["Test"], caption, "tab:ablation_graph"))
    lead_rows = ablation_rows({m: {l: x[:, l] for l in range(base.shape[1])} for m, x in rmse.items()})
    caption = (f"Graph parameters per lead: area-weighted test RMSE of {BASE_LABEL} in cm (mean $\\pm$ standard deviation "
               f"over {len(base)} seeds), and for each variant ({n_seeds} seeds) $\\Delta$RMSE, the difference to "
               f"{BASE_LABEL} in cm, with below it the 95\\,\\% confidence interval of a Welch $t$-test over the seeds. "
               "Each variant changes one parameter of 20 ocean clusters (OC), 10 M2M edges and 3 M2G edges.")
    with open(os.path.join(OUT_DIR, "ablation_graph_per_lead.tex"), "w") as f:
        f.write(lead_latex(lead_rows, caption, "tab:ablation_graph_per_lead"))
    with open(os.path.join(OUT_DIR, "ablation_graph.md"), "w") as f:
        f.write(f"# Graph parameters ({n_seeds} seeds, {len(base)} for {BASE_LABEL}, lead-mean area-weighted RMSE)\n\n"
                f"ΔRMSE: seed-mean RMSE of the variant minus that of {BASE_LABEL}, in cm; > 0 means the variant is "
                "worse. 95 % CI and p from a Welch t-test over the seeds.\n\n" + ablation_markdown(rows, ["Test"])
                + "\n## Per lead, test\n\n" + ablation_lead_markdown(lead_rows))
    print(f"wrote {OUT_DIR}/ablation_graph.tex, {OUT_DIR}/ablation_graph_per_lead.tex, {OUT_DIR}/ablation_graph.md")


def main():
    parser = argparse.ArgumentParser(prog="run_graph_ablation.py", description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", type=int, required=True, help="GPU for training and evaluation")
    parser.add_argument("--seeds", nargs="+", default=["42", "43", "44", "45"])
    args = parser.parse_args()

    base = base_config()
    for name, (_, overrides) in VARIANTS.items():
        config = build_config(base, name, overrides, args.gpu)
        os.makedirs(config["experiment"]["exp_dir"], exist_ok=True)
        path = os.path.join(config["experiment"]["exp_dir"], "config.yaml")
        save_config(config, path)
        subprocess.run([sys.executable, "scripts/run_seed_study.py", "--config_path", path, "--seeds", *args.seeds,
                        "--gpu", str(args.gpu), "--datasets", "test", "--scores_only"])

    write_tables(len(args.seeds))


if __name__ == "__main__":
    main()

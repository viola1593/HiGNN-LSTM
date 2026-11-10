"""N1 results tables: per-lead area-weighted RMSE, anomaly NSE and anomaly r on val, test and GRACE.

Collects the eval_on_{val,test,grace,grace_graceclim}/per_timestep_scores.csv of every seed of each study in MODELS
and writes one LaTeX table per dataset (seed mean +- SD, ddof=1). No model is evaluated here: a study
whose seeds do not all have scores for a dataset is left out of that table, with a warning.

An ablation table (ablation.tex, and ablation.md for reading) compares the lead-mean RMSE of every
other study (ablations and ConvLSTM) on val and test to the first one in MODELS: difference in cm, its
95 % confidence interval and the p-value of a Welch t-test over the seeds. ablation_per_lead_{val,test}.tex
(also in ablation.md) do the same per lead, with the leads as columns.

Two reference rows are added. Climatology: the training climatology's RMSE, read from the same csvs
(its anomaly NSE is 0 by definition and its anomaly r undefined). Anomaly persistence, which no
evaluation stores, is computed here:

    persistence = C[target month] + (TWSA[t-1] - C[month of t-1]),

with C the reconstruction's training climatology, also on GRACE (D4), except on grace_graceclim (GRACE,
with the models run on C_grace instead), where C is C_grace for persistence and climatology alike. It is scored on the forecast
windows and cells of the HiGNN evaluation: its targets are the HiGNN's stored targets (predictions.pt
of the first seed), which the script checks against its own reconstruction, and it uses the
evaluation's metric functions.

Run from the repo root:

    PYTHONPATH=src/ python scripts/results_tables.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.stats import ttest_ind

from utils.eval_metrics import _rmse, per_lead_skill_scores, valid_mask
from utils.utils import read_config

MODELS = [
    ("HiGNN-LSTM", "experiments/HiGNN/V0anom/seeds_era5anom_nores_02"),
    ("No MP", "experiments/HiGNN/V0anom/seeds_era5anom_nores_skipgnn_02"),
    ("No ocean nodes", "experiments/HiGNN/V1anom/seeds_nores_era5anom_02"),
    ("No ERA5", "experiments/HiGNN/V0anom/seeds_nores_noera5_02"),
    ("ConvLSTM", "experiments/baseline2D/anom/seeds_era5anom_nores_02"),
]
METRICS = [("rmse_latw", "RMSE [cm]"), ("nse_deseas_latw", "NSE$_\\mathrm{anom}$"),
           ("corr_deseas_latw", "$r_\\mathrm{anom}$")]
DATASETS = {"val": "Validation", "test": "Test", "grace": "GRACE/-FO",
            "grace_graceclim": "GRACE/-FO (GRACE climatology)"}
GRACE_PATH = "data/processed/GRACE_era5.nc"  # as in TWSA_HeteroGraphDatamodule.GRACE_dataloader
GRACE_CLIM_PATH = "data/processed/GRACE_climatology_pre2008.nc"  # C_grace, scripts/preprocess_data/05_grace_climatology.py
ABLATION_DATASETS = ("val", "test")
LAND_THRESHOLD = 0.8  # TWSA_HeteroGraphDatamodule default
OUT_DIR = "experiments/HiGNN/tables_02"


def sample_starts(times, seq_len, pred_len):
    """First target index t of every window whose months t-seq_len .. t+pred_len-1 are consecutive.

    Mirrors TWSAGraphDataset.get_sample_indices / filter_samples with delay 0.
    """
    months = times.astype("datetime64[M]")
    return np.array([t for t in range(seq_len, len(times) - pred_len + 1)
                     if months[t + pred_len - 1] - months[t - seq_len] == np.timedelta64(seq_len + pred_len - 1, "M")])


def anomaly_persistence(field, clim, months, starts, H):
    """Anomaly persistence [windows, nodes, leads] from `field` [T, N], `clim` [12, N], `months` [T] (0-11)."""
    a0 = field[starts - 1] - clim[months[starts - 1]]  # [S, N]
    target_months = months[starts[:, None] + np.arange(H)]  # [S, H]
    return (clim[target_months] + a0[:, None]).transpose(0, 2, 1)


def climatology(dataset, train):
    """[12, lat, lon] climatology of `dataset`: C_grace on grace_graceclim, else the mean per calendar
    month of the training twsa `train`, NaN -> 0 as in the datamodule."""
    if dataset == "grace_graceclim":
        c = xr.open_dataset(GRACE_CLIM_PATH)["twsa"]
        assert (c.lat.values == train.lat.values).all() and (c.lon.values == train.lon.values).all()
        return c.values
    return train.groupby("time.month").mean("time").fillna(0).astype(np.float32).values


def seed_summary(frames, column):
    """Mean and sample SD (ddof=1) over seeds of one per-lead score column."""
    values = np.stack([f[column].to_numpy() for f in frames])
    return values.mean(0), values.std(0, ddof=1)


def latex_table(rows, caption, label):
    """`rows`: [(model, [(metric label, mean [leads], sd [leads] or None), ...]), ...]."""
    leads = len(rows[0][1][0][1])
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              f"\\begin{{tabular}}{{ll{'c' * leads}}}", "\\tophline",
             "Model & Metric & " + " & ".join(f"L{i + 1}" for i in range(leads)) + " \\\\"]
    for model, metric_rows in rows:
        lines.append("\\middlehline")
        for i, (metric, mean, sd) in enumerate(metric_rows):
            name = "" if i else (model if len(metric_rows) == 1 else f"\\multirow{{{len(metric_rows)}}}{{*}}{{{model}}}")
            cells = [f"${m:.3f}$" if sd is None else f"${m:.3f} \\pm {s:.3f}$"
                     for m, s in zip(mean, [None] * leads if sd is None else sd)]
            lines.append(" & ".join([name, metric] + cells) + " \\\\")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def welch(ablation, full):
    """Welch t-test (unequal variances) of two sets of per-seed scores.

    Returns the mean difference ablation - full, its 95 % confidence interval and the two-sided p-value.
    """
    res = ttest_ind(ablation, full, equal_var=False)
    ci = res.confidence_interval(0.95)
    return ablation.mean() - full.mean(), ci.low, ci.high, res.pvalue


def ablation_rows(lead_means):
    """`lead_means`: {model: {key: RMSE per seed}}, the full model first; a key is a dataset (lead-mean
    RMSE) or a lead.

    Returns [(model, [(mean, sd, None or (delta, CI low, CI high, p)) per key])], the differences to the
    full model in cm.
    """
    (ref_model, ref), *_ = lead_means.items()
    rows = []
    for model, scores in lead_means.items():
        cells = []
        for dataset, x in scores.items():
            s = None if model == ref_model else welch(x, ref[dataset])
            cells.append((x.mean(), x.std(ddof=1), s))
        rows.append((model, cells))
    return rows


def format_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ablation_markdown(rows, titles):
    lines = ["| Variant | " + " | ".join(f"{t} RMSE [cm] | {t} ΔRMSE [cm] (95 % CI) | {t} p" for t in titles) + " |",
             "|---" * (1 + 3 * len(titles)) + "|"]
    for model, cells in rows:
        parts = [model]
        for mean, sd, s in cells:
            parts.append(f"{mean:.3f} ± {sd:.3f}")
            parts += ["–", "–"] if s is None else [f"{s[0]:+.3f} ({s[1]:+.3f}, {s[2]:+.3f})", format_p(s[3])]
        lines.append("| " + " | ".join(parts) + " |")
    return "\n".join(lines) + "\n"


def ablation_latex(rows, titles, caption, label):
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{l{'ccc' * len(titles)}}}", "\\tophline",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{t}}}" for t in titles) + " \\\\",
             "Variant & " + " & ".join(["RMSE [cm] & $\\Delta$RMSE [cm] (95\\,\\% CI) & $p$"] * len(titles)) + " \\\\",
             "\\middlehline"]
    for model, cells in rows:
        parts = [model]
        for mean, sd, s in cells:
            parts.append(f"${mean:.3f} \\pm {sd:.3f}$")
            parts += ["--", "--"] if s is None else [f"${s[0]:+.3f}$ (${s[1]:+.3f}$, ${s[2]:+.3f}$)", f"${format_p(s[3])}$"]
        lines.append(" & ".join(parts) + " \\\\")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def ablation_lead_metrics(rows, latex):
    """`rows`: ablation_rows with one key per lead, the full model first.

    Returns [(model, [(metric, [cell per lead])])]: the full model's RMSE, and ΔRMSE, 95 % CI and p of every
    other model, as markdown or LaTeX strings.
    """
    m = (lambda s: f"${s}$") if latex else str
    (ref, ref_cells), *others = rows
    pm = "\\pm" if latex else "±"
    metrics = [(ref, [("RMSE [cm]", [m(f"{mean:.3f} {pm} {sd:.3f}") for mean, sd, _ in ref_cells])])]
    for model, cells in others:
        s = [c[2] for c in cells]
        metrics.append((model, [
            ("$\\Delta$RMSE [cm]" if latex else "ΔRMSE [cm]", [m(f"{x[0]:+.3f}") for x in s]),
            ("95\\,\\% CI" if latex else "95 % CI", [f"({m(f'{x[1]:+.3f}')}, {m(f'{x[2]:+.3f}')})" for x in s]),
            ("$p$" if latex else "p", [m(format_p(x[3])) for x in s])]))
    return metrics


def ablation_lead_markdown(rows):
    leads = len(rows[0][1])
    lines = ["| Model | Metric | " + " | ".join(f"L{i + 1}" for i in range(leads)) + " |", "|---" * (2 + leads) + "|"]
    for model, metric_rows in ablation_lead_metrics(rows, latex=False):
        for i, (metric, cells) in enumerate(metric_rows):
            lines.append("| " + " | ".join(["" if i else model, metric] + cells) + " |")
    return "\n".join(lines) + "\n"


def ablation_lead_latex(rows, caption, label):
    leads = len(rows[0][1])
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{ll{'c' * leads}}}", "\\tophline",
             "Model & Metric & " + " & ".join(f"L{i + 1}" for i in range(leads)) + " \\\\"]
    for model, metric_rows in ablation_lead_metrics(rows, latex=True):
        lines.append("\\middlehline")
        for i, (metric, cells) in enumerate(metric_rows):
            name = "" if i else (model if len(metric_rows) == 1 else f"\\multirow{{{len(metric_rows)}}}{{*}}{{{model}}}")
            lines.append(" & ".join([name, metric] + cells) + " \\\\")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def load_scores(study_dir, dataset):
    """Per-seed per_timestep_scores of one study, or None if a seed has none for this dataset."""
    manifest = json.load(open(glob.glob(os.path.join(study_dir, "*_manifest.json"))[0]))
    paths = [os.path.join(run, f"eval_on_{dataset}", "per_timestep_scores.csv") for run in manifest.values()]
    if not all(os.path.exists(p) for p in paths):
        return None
    return [pd.read_csv(p) for p in paths]


def persistence_scores(ref_run, dataset):
    """Per-lead anomaly persistence and climatology scores on the cells and windows of the HiGNN run `ref_run`."""
    cfg = read_config(os.path.join(ref_run, "config.yaml"))["datamodule"]
    seq_len, H = cfg["input_seq_len"], cfg["output_seq_len"]
    ds = xr.open_dataset(cfg["ds_path"])

    # periods as in TWSA_HeteroGraphDatamodule.get_train_val_test_periods
    time = ds.time.values
    ntr, nva = int(len(time) * cfg["train_val_test_split"][0]), int(len(time) * cfg["train_val_test_split"][1])
    periods = {"train": (time[0], time[ntr]), "val": (time[ntr + 1], time[ntr + nva]),
               "test": (time[ntr + nva + 1], time[-1])}

    # land nodes in the datamodule's row-major order, minus Antarctica and Hawaii as in the evaluation
    land = (ds["lsm"].isel(time=0) > LAND_THRESHOLD).values
    lat2d, lon2d = np.meshgrid(ds.lat.values, ds.lon.values, indexing="ij")
    lat, lon = lat2d[land], lon2d[land]
    keep = ~(((lon > -161) & (lon < -150) & (lat > 18) & (lat < 23)) | (lat < -60))

    def nodes(da):
        return da.values[:, land][:, keep].astype(np.float64)

    train = ds["twsa"].sel(time=slice(*periods["train"]))
    clim = climatology(dataset, train)[:, land][:, keep]

    if dataset in ("grace", "grace_graceclim"):
        src = xr.open_dataset(GRACE_PATH)
        assert (src.lat.values == ds.lat.values).all() and (src.lon.values == ds.lon.values).all()
        twsa = src["twsa"].sel(time=slice(periods["train"][1], src.time.values[-1]))
    else:
        twsa = ds["twsa"].sel(time=slice(*periods[dataset]))
    field, months = nodes(twsa), twsa.time.dt.month.values - 1
    starts = sample_starts(twsa.time.values, seq_len, H)
    pers = anomaly_persistence(field, clim, months, starts, H)
    clim_t = clim[months[starts[:, None] + np.arange(H)]].transpose(0, 2, 1)

    # score against the HiGNN's own targets, after checking they are the windows reconstructed here
    P = torch.load(os.path.join(ref_run, f"eval_on_{dataset}", "predictions.pt"), weights_only=False)
    y = torch.cat([batch[1].reshape(-1, land.sum(), H) for batch in P])[:, keep]
    truth = field[starts[:, None] + np.arange(H)].transpose(0, 2, 1)
    assert y.shape == truth.shape, (y.shape, truth.shape)
    # the stored targets are NaN where the reconstruction is, also on GRACE; those cells are not scored
    finite = np.isfinite(y.numpy())
    assert np.allclose(y.numpy()[finite], truth[finite], atol=1e-3), f"{dataset}: stored targets differ from the reconstruction"

    w = torch.cos(torch.deg2rad(torch.tensor(lat[keep], dtype=torch.float32)))
    pers_t, clim_tt = torch.tensor(pers, dtype=torch.float32), torch.tensor(clim_t)
    mask = valid_mask(pers_t, clim_tt, y)
    assert torch.equal(mask, valid_mask(clim_tt, y)), f"{dataset}: persistence is not finite on every scored cell"
    skill = per_lead_skill_scores(pers, y.numpy(), clim_t, w.numpy())
    return {"rmse_latw": _rmse(pers_t, y, mask=mask, lat_weights=w).mean(0).numpy(),
            "nse_deseas_latw": skill["nse_deseas_latw"], "corr_deseas_latw": skill["corr_deseas_latw"],
            "clim_rmse_latw": _rmse(clim_tt, y, mask=mask, lat_weights=w).mean(0).numpy()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ref_dir = MODELS[0][1]
    ref_run = list(json.load(open(glob.glob(os.path.join(ref_dir, "*_manifest.json"))[0])).values())[0]
    seed_rmse = {}  # {model: {dataset: [seeds, leads]}}
    for dataset, title in DATASETS.items():
        rows, clim = [], None
        for model, study_dir in MODELS:
            frames = load_scores(study_dir, dataset)
            if frames is None:
                print(f"WARNING: {model} ({study_dir}) has no {dataset} scores for every seed, left out")
                continue
            rows.append((model, [(label, *seed_summary(frames, col)) for col, label in METRICS]))
            if dataset in ABLATION_DATASETS:
                seed_rmse.setdefault(model, {})[dataset] = np.stack([f["rmse_latw"].to_numpy() for f in frames])
            if study_dir == ref_dir:
                clim, n_seeds = frames[0]["clim_rmse_latw"].to_numpy(), len(frames)

        pers = persistence_scores(ref_run, dataset)
        # the climatology of the csvs is scored on the same cells as the persistence
        assert np.allclose(pers["clim_rmse_latw"], clim, atol=1e-4), (pers["clim_rmse_latw"], clim)
        rows.append(("Persistence", [(label, pers[col], None) for col, label in METRICS]))
        rows.append(("Climatology", [(METRICS[0][1], clim, None)]))

        source = ("GRACE/-FO from 2002-04 to 2008-02" if dataset == "grace_graceclim"
                  else "the reconstruction's training period")
        caption = (f"{title} scores per lead time (mean $\\pm$ standard deviation over {n_seeds} seeds). "
                   "RMSE, anomaly NSE and anomaly correlation $r$ are area-weighted. Persistence: the anomaly "
                   "of the last input month added to the climatology of the target month. The climatology is "
                   f"from {source}.")
        path = os.path.join(OUT_DIR, f"results_{dataset}.tex")
        with open(path, "w") as f:
            f.write(latex_table(rows, caption, f"tab:results_{dataset}"))
        print(f"wrote {path}")

    # ablations: only variants with scores on every ablation dataset
    seed_rmse = {m: s for m, s in seed_rmse.items() if len(s) == len(ABLATION_DATASETS)}
    rows = ablation_rows({m: {d: x.mean(1) for d, x in s.items()} for m, s in seed_rmse.items()})
    titles = [DATASETS[d] for d in ABLATION_DATASETS]
    caption = (f"Ablations and ConvLSTM: area-weighted RMSE averaged over leads 1--{len(clim)} (mean $\\pm$ standard "
               f"deviation over {n_seeds} seeds), and the difference to {MODELS[0][0]} in cm, with the "
               "95\\,\\% confidence interval and $p$-value of a Welch $t$-test over the seeds.")
    with open(os.path.join(OUT_DIR, "ablation.tex"), "w") as f:
        f.write(ablation_latex(rows, titles, caption, "tab:ablation"))
    md = (f"# Ablations and ConvLSTM ({n_seeds} seeds, lead-mean area-weighted RMSE)\n\n"
          f"ΔRMSE: seed-mean RMSE of the model minus that of {MODELS[0][0]}, in cm; > 0 means the model is worse. "
          "95 % CI and p from a Welch t-test over the seeds. p < 0.05 (CI excludes 0): the difference is "
          "unlikely to be seed noise.\n\n" + ablation_markdown(rows, titles))

    # per lead, one table per dataset
    for dataset, title in zip(ABLATION_DATASETS, titles):
        rows = ablation_rows({m: {l: s[dataset][:, l] for l in range(len(clim))} for m, s in seed_rmse.items()})
        caption = (f"Ablations and ConvLSTM per lead, {title.lower()}: area-weighted RMSE of {MODELS[0][0]} (mean "
                   f"$\\pm$ standard deviation over {n_seeds} seeds) and the difference of each model to it in cm, "
                   "with the 95\\,\\% confidence interval and $p$-value of a Welch $t$-test over the seeds.")
        with open(os.path.join(OUT_DIR, f"ablation_per_lead_{dataset}.tex"), "w") as f:
            f.write(ablation_lead_latex(rows, caption, f"tab:ablation_per_lead_{dataset}"))
        md += f"\n## Per lead, {title.lower()}\n\n" + ablation_lead_markdown(rows)

    with open(os.path.join(OUT_DIR, "ablation.md"), "w") as f:
        f.write(md)
    print(f"wrote {OUT_DIR}/ablation*.tex, {OUT_DIR}/ablation.md")


if __name__ == "__main__":
    main()

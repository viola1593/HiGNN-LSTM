"""Per-basin anomaly skill of HiGNN-LSTM (anomaly target) vs. climatology and anomaly persistence, per lead.

Everything is computed on area-weighted basin means over the test starts:
- ACC = corr(pred - C, truth - C), C = training climatology of the target month (as the datamodule's
  twsa_climatology)
- skill = 1 - RMSE_model / RMSE_ref, ref = climatology or anomaly persistence (C + anomaly of the last
  input month)
Model values are seed means. Intervals are 90 % moving-block bootstraps over the forecast starts
(12-month blocks), with the same resampled starts for every seed and reference.
No training and no inference: the cached eval_on_{dataset}/predictions.pt of each run are rescored.
On GRACE (--dataset grace) truth and persistence come from GRACE_PATH from the end of training on, C stays the
reconstruction's training climatology (as results_tables.py), and windows across the GRACE gap are skipped.
Run from the repo root: python scripts/basin_anomaly_skill.py [--dataset grace]
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

STUDY = ("HiGNN-LSTM", "experiments/HiGNN/V0anom/seeds_era5anom_nores_02")
DS_PATH = "data/processed/02_RecTWSA_era5.nc"
GRACE_PATH = "data/processed/GRACE_era5.nc"  # as in TWSA_HeteroGraphDatamodule.GRACE_dataloader
DATASET_NAMES = {"test": "the test set", "grace": "GRACE/-FO"}
LAND_CSV = "graphs/full_unitsphere_v2/landpoints_to_basin.csv"
OUT = "experiments/HiGNN/V0anom/seeds_era5anom_nores_02/basin_anomaly_skill"
SPLIT = [0.7, 0.2, 0.1]
SEQ_LEN, PRED_LEN = 12, 6
BLOCK, N_BOOT, CI = 12, 1000, (5, 95)
# colorblind-safe blue of the repo's ablation figure for the model, neutral ink for the reference
MODEL_COLOR, PERS_COLOR = "#2a78d6", "#52514e"
TABLE_METRICS = [("acc", "$r_\\mathrm{anom}$"), ("skill_pers", "SS$_\\mathrm{pers}$"),
                 ("skill_clim", "SS$_\\mathrm{clim}$")]
TABLE_CAPTION = ("Anomaly skill of HiGNN-LSTM for the 16 test basins per lead time on {data} (mean over 8 seeds, "
                 "{n_starts} forecast starts), computed on area-weighted basin means. $r_\\mathrm{anom}$: anomaly correlation; "
                 "SS$_\\mathrm{pers}$ and SS$_\\mathrm{clim}$: skill score $1 - \\mathrm{RMSE}/\\mathrm{RMSE}_\\mathrm{ref}$ "
                 "against anomaly persistence and the training climatology, positive where the model is better. "
                 "The scores of all basins, with 90\\,\\% bootstrap intervals, are in the Supplement.")


def out_dir(dataset):
    return OUT if dataset == "test" else f"{OUT}_{dataset}"


def start_dates(times, seq_len=SEQ_LEN, pred_len=PRED_LEN):
    """First forecast month of every sample, as TWSAGraphHeteroData.get_sample_indices with delay=0."""
    months = np.asarray(times).astype("datetime64[M]")
    assert np.all(np.diff(months).astype(int) == 1), "the sample rule below assumes a gapless monthly axis"
    return np.asarray(times)[seq_len:len(times) - pred_len + 1]


def last_input_anomaly(anom, times, starts):
    """anom [T, N] on the monthly axis `times` -> anomaly of the month before each start [B, N]."""
    times, starts = np.asarray(times), np.asarray(starts)
    pos = np.searchsorted(times, starts)
    assert (times[pos] == starts).all(), "every start must be on the time axis"
    return anom[pos - 1]


def persistence_forecast(clim_target, a0):
    """C of each target month [B, 6, N] plus the last observed anomaly [B, N], held over all leads."""
    return clim_target + a0[:, None]


def rmse(pred, truth):
    """[B, 6, K] -> [6, K] over the forecast starts."""
    return np.sqrt(np.mean((pred - truth) ** 2, axis=0))


def anomaly_correlation(pred, truth, clim):
    """Pearson correlation of pred - C and truth - C over the forecast starts, [B, 6, K] -> [6, K]."""
    pa, ta = pred - clim, truth - clim
    pa, ta = pa - pa.mean(0), ta - ta.mean(0)
    return (pa * ta).sum(0) / np.sqrt((pa ** 2).sum(0) * (ta ** 2).sum(0))


def skill_score(rmse_model, rmse_ref):
    return 1 - rmse_model / rmse_ref


def moving_block_indices(n, block, n_boot, rng):
    """[n_boot, n] resampled indices, each row ceil(n / block) random contiguous blocks, cut to n."""
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    return (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n]


def scores(preds, truth, clim, pers):
    """preds [S, B, 6, K] -> metric name -> [6, K]; model metrics are seed means."""
    rc, rp = rmse(clim, truth), rmse(pers, truth)
    rm = np.stack([rmse(p, truth) for p in preds])
    return {"acc": np.mean([anomaly_correlation(p, truth, clim) for p in preds], axis=0),
            "skill_clim": skill_score(rm, rc).mean(0),
            "skill_pers": skill_score(rm, rp).mean(0),
            "acc_persistence": anomaly_correlation(pers, truth, clim),
            "skill_clim_persistence": skill_score(rp, rc)}


def load_predictions(run_dir, keep, num_nodes, dataset="test"):
    """y_hat, y_true of one run on `dataset`, [B, 6, N] per land cell."""
    import torch

    preds = torch.load(os.path.join(run_dir, f"eval_on_{dataset}", "predictions.pt"), weights_only=False)
    out = [torch.cat([b[i].reshape(-1, num_nodes, PRED_LEN) for b in preds]).float().numpy()[:, keep]
           for i in (0, 1)]
    return [v.transpose(0, 2, 1) for v in out]


def main(dataset="test"):
    import xarray as xr
    from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
    from results_tables import sample_starts
    from utils.eval_metrics import _grouped_nanmean, basin_grouping

    out = out_dir(dataset)
    os.makedirs(out, exist_ok=True)

    # land cells in node order, with the evaluation's Antarctica/Hawaii mask
    land = pd.read_csv(LAND_CSV)
    lon_all, lat_all = land["lon"].to_numpy(), land["lat"].to_numpy()
    hawaii = (lon_all > -161) & (lon_all < -150) & (lat_all > 18) & (lat_all < 23)
    keep = ~(hawaii | (lat_all < -60))
    num_nodes = len(land)
    lat_weights = np.cos(np.deg2rad(lat_all[keep]))
    dm = SimpleNamespace(get_antarctica_landmask=lambda: ~keep,
                         data_pred=SimpleNamespace(land_to_basin_ids=land["HYBAS_ID"].astype("int64")))
    grouping = basin_grouping(dm)
    basin_ids = grouping[0]

    ds = xr.open_dataset(DS_PATH)
    periods = TWSA_HeteroGraphDatamodule.get_train_val_test_periods(
        SimpleNamespace(ds=SimpleNamespace(time=ds.time), train_val_test_split=SPLIT))

    def nodes(src):
        return src["twsa"].sel(lat=xr.DataArray(lat_all[keep], dims="node"),
                               lon=xr.DataArray(lon_all[keep], dims="node")).load()  # [T, N]

    tw = nodes(ds)
    clim = tw.sel(time=slice(*periods["train"])).groupby("time.month").mean("time").fillna(0).values  # [12, N]
    if dataset == "grace":
        # the period of TWSA_HeteroGraphDatamodule.GRACE_dataloader: end of training to the last GRACE-FO month
        tw = nodes(xr.open_dataset(GRACE_PATH)).sel(time=slice(periods["train"][1], None))
    else:
        tw = tw.sel(time=slice(*periods["test"]))
    ds_times = tw.time.values
    month = tw.time.dt.month.values - 1
    twv = tw.values.astype(np.float64)
    anom = twv - clim[month]

    starts = ds_times[sample_starts(ds_times, SEQ_LEN, PRED_LEN)]  # skips windows across the GRACE gap
    tt = np.searchsorted(ds_times, starts)[:, None] + np.arange(PRED_LEN)  # [B, 6] target-month indices
    truth = twv[tt]
    clim_t = clim[month[tt]]
    a0 = last_input_anomaly(anom, ds_times, starts)
    pers = persistence_forecast(clim_t, a0)
    invalid = ~np.isfinite(truth) | ~np.isfinite(a0)[:, None]  # score every forecast on the same cells

    manifest = json.load(open([os.path.join(STUDY[1], f) for f in os.listdir(STUDY[1])
                               if f.endswith("_manifest.json")][0]))
    runs = {seed: load_predictions(run_dir, keep, num_nodes, dataset) for seed, run_dir in manifest.items()}
    # the stored targets are NaN where the reconstruction is, also on GRACE; those cells are not scored
    invalid |= ~np.isfinite(next(iter(runs.values()))[1])
    preds = []
    for seed, (y_hat, y_true) in runs.items():
        assert y_hat.shape == truth.shape, (y_hat.shape, truth.shape)
        ok = ~invalid
        err = np.abs(y_true[ok] - truth[ok]).max()
        print(f"seed {seed}: max |y_true - twsa| on scored cells {err:.2e} cm", flush=True)
        assert err < 1e-2, "predictions.pt targets do not line up with the dataset"
        preds.append(np.where(invalid, np.nan, y_hat))

    def basin_mean(x):
        return _grouped_nanmean(np.where(invalid, np.nan, x), grouping, lat_weights).astype(np.float64)

    truth_b, clim_b, pers_b = basin_mean(truth), basin_mean(clim_t), basin_mean(pers)
    preds_b = np.stack([basin_mean(p) for p in preds])  # [S, B, 6, K]

    point = scores(preds_b, truth_b, clim_b, pers_b)
    idx = moving_block_indices(len(starts), BLOCK, N_BOOT, np.random.default_rng(0))
    boot = [scores(preds_b[:, i], truth_b[i], clim_b[i], pers_b[i]) for i in idx]
    ci = {k: np.percentile([b[k] for b in boot], CI, axis=0) for k in point}  # [2, 6, K]

    from evaluation_HiGNN import TEST_BASIN_IDS
    names = {v: k for k, v in TEST_BASIN_IDS.items()}
    rows = [dict(basin_id=int(b), basin=names.get(int(b), ""), lead=lead + 1, metric=k,
                 value=point[k][lead, j], ci_lo=ci[k][0, lead, j], ci_hi=ci[k][1, lead, j])
            for k in point for lead in range(PRED_LEN) for j, b in enumerate(basin_ids)]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out, "basin_anomaly_skill.csv"), index=False)
    write_summary(table, len(starts), starts, dataset)
    with open(os.path.join(out, "test_basins.tex"), "w") as f:
        f.write(latex_table(table, list(TEST_BASIN_IDS), len(starts), dataset))

    col = {int(b): j for j, b in enumerate(basin_ids)}
    for metric, ref, ylabel, name in (("acc", "acc_persistence", "Anomaly correlation", "acc"),
                                      ("skill_clim", "skill_clim_persistence",
                                       "Skill vs. climatology (1 - RMSE/RMSE$_{clim}$)", "skill_clim")):
        plot_panels(point, ci, metric, ref, col, TEST_BASIN_IDS, ylabel,
                    os.path.join(out, f"{name}_test_basins"))


def write_summary(table, n_starts, starts, dataset="test"):
    t = table[table.metric == "skill_pers"].dropna(subset=["value"])
    test = t[t.basin != ""]
    lines = [f"# Per-basin anomaly skill, {STUDY[0]} ({STUDY[1]}), {DATASET_NAMES[dataset]}", "",
             f"{n_starts} forecast starts ({pd.Timestamp(starts[0]):%Y-%m} to {pd.Timestamp(starts[-1]):%Y-%m}), "
             f"{N_BOOT} moving-block bootstraps ({BLOCK}-month blocks), {CI[1] - CI[0]} % intervals.", "",
             "## Skill vs. anomaly persistence: basins where the model is better", "",
             "| lead | all basins: skill > 0 | all: CI > 0 | all: CI < 0 | 16 test basins: skill > 0 | test: CI > 0 |",
             "|---|---|---|---|---|---|"]
    for lead in range(1, PRED_LEN + 1):
        a, s = t[t.lead == lead], test[test.lead == lead]
        lines.append(f"| {lead} | {(a.value > 0).sum()}/{len(a)} | {(a.ci_lo > 0).sum()} | {(a.ci_hi < 0).sum()} | "
                     f"{(s.value > 0).sum()}/{len(s)} | {(s.ci_lo > 0).sum()} |")
    lines += ["", "## Median over all basins", "", "| lead | ACC model | ACC persistence | skill vs clim model | "
              "skill vs clim persistence |", "|---|---|---|---|---|"]
    med = table.groupby(["metric", "lead"]).value.median()
    for lead in range(1, PRED_LEN + 1):
        lines.append(f"| {lead} | {med['acc', lead]:.2f} | {med['acc_persistence', lead]:.2f} | "
                     f"{med['skill_clim', lead]:.2f} | {med['skill_clim_persistence', lead]:.2f} |")
    leads = range(1, PRED_LEN + 1)
    lines += ["", "all basins, skill > 0 (whole interval > 0) and medians", "",
              "| | " + " | ".join(f"L{lead}" for lead in leads) + " |", "|---" * (PRED_LEN + 1) + "|"]
    for label, metric in (("persistence", "skill_pers"), ("climatology", "skill_clim")):
        s = table[table.metric == metric].dropna(subset=["value"])
        lines.append(f"| Basins better than {label} (interval > 0) | " + " | ".join(
            f"{(s[s.lead == lead].value > 0).sum()} ({(s[s.lead == lead].ci_lo > 0).sum()})" for lead in leads) + " |")
    for label, metric in (("anomaly r", "acc"), ("skill vs climatology", "skill_clim"),
                          ("skill vs persistence", "skill_pers")):
        lines.append(f"| Median {label}, model | " + " | ".join(f"{med[metric, lead]:.2f}" for lead in leads) + " |")
    with open(os.path.join(out_dir(dataset), "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def latex_table(table, basins, n_starts, dataset="test"):
    """TABLE_METRICS of `basins` from the long table of main, with the leads as columns."""
    values = table[table.basin.isin(basins)].set_index(["basin", "metric", "lead"]).value.sort_index()
    leads = range(1, PRED_LEN + 1)
    # str.replace, not str.format: the caption's LaTeX has braces of its own
    caption = TABLE_CAPTION.replace("{data}", DATASET_NAMES[dataset]).replace("{n_starts}", str(n_starts))
    label = "tab:basin_anomaly_skill" + ("" if dataset == "test" else f"_{dataset}")
    lines = ["\\begin{table}[htbp]", "\\centering", "\\renewcommand{\\baselinestretch}{1}\\scriptsize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{ll{'c' * PRED_LEN}}}", "\\tophline",
             "Basin & Metric & " + " & ".join(f"L{lead}" for lead in leads) + " \\\\"]
    for basin in basins:
        lines.append("\\middlehline")
        for i, (metric, label) in enumerate(TABLE_METRICS):
            name = "" if i else f"\\multirow{{{len(TABLE_METRICS)}}}{{*}}{{{basin}}}"
            lines.append(" & ".join([name, label] + [f"${values[basin, metric, lead]:.2f}$" for lead in leads]) + " \\\\")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def plot_panels(point, ci, metric, ref, col, basins, ylabel, path):
    import matplotlib.pyplot as plt

    leads = np.arange(1, PRED_LEN + 1)
    fig, axes = plt.subplots(4, 4, figsize=(9, 8), sharex=True, sharey=True)
    for ax, (name, bid) in zip(axes.flat, basins.items()):
        j = col[bid]
        ax.axhline(0, color=PERS_COLOR, lw=0.8)
        ax.fill_between(leads, ci[metric][0, :, j], ci[metric][1, :, j], color=MODEL_COLOR, alpha=0.2, lw=0)
        ax.plot(leads, point[metric][:, j], "-o", color=MODEL_COLOR, lw=1.5, ms=3.5, label=STUDY[0])
        ax.plot(leads, point[ref][:, j], "--o", color=PERS_COLOR, lw=1.2, ms=3,
                mfc="white", label="Anomaly persistence")
        ax.set_title(name, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(leads)
    for ax in axes[-1]:
        ax.set_xlabel("Lead time [months]")
    fig.supylabel(ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.insert(1, plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR, alpha=0.2, lw=0))
    labels.insert(1, f"{CI[1] - CI[0]} % bootstrap interval")
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    
    fig.savefig(f"{path}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"saved {path}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_NAMES), default="test")
    main(parser.parse_args().dataset)

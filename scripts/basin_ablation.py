"""Per-basin ablation table on the test period: RMSE change of every HiGNN ablation vs the full model.

For each ablation in results_tables.MODELS (ConvLSTM left out, it is not on the land-node graph), basin and lead,
the basin-mean anomaly forecast of every seed is scored against the reconstruction (RMSE over the test starts,
masks and area weights as scripts/basin_anomaly_skill.py). The table gives
    L1..L6  seed-mean RMSE(ablation) - seed-mean RMSE(full) [cm]
    Mean    the same for the lead-mean RMSE
    p       Welch t-test of the per-seed lead-mean RMSE (results_tables.welch), uncorrected.
Basins with a missing score are dropped. Writes basin_ablation.md (full table plus the basins with p < 0.05),
basin_ablation.tex (full table) and basin_ablation.csv (for scripts/plotting/plot_basin_ablation.py) to
results_tables.OUT_DIR.

Run from the repo root:

    PYTHONPATH=src/ python scripts/basin_ablation.py
"""
import json
import os

import numpy as np
import pandas as pd

from results_tables import MODELS as ALL_MODELS, OUT_DIR, welch

MODELS = [m for m in ALL_MODELS if m[0] != "ConvLSTM"]
SEEDS = [str(s) for s in range(42, 50)]
CONTINENTS = {1: "Africa", 2: "Europe", 3: "Siberia", 4: "Asia", 5: "Australasia", 6: "S America",
              7: "N America", 8: "Arctic NA", 9: "Greenland"}  # first digit of the HydroBASINS ID


def basin_rows(rmse, models, basin_ids, names):
    """`rmse` [models, seeds, leads, basins], the full model first; `names` {basin ID: name}
    -> one row per ablation and basin."""
    ok = np.isfinite(rmse).all(axis=(0, 1, 2))
    rmse, basin_ids = rmse[..., ok], basin_ids[ok]
    full = rmse[0]
    rows = []
    for m in range(1, len(models)):
        for k, basin in enumerate(basin_ids):
            row = dict(model=models[m], basin=int(basin), name=names.get(int(basin), ""),
                       continent=CONTINENTS[int(str(basin)[0])])
            for lead in range(rmse.shape[2]):
                row[f"L{lead + 1}"] = rmse[m, :, lead, k].mean() - full[:, lead, k].mean()
            diff, _, _, p = welch(rmse[m, :, :, k].mean(1), full[:, :, k].mean(1))
            row.update(Mean=diff, p=p)
            rows.append(row)
    return pd.DataFrame(rows)


def significant(df):
    """Rows with lead-mean p < 0.05, by model and Mean."""
    return df[df["p"] < 0.05].sort_values(["model", "Mean"])


def _leads(df):
    return [c for c in df.columns if c.startswith("L") and c[1:].isdigit()]


def markdown(df):
    cols = ["basin", "name", "continent"] + _leads(df) + ["Mean", "p"]
    fmt = df.copy()
    for c in _leads(df) + ["Mean"]:
        fmt[c] = fmt[c].map(lambda v: f"{v:+.3f}")
    fmt["p"] = fmt["p"].map(lambda v: f"{v:.3f}")

    def table(sub):
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                         + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in sub.iterrows()])

    n_basins = df.groupby("model", sort=False).size().iloc[0]
    out = [f"# Per-basin ablation, test period\n\nΔRMSE = ablation − full [cm], seed means of 8 seeds, "
           f"{n_basins} basins. p: Welch t-test of the per-seed lead-mean RMSE, uncorrected "
           f"(~{0.05 * n_basins:.0f} basins per ablation expected below 0.05 by chance).\n",
           "## Basins with lead-mean p < 0.05\n"]
    sig = significant(df)
    for model in df["model"].unique():
        sub = fmt.loc[sig.index[sig["model"] == model]]
        out += [f"### {model} ({len(sub)} basins: {(sig.loc[sub.index, 'Mean'] > 0).sum()} worse, "
                f"{(sig.loc[sub.index, 'Mean'] < 0).sum()} better)\n", table(sub) + "\n"]
    out.append("## All basins\n")
    for model in df["model"].unique():
        out += [f"### {model}\n", table(fmt[df["model"] == model]) + "\n"]
    return "\n".join(out)


def latex(df):
    leads = _leads(df)
    ncol = 3 + len(leads) + 2
    lines = ["\\begin{longtable}{lll" + "r" * (len(leads) + 2) + "}",
             "\\caption{Per-basin RMSE change of each ablation vs.\\ the full model on the test period "
             "(seed means of 8 seeds, cm). p: Welch t-test of the per-seed lead-mean RMSE, uncorrected.}"
             "\\label{tab:basin_ablation}\\\\",
             "\\tophline",
             "Basin & Name & Continent & " + " & ".join(leads) + " & Mean & p \\\\",
             "\\middlehline", "\\endhead"]
    for model in df["model"].unique():
        lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\textbf{{{model}}}}} \\\\")
        for _, r in df[df["model"] == model].iterrows():
            lines.append(f"{r['basin']} & {r['name']} & {r['continent']} & "
                         + " & ".join(f"{r[c]:+.3f}" for c in leads)
                         + f" & {r['Mean']:+.3f} & {r['p']:.3f} \\\\")
    lines += ["\\bottomhline", "\\end{longtable}"]
    return "\n".join(lines) + "\n"


def basin_rmse():
    """[models, seeds, leads, basins] RMSE of the basin-mean forecast on the test period, and the basin IDs."""
    from types import SimpleNamespace

    import xarray as xr
    from basin_anomaly_skill import DS_PATH, LAND_CSV, PRED_LEN, SPLIT, load_predictions, start_dates
    from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
    from utils.eval_metrics import _grouped_nanmean, basin_grouping

    land = pd.read_csv(LAND_CSV)
    lon_all, lat_all = land["lon"].to_numpy(), land["lat"].to_numpy()
    hawaii = (lon_all > -161) & (lon_all < -150) & (lat_all > 18) & (lat_all < 23)
    keep = ~(hawaii | (lat_all < -60))
    lat_weights = np.cos(np.deg2rad(lat_all[keep]))
    dm = SimpleNamespace(get_antarctica_landmask=lambda: ~keep,
                         data_pred=SimpleNamespace(land_to_basin_ids=land["HYBAS_ID"].astype("int64")))
    grouping = basin_grouping(dm)

    ds = xr.open_dataset(DS_PATH)
    periods = TWSA_HeteroGraphDatamodule.get_train_val_test_periods(
        SimpleNamespace(ds=SimpleNamespace(time=ds.time), train_val_test_split=SPLIT))
    twv = ds["twsa"].sel(lat=xr.DataArray(lat_all[keep], dims="node"),
                         lon=xr.DataArray(lon_all[keep], dims="node")).values.astype(np.float64)
    starts = start_dates(ds.time.sel(time=slice(*periods["test"])).values)
    truth = twv[np.searchsorted(ds.time.values, starts)[:, None] + np.arange(PRED_LEN)]
    invalid = ~np.isfinite(truth)
    truth_b = _grouped_nanmean(truth, grouping, lat_weights).astype(np.float64)

    rm = np.full((len(MODELS), len(SEEDS), PRED_LEN, len(grouping[0])), np.nan)
    for m, (model, study) in enumerate(MODELS):
        manifest = json.load(open([os.path.join(study, f) for f in os.listdir(study)
                                   if f.endswith("_manifest.json")][0]))
        assert sorted(manifest) == SEEDS, (model, sorted(manifest))
        for s, seed in enumerate(SEEDS):
            y_hat, y_true = load_predictions(manifest[seed], keep, len(land), "test")
            assert y_hat.shape == truth.shape, (y_hat.shape, truth.shape)
            err = np.abs(y_true[~invalid] - truth[~invalid]).max()
            assert err < 1e-2, (model, seed, err)
            pred_b = _grouped_nanmean(np.where(invalid, np.nan, y_hat), grouping, lat_weights).astype(np.float64)
            rm[m, s] = np.sqrt(np.nanmean((pred_b - truth_b) ** 2, axis=0))
            print(model, seed, f"err {err:.1e} median basin RMSE {np.nanmedian(rm[m, s]):.3f}", flush=True)
    return rm, grouping[0]


def main():
    from evaluation_HiGNN import TEST_BASIN_IDS

    rm, basin_ids = basin_rmse()
    names = {v: k for k, v in TEST_BASIN_IDS.items()}
    df = basin_rows(rm, [m for m, _ in MODELS], basin_ids, names)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "basin_ablation.md"), "w") as f:
        f.write(markdown(df))
    with open(os.path.join(OUT_DIR, "basin_ablation.tex"), "w") as f:
        f.write(latex(df))
    df.to_csv(os.path.join(OUT_DIR, "basin_ablation.csv"), index=False)  # for scripts/plotting/plot_basin_ablation.py
    print(f"wrote {OUT_DIR}/basin_ablation.md, .tex, .csv")


if __name__ == "__main__":
    main()

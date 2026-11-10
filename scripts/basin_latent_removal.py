"""Per-basin input removal: does a latent channel matter for some basins only, which the global score would hide?

Same removals and fills as scripts/run_latent_removal.py (one LSTM input at a time replaced by its train-period
mean, fixed trained model), scored per basin as scripts/basin_ablation.py: RMSE of the area-weighted basin-mean
forecast against the basin-mean truth over the test starts. Basins with a missing score are dropped. One seed per
run, because channels do not match across seeds. ΔRMSE is not normalised, as in basin_ablation, so basins with large
TWSA variability get larger Δ from any removal; this raises the correlation between the maps, so a low correlation
is strong evidence for channels that serve different basins and a high one is ambiguous.

The table gives per removal the global lead-mean ΔRMSE (latent_removal_03/scores.csv, which this run has to
reproduce), the median basin lead-mean ΔRMSE and the number of basins that get better, and the Spearman correlation
of the per-basin lead-mean ΔRMSE between all removals. TWSA history and lead embedding are references: both hurt
everywhere, so a channel's correlation with them shows how much of its map is where any removal hurts.

Scores are kept in STUDY_DIR/basin_scores_seed<seed>.csv: delete it to score again.
Run from the repo root:

    PYTHONPATH=src/ python scripts/basin_latent_removal.py --gpu 1 --seed 45
"""
import argparse
import glob
import json
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import run_latent_removal as rlr
from results_tables import OUT_DIR
from utils.eval_metrics import _grouped_nanmean, basin_grouping

LAND_CSV = "graphs/full_unitsphere_v2/landpoints_to_basin.csv"
COLUMNS = ["seed", "removal", "basin", "lead", "rmse"]


def basin_rmse(pred, true, grouping, lat_weights):
    """`pred`, `true` [windows, cells, leads] -> RMSE of the basin-mean forecast [leads, basins]."""
    pred, true = (np.asarray(v, dtype=np.float64).transpose(0, 2, 1) for v in (pred, true))
    pred_b = _grouped_nanmean(np.where(np.isfinite(true), pred, np.nan), grouping, lat_weights)
    true_b = _grouped_nanmean(true, grouping, lat_weights)
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean((pred_b - true_b) ** 2, axis=0))


def lead_mean_deltas(scores):
    """Long scores of one seed -> lead-mean ΔRMSE against no removal, rows basins, columns removals as rlr.LABELS."""
    rmse = scores.set_index(["basin", "removal", "lead"])["rmse"].unstack(["removal", "lead"])
    deltas = pd.DataFrame({r: (rmse[r] - rmse["none"]).mean(axis=1, skipna=False) for r in rlr.LABELS})
    return deltas.dropna()


def spearman(deltas):
    return deltas.corr(method="spearman")


def markdown(deltas, global_deltas, seed):
    """`deltas` from lead_mean_deltas, `global_deltas` the global lead-mean ΔRMSE per removal."""
    labels = [rlr.LABELS[r] for r in deltas.columns]
    n = len(deltas)
    out = [f"# Per-basin input removal, test period, seed {seed}\n",
           f"Lead-mean ΔRMSE in cm against the same checkpoint with nothing removed, {n} basins (basin-mean forecast "
           "as basin_ablation, not normalised). One LSTM input at a time is replaced by its training-period mean, as "
           "latent_removal.md. Global: lead mean of the area-weighted grid-cell ΔRMSE.\n",
           "| Removed | Global ΔRMSE | Median basin ΔRMSE | Basins with Δ < 0 |", "|---|---|---|---|"]
    for r, label in zip(deltas.columns, labels):
        out.append(f"| {label} | {global_deltas[r]:+.3f} | {deltas[r].median():+.3f} | {(deltas[r] < 0).sum()} / {n} |")
    corr = spearman(deltas).to_numpy()
    out += ["\n## Spearman correlation of the per-basin lead-mean ΔRMSE\n",
            "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    out += [f"| {label} | " + " | ".join(f"{x:.2f}" for x in row) + " |" for label, row in zip(labels, corr)]
    return "\n".join(out) + "\n"


def score_seed(run, seed, gpu):
    """Long rows of COLUMNS: basin RMSE per lead of every removal and of no removal."""
    from basin_anomaly_skill import load_predictions

    trainer, dm, model, _ = rlr.load_run(run, gpu)
    rnn = model.model.rnn
    n_nodes, horizon = dm.data_train.num_land_nodes(), dm.target_seq_len
    keep = ~dm.get_antarctica_landmask()
    lat = np.asarray(dm.data_train.land[0][:, 1][keep], dtype=np.float64)
    lat_weights = np.cos(np.deg2rad(lat))
    land = pd.read_csv(LAND_CSV)
    assert len(land) == n_nodes and np.allclose(land["lat"].to_numpy()[keep], lat)
    grouping = basin_grouping(SimpleNamespace(get_antarctica_landmask=dm.get_antarctica_landmask,
                                              data_pred=SimpleNamespace(land_to_basin_ids=land["HYBAS_ID"].astype("int64"))))

    record = rlr.RecordMeans(horizon, n_nodes, dm.input_seq_len, rnn.input_size)
    handle = rnn.register_forward_pre_hook(rlr.on_lstm_input(record))
    trainer.predict(model, dm.predict_dataloader(period=dm.train_val_test_periods["train"]), return_predictions=False)
    handle.remove()
    fill = rlr.channel_mean(record.cell_mean())
    columns = rlr.lstm_input_columns(model.model)

    loader = dm.predict_dataloader(period=dm.train_val_test_periods["test"])
    stored_global = pd.read_csv(os.path.join(rlr.STUDY_DIR, "scores.csv"), dtype={"seed": str})
    stored_global = stored_global[stored_global["seed"] == seed]

    def predict():
        out = trainer.predict(model, loader)
        return [torch.cat([b[i].reshape(-1, n_nodes, horizon) for b in out])[:, keep].float() for i in (0, 1)]

    rows = []
    for removal in ["none", *rlr.LABELS]:
        handle = None if removal == "none" else rnn.register_forward_pre_hook(
            rlr.on_lstm_input(rlr.ReplaceChannels(fill, columns[removal])))
        pred, true = predict()
        if handle is not None:
            handle.remove()
        stored = stored_global[stored_global["removal"] == removal].sort_values("lead")["rmse"].to_numpy()
        glob_rmse = rlr.rmse_per_lead(pred, true, torch.tensor(lat_weights, dtype=torch.float32)).numpy()
        assert np.allclose(glob_rmse, stored, atol=1e-3), (removal, glob_rmse, stored)
        rmse = basin_rmse(pred.numpy(), true.numpy(), grouping, lat_weights)
        if removal == "none":  # as basin_ablation's full model, from the stored predictions
            y_hat, y_true = load_predictions(run, keep, n_nodes, "test")
            ref = basin_rmse(y_hat.transpose(0, 2, 1), y_true.transpose(0, 2, 1), grouping, lat_weights)
            assert np.allclose(rmse, ref, atol=1e-3, equal_nan=True), np.nanmax(np.abs(rmse - ref))
        rows += [(seed, removal, int(b), lead, float(rmse[lead, k]))
                 for lead in range(horizon) for k, b in enumerate(grouping[0])]
        print(removal, f"global lead mean {glob_rmse.mean():.3f}, median basin {np.nanmedian(rmse):.3f}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(prog="basin_latent_removal.py", description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", type=int, required=True, help="GPU for the predictions")
    parser.add_argument("--seed", default=str(rlr.TABLE_SEED), choices=rlr.SEEDS)
    args = parser.parse_args()

    path = os.path.join(rlr.STUDY_DIR, f"basin_scores_seed{args.seed}.csv")
    if os.path.exists(path):
        scores = pd.read_csv(path, dtype={"seed": str})
    else:
        runs = json.load(open(glob.glob(os.path.join(rlr.SEED_STUDY, "*_manifest.json"))[0]))
        scores = pd.DataFrame(score_seed(runs[args.seed], args.seed, args.gpu), columns=COLUMNS)
        scores.to_csv(path, index=False)

    global_scores = pd.read_csv(os.path.join(rlr.STUDY_DIR, "scores.csv"), dtype={"seed": str})
    global_deltas = rlr.lead_deltas(global_scores).loc[args.seed].mean(axis=1)
    out = os.path.join(OUT_DIR, f"basin_latent_removal_seed{args.seed}.md")
    with open(out, "w") as f:
        f.write(markdown(lead_mean_deltas(scores), global_deltas, args.seed))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

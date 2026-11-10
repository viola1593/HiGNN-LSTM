"""Input removal: how much does the LSTM rely on each of its inputs, the four GNN latent channels, the
TWSA history and the lead embedding? Measured on a fixed trained model, so this is the contribution of an input given
that model, not what a retrained model would lose, and it says nothing about what a latent channel represents
physically. It is not comparable to the retrained ablations.

A forward pre-hook on the LSTM replaces one input at a time by a constant: its mean over the train period of that
checkpoint, over all windows, cells, pseudo-steps and leads, one value per column. A latent channel and the TWSA
history become one value, the lead embedding one vector, so every lead gets the same forecast conditioning. All
change at the LSTM only: the encoder still sees TWSA and FiLM still sees the lead. The question is the importance to
the LSTM, not of the input overall. Nothing in src/ changes and the checkpoints are only read.

Runs: the 8 full seeds, test only. Scores: area-weighted RMSE, as ablation.md, against the same checkpoint with
nothing removed, which first has to reproduce its stored per-lead scores. Channels do not match across seeds, so the
.md table lists every seed on its own and the .tex table only seed 45, chosen because its lead-mean test RMSE is the
closest to the seed mean.

Scores are kept in STUDY_DIR/scores.csv: delete it to score everything again.

Run from the repo root:

    PYTHONPATH=src/ python scripts/run_latent_removal.py --gpu 1
"""

import argparse
import itertools
import json
import glob
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from evaluation_HiGNN import build_model_from_checkpoint, generate_trainer
from results_tables import OUT_DIR
from utils.eval_metrics import _rmse, valid_mask
from utils.utils import read_config

STUDY_DIR = "experiments/HiGNN/V0anom/latent_removal_03"
SEED_STUDY = "experiments/HiGNN/V0anom/seeds_era5anom_nores_02"
SEEDS = [str(s) for s in range(42, 50)]
# mode labels as in latent_removal_02, which plot_latent_removal.py reads
REMOVALS = {"channel_mean": ["dim1", "dim2", "dim3", "dim4"], "lstm_input": ["twsa", "lead"]}
LABELS = {"dim1": "Dim 1", "dim2": "Dim 2", "dim3": "Dim 3", "dim4": "Dim 4", "twsa": "TWSA history",
          "lead": "Lead embedding"}
TABLE_SEED = 45
COLUMNS = ["model", "seed", "dataset", "mode", "removal", "lead", "rmse"]


class RecordMeans:
    """Forward hook that runs once per lead and in lead order: means of its output."""

    def __init__(self, horizon, n_nodes, seq_len, n_channels):
        self.sum = torch.zeros(horizon, n_nodes, seq_len, n_channels, dtype=torch.float64)
        self.samples = torch.zeros(horizon, dtype=torch.float64)
        self.leads = itertools.cycle(range(horizon))

    def __call__(self, module, inputs, output):
        lead = next(self.leads)
        z = output.reshape(-1, *self.sum.shape[1:]).double().cpu()
        self.sum[lead] += z.sum(0)
        self.samples[lead] += len(z)

    def cell_mean(self):
        """[horizon, nodes, seq_len, channels]"""
        return (self.sum / self.samples.reshape(-1, 1, 1, 1)).float()


def channel_mean(cell_mean):
    """One value per channel, over all leads, cells and pseudo-steps, in the shape of `cell_mean`."""
    return cell_mean.mean((0, 1, 2), keepdim=True).expand_as(cell_mean)


class ReplaceChannels:
    """Forward hook: replaces `channels` by `fill` [horizon, nodes, seq_len, channels]."""

    def __init__(self, fill, channels):
        self.fill = fill
        self.replaced = torch.tensor([c in channels for c in range(fill.shape[-1])])
        self.leads = itertools.cycle(range(len(fill)))

    def __call__(self, module, inputs, output):
        fill = self.fill[next(self.leads)].to(output)
        z = output.reshape(-1, *fill.shape)
        # no indexed assignment: index_put_ goes dense under pl's deterministic flag and does not fit on the GPU
        return torch.where(self.replaced.to(output.device), fill, z).reshape(output.shape)


def on_lstm_input(hook):
    """A forward hook as forward pre-hook on the LSTM."""
    def pre_hook(module, args):
        replaced = hook(module, args, args[0])
        return None if replaced is None else (replaced, *args[1:])
    return pre_hook


def lstm_input_columns(model):
    """Columns of the LSTM input, which is cat(lead embedding, twsa, latents)."""
    e = model.timestep_embed_dim
    return {"lead": list(range(e)), "twsa": [e], **{f"dim{c + 1}": [e + 1 + c] for c in range(model.gnn_embedding_dim)}}


def rmse_per_lead(pred, true, lat_weights):
    """`pred`, `true` [windows, cells, leads], scored as in evaluation_HiGNN."""
    return _rmse(pred, true, mask=valid_mask(pred, true), lat_weights=lat_weights).mean(dim=0)


def check_reproduces(rmse, stored):
    assert np.allclose(rmse.numpy(), stored["rmse_latw"].to_numpy(), atol=1e-3), \
        f"no removal gives {rmse} instead of the stored {stored['rmse_latw'].to_numpy()}"


def forecast_rmse(trainer, model, loader, n_nodes, horizon, keep, lat_weights):
    predictions = trainer.predict(model, loader)
    pred = torch.cat([b[0].reshape(-1, n_nodes, horizon) for b in predictions])[:, keep].float()
    true = torch.cat([b[1].reshape(-1, n_nodes, horizon) for b in predictions])[:, keep].float()
    return rmse_per_lead(pred, true, lat_weights)


def load_run(run, gpu):
    """Trainer, datamodule and model of one checkpoint, and the function that scores a loader."""
    config = read_config(os.path.join(run, "config.yaml"))
    config["pl"]["devices"] = [gpu]
    config["experiment"]["save_dir"] = STUDY_DIR
    pl.seed_everything(config["experiment"].get("seed", 42), workers=True)
    trainer = generate_trainer(config)
    dm = TWSA_HeteroGraphDatamodule(**config["datamodule"])
    dm.setup(stage="fit")
    trainer.datamodule = dm
    model = build_model_from_checkpoint(run, config, gpu, dm)
    n_nodes, horizon = dm.data_train.num_land_nodes(), dm.target_seq_len
    keep = ~dm.get_antarctica_landmask()
    lat_weights = torch.cos(torch.deg2rad(torch.Tensor(dm.data_train.land[0][:, 1][keep])))
    return trainer, dm, model, lambda loader: forecast_rmse(trainer, model, loader, n_nodes, horizon, keep, lat_weights)


def stored_scores(run):
    return pd.read_csv(os.path.join(run, "eval_on_test", "per_timestep_scores.csv"))


def score_run(run, seed, gpu):
    """Long-format test rows of COLUMNS of one checkpoint: every removal and the run with nothing removed."""
    trainer, dm, model, score = load_run(run, gpu)
    rnn = model.model.rnn

    record = RecordMeans(dm.target_seq_len, dm.data_train.num_land_nodes(), dm.input_seq_len, rnn.input_size)
    handle = rnn.register_forward_pre_hook(on_lstm_input(record))
    trainer.predict(model, dm.predict_dataloader(period=dm.train_val_test_periods["train"]), return_predictions=False)
    handle.remove()
    fill = channel_mean(record.cell_mean())
    columns = lstm_input_columns(model.model)

    loader = dm.predict_dataloader(period=dm.train_val_test_periods["test"])
    base = score(loader)
    check_reproduces(base, stored_scores(run))
    rows = [("full", seed, "test", "none", "none", lead, float(x)) for lead, x in enumerate(base)]
    for mode, removals in REMOVALS.items():
        for removal in removals:
            handle = rnn.register_forward_pre_hook(on_lstm_input(ReplaceChannels(fill, columns[removal])))
            rmse = score(loader)
            handle.remove()
            rows += [("full", seed, "test", mode, removal, lead, float(x)) for lead, x in enumerate(rmse)]
    return rows


def lead_deltas(scores):
    """RMSE with a removal minus that of the same checkpoint without one, in cm: rows (seed, removal) in the order
    of LABELS, columns leads."""
    rmse = scores.pivot_table(index=["seed", "removal"], columns="lead", values="rmse")
    deltas = rmse.drop(index="none", level="removal").sub(rmse.xs("none", level="removal"), level="seed")
    return deltas.reindex(pd.MultiIndex.from_product([deltas.index.unique("seed"), list(LABELS)]))


def write_tables(scores):
    deltas = lead_deltas(scores)
    base = scores[scores["removal"] == "none"].groupby("seed")["rmse"].mean()
    leads = [f"L{i + 1}" for i in range(deltas.shape[1])]

    md = ["# Input removal (test, area-weighted RMSE per lead and lead mean)\n",
          "ΔRMSE in cm against the same checkpoint with nothing removed (its lead-mean RMSE next to the seed); > 0 "
          "means worse. Mean |Δ| is the lead mean of the absolute ΔRMSE. One LSTM input at a time is replaced by its "
          "training-period mean over all cells, pseudo-steps and leads, one value per column. Latent channels do not "
          "match across seeds. Fixed trained model: not comparable to the retrained ablations.\n",
          "| " + " | ".join(["Seed", "Removed", *leads, "Mean", "Mean \\|Δ\\|"]) + " |", "|---" * (len(leads) + 4) + "|"]
    for seed in deltas.index.unique(0):
        for i, (removal, label) in enumerate(LABELS.items()):
            values = deltas.loc[(seed, removal)].to_numpy()
            name = f"Seed {seed} ({base[seed]:.3f} cm)" if i == 0 else ""
            md.append(f"| {name} | {label} | " + " | ".join(f"{x:+.3f}" for x in [*values, values.mean()])
                      + f" | {np.abs(values).mean():.3f} |")
    with open(os.path.join(OUT_DIR, "latent_removal.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    seed = str(TABLE_SEED)
    header = ["Removed", *leads, "Mean", "Mean $|\\Delta|$"]
    caption = (f"Input removal, test period, seed {seed}: change in area-weighted RMSE per lead and averaged over the "
               "leads, in cm, against the same trained model with nothing removed "
               f"({base[seed]:.3f} cm lead mean); Mean $|\\Delta|$ is the lead mean of its absolute value. "
               "One LSTM input at a time is replaced by its training-period mean "
               "over all cells, pseudo-steps and leads, one value per column, so the lead embedding becomes the same "
               "vector for every lead. Replaced at the LSTM only: the encoder still sees the TWSA history and FiLM "
               "still sees the lead.")
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", "\\label{tab:latent_removal_test}",
             f"\\begin{{tabular}}{{l{'c' * (len(header) - 1)}}}", "\\tophline", " & ".join(header) + " \\\\",
             "\\middlehline"]
    for removal, label in LABELS.items():
        values = deltas.loc[(seed, removal)].to_numpy()
        lines.append(" & ".join([label] + [f"${x:+.3f}$" for x in [*values, values.mean()]]
                                + [f"${np.abs(values).mean():.3f}$"]) + " \\\\")
    lines += ["\\bottomhline", "\\end{tabular}", "\\end{table}"]
    with open(os.path.join(OUT_DIR, "latent_removal_test.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_DIR}/latent_removal.md, {OUT_DIR}/latent_removal_test.tex")


def main():
    parser = argparse.ArgumentParser(prog="run_latent_removal.py", description=__doc__.split("\n")[0])
    parser.add_argument("--gpu", type=int, required=True, help="GPU for the predictions")
    args = parser.parse_args()

    path = os.path.join(STUDY_DIR, "scores.csv")
    if os.path.exists(path):
        scores = pd.read_csv(path, dtype={"seed": str})
    else:
        os.makedirs(STUDY_DIR, exist_ok=True)
        runs = json.load(open(glob.glob(os.path.join(SEED_STUDY, "*_manifest.json"))[0]))
        scores = pd.DataFrame([row for seed in SEEDS for row in score_run(runs[seed], seed, args.gpu)], columns=COLUMNS)
        scores.to_csv(path, index=False)
    write_tables(scores)


if __name__ == "__main__":
    main()

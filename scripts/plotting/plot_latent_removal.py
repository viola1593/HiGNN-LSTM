"""Flat-input figure: test RMSE per lead when one LSTM input is replaced by a constant, against the same checkpoint
with nothing removed, in cm. Reads the scores of run_latent_removal.py, the model is fixed and not retrained.

Thick lines are the table seed. Thin lines are the other seeds. For the latents they are colored by seed (Pastel1),
not by dim, because channels do not match across seeds. The right panel leaves out the TWSA history and zooms in on the rest.

Run from the repo root:

    PYTHONPATH=src/ python scripts/plotting/plot_latent_removal.py
"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/

from run_latent_removal import STUDY_DIR, TABLE_SEED

OUT = "experiments/HiGNN/latent_flat_delta_rmse_test_02.png"
INPUTS = {("lstm_input", "twsa"): "TWSA history", ("lstm_input", "lead"): "Lead embedding",
          ("channel_mean", "dim1"): "Dim 1", ("channel_mean", "dim2"): "Dim 2",
          ("channel_mean", "dim3"): "Dim 3", ("channel_mean", "dim4"): "Dim 4"}
# colorblind-safe categorical palette in fixed order, with a marker as second cue
COLORS = ["#187d30", "#8a8888", "#ac10db", "#ea18e3", "#ed8ded", "#D76679"]
MARKERS = ["o", "o", "s", "s", "s", "s"]


def flat_deltas(scores):
    """Test delta RMSE in cm against the same checkpoint: rows (input, seed), columns leads."""
    full = scores[(scores["model"] == "full") & (scores["dataset"] == "test")]
    rmse = full.pivot_table(index=["mode", "removal", "seed"], columns="lead", values="rmse")
    return pd.concat({label: rmse.loc[key] - rmse.loc[("none", "none")] for key, label in INPUTS.items()},
                     names=["input"])


def plot(deltas, path):
    labels = list(INPUTS.values())
    leads = np.arange(1, deltas.shape[1] + 1)
    table_seed = str(TABLE_SEED)
    other_seeds = deltas.loc[labels[0]].index.drop(table_seed)
    seed_colors = dict(zip(other_seeds, plt.get_cmap("Pastel1").colors))
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))

    for ax, shown, panel in zip(axes, (labels, labels[1:]), ("(a) All inputs", "(b) Without TWSA history, zoomed")):
        ax.axhline(0, color="#52514e", lw=0.8)
        for label, color, marker in zip(labels, COLORS, MARKERS):
            if label not in shown:
                continue
            others = deltas.loc[label].drop(table_seed)
            dim = label.startswith("Dim")
            # pastel colors are light, so seed lines are opaque and a bit thicker
            for seed, row in others.iterrows():
                ax.plot(leads, row.to_numpy(), lw=0.9 if dim else 0.6, zorder=2, alpha=1 if dim else 0.6,
                        color=seed_colors[seed] if dim else color)
            ax.plot(leads, deltas.loc[(label, table_seed)].to_numpy(), lw=1.5, marker=marker, ms=4.5, color=color,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(leads)
        ax.set_xticklabels([f"L{l}" for l in leads])
        ax.set_xlabel("Lead time [months]")
        ax.set_title(panel, fontsize=9, loc="left")
    axes[0].set_ylabel("$\\Delta$RMSE, input held constant [cm]")
    axes[1].legend(handles=[Line2D([], [], lw=0.9, color=c, label=f"Seed {s}") for s, c in seed_colors.items()],
                   frameon=False, fontsize=7, ncol=2, loc="upper right", title="Dims, other seeds", title_fontsize=7)

    handles = [Line2D([], [], lw=1.5, marker=m, ms=4.5, color=c, markeredgecolor="white", markeredgewidth=0.5, label=l)
               for l, c, m in zip(labels, COLORS, MARKERS)]
    handles.append(Line2D([], [], lw=0.9, color=seed_colors[other_seeds[0]], label="Thin: other seeds"))
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"saved {path}")


if __name__ == "__main__":
    scores = pd.read_csv(os.path.join(STUDY_DIR, "scores.csv"), dtype={"seed": str})
    plot(flat_deltas(scores), OUT)

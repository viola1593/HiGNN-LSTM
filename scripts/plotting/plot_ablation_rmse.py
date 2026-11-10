"""Ablation figure: test RMSE per lead of each HiGNN config relative to the V0 seed-mean, in %.

Dots are the individual seeded runs, lines the seed mean with +-1 SD bars. The grey band is the V0 seed +-1 SD.
Run from the repo root: python scripts/plotting/plot_ablation_rmse.py
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = ("HiGNN-LSTM", "experiments/HiGNN/V0anom/seeds_era5anom_nores_02")
CONFIGS = [
    ("No MP", "experiments/HiGNN/V0anom/seeds_era5anom_nores_skipgnn_02"),
    ("No Ocean Nodes", "experiments/HiGNN/V1anom/seeds_nores_era5anom_02"),
    ("No ERA5", "experiments/HiGNN/V0anom/seeds_nores_noera5_02"),
    #  ("Fully connected", "experiments/HiGNN/V0anom/seeds_era5anom_nores_fullyconn_02"),
]
OUT = "experiments/HiGNN/ablation_delta_rmse_test_02.png"
# colorblind-safe categorical palette (blue, orange, aqua), one color per entry in CONFIGS
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def load_rmse(study_dir):
    """Return array (seeds, leads) of area-weighted test RMSE."""
    manifest = json.load(open(glob.glob(os.path.join(study_dir, "*_manifest.json"))[0]))
    return np.stack([
        pd.read_csv(os.path.join(run, "eval_on_test", "per_timestep_scores.csv"))["rmse_latw"].to_numpy()
        for run in manifest.values()
    ])


base = load_rmse(BASE[1])
ref = base.mean(axis=0)
leads = np.arange(1, base.shape[1] + 1)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.axhline(0, color="#52514e", lw=0.8)
base_sd = 100 * base.std(axis=0, ddof=1) / ref
ax.fill_between(leads, -base_sd, base_sd, color="#c3c2b7", alpha=0.45, lw=0, label=f"{BASE[0]} ±1 SD")

offsets = np.linspace(-0.15, 0.15, len(CONFIGS))
for (name, d), off, color in zip(CONFIGS, offsets, COLORS):
    delta = 100 * (load_rmse(d) - ref) / ref
    x = leads + off
    ax.errorbar(x, delta.mean(axis=0), yerr=delta.std(axis=0, ddof=1), fmt="-", lw=1.5,
                elinewidth=1.2, capsize=2, color=color, label=name, zorder=4)
    ax.scatter(np.broadcast_to(x, delta.shape), delta, s=14,
               color=color, edgecolor="white", lw=0.5, alpha=0.3, zorder=3)

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e5e4e0", lw=0.6)
ax.set_axisbelow(True)
ax.set_xticks(leads)
ax.set_xticklabels([f"L{l}" for l in leads])
ax.set_xlabel("Lead time [months]")
ax.set_ylabel(f"$\\Delta$RMSE vs. {BASE[0]} [%]")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight", dpi=300)
print(f"saved {OUT}")

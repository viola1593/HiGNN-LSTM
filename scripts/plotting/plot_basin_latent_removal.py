"""World maps of the per-basin latent-channel removal effect on the test period, one panel per latent dim.

Basins are coloured by the lead-mean ΔRMSE (removal − no removal, cm) of one seed, from
latent_removal_03/basin_scores_seed<seed>.csv (scripts/basin_latent_removal.py), with the colours of
plot_basin_ablation.py: red where the removal is worse, blue where it is better, on one symmetric scale shared by the
four dims (98th percentile of |ΔRMSE|, extended beyond). TWSA history and lead embedding are not drawn and do not set
the scale. No outlines: one fixed checkpoint gives no seed-based p-value. Basins without a score are left unfilled.

Run from the repo root:

    PYTHONPATH=src/ python scripts/plotting/plot_basin_latent_removal.py --seed 45
"""
import argparse
import os
import sys

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # scripts/

import run_latent_removal as rlr
from basin_latent_removal import lead_mean_deltas
from plot_basin_ablation import BASINS, CMAP, INK, MUTED

DIMS = ["dim1", "dim2", "dim3", "dim4"]


def panel(basins, deltas, removal):
    """Polygons of all basins with the lead-mean ΔRMSE of `removal` (NaN where unscored)."""
    out = basins[["HYBAS_ID", "geometry"]].copy()
    out["Mean"] = out["HYBAS_ID"].map(deltas[removal])
    return out


def colour_limit(deltas):
    return np.percentile(np.abs(deltas[DIMS].to_numpy()), 98)


def plot(basins, deltas, path):
    proj = ccrs.Robinson()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.0), subplot_kw=dict(projection=proj))
    vmax = colour_limit(deltas)
    norm = TwoSlopeNorm(0, -vmax, vmax)
    lonlat = ccrs.PlateCarree()  # cartopy splits polygons that cross the dateline
    for ax, dim in zip(axes.flat, DIMS):
        p = panel(basins, deltas, dim)
        scored = p[p["Mean"].notna()]
        ax.add_geometries(p[p["Mean"].isna()].geometry, crs=lonlat, facecolor="none", edgecolor=MUTED,
                          linewidth=0.3)
        for geom, value in zip(scored.geometry, scored["Mean"]):
            ax.add_geometries([geom], crs=lonlat, facecolor=CMAP(norm(value)), edgecolor="white", linewidth=0.2)
        ax.set_title(f"{rlr.LABELS[dim]}: Δ < 0 in {(deltas[dim] < 0).sum()} / {len(deltas)} basins",
                     fontsize=9, color=INK)
        ax.set_global()
        ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.03, extend="both", shrink=0.6)
    cb.set_label("Lead-mean ΔRMSE, removal − no removal [cm]\nred: removal worse, blue: better",
                 color=INK, fontsize=9)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    return fig


def main():
    parser = argparse.ArgumentParser(prog="plot_basin_latent_removal.py", description=__doc__.split("\n")[0])
    parser.add_argument("--seed", default=str(rlr.TABLE_SEED), choices=rlr.SEEDS)
    args = parser.parse_args()
    basins = gpd.read_file(BASINS)
    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326")
    scores = pd.read_csv(os.path.join(rlr.STUDY_DIR, f"basin_scores_seed{args.seed}.csv"), dtype={"seed": str})
    out = f"experiments/HiGNN/basin_latent_removal_map_seed{args.seed}.png"
    plot(basins, lead_mean_deltas(scores), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

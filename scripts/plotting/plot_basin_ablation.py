"""World maps of the per-basin ablation effect on the test period, one panel per reported ablation.

Basins are coloured by the lead-mean ΔRMSE (ablation − full, cm) from tables_02/basin_ablation.csv
(scripts/basin_ablation.py): red where the ablation is worse, blue where it is better, on one symmetric scale
shared by all panels (98th percentile of |ΔRMSE|, extended beyond). Basins with lead-mean Welch p < 0.05 are
outlined. Basins without a score are left unfilled.

Run from the repo root:

    python scripts/plotting/plot_basin_ablation.py
"""
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

TABLE = "experiments/HiGNN/tables_02/basin_ablation.csv"
BASINS = "data/orderedhydrosheds/hydrosheds_basins.gpkg"
OUT = "experiments/HiGNN/basin_ablation_map_test_02.png"
# diverging blue <-> red with a neutral gray midpoint, equal steps per arm
CMAP = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a9a8", "#e34948", "#9e2525"])
INK, MUTED = "#1a1a19", "#c8c7c2"


def panel(basins, table, model):
    """Polygons of all basins with the lead-mean ΔRMSE and p < 0.05 flag of `model` (NaN where unscored)."""
    sub = table[table["model"] == model].set_index("basin")
    out = basins[["HYBAS_ID", "geometry"]].copy()
    out["Mean"] = out["HYBAS_ID"].map(sub["Mean"])
    out["significant"] = out["HYBAS_ID"].isin(sub.index[sub["p"] < 0.05])
    return out


def colour_limit(table):
    return np.percentile(np.abs(table["Mean"]), 98)


def plot(basins, table, path):
    models = list(table["model"].unique())
    ncols = 2
    nrows = int(np.ceil(len(models) / ncols))
    proj = ccrs.Robinson()
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.0 * nrows), subplot_kw=dict(projection=proj),
                             squeeze=False)
    vmax = colour_limit(table)
    norm = TwoSlopeNorm(0, -vmax, vmax)
    lonlat = ccrs.PlateCarree()  # cartopy splits polygons that cross the dateline
    for ax, model in zip(axes.flat, models):
        p = panel(basins, table, model)
        scored = p[p["Mean"].notna()]
        ax.add_geometries(p[p["Mean"].isna()].geometry, crs=lonlat, facecolor="none", edgecolor=MUTED,
                          linewidth=0.3)
        for geom, value in zip(scored.geometry, scored["Mean"]):
            ax.add_geometries([geom], crs=lonlat, facecolor=CMAP(norm(value)), edgecolor="white", linewidth=0.2)
        sig = scored[scored["significant"]]
        ax.add_geometries(sig.geometry, crs=lonlat, facecolor="none", edgecolor=INK, linewidth=0.7)
        ax.set_title(f"{model}: {len(sig)} basins p < 0.05 "
                     f"({(sig['Mean'] > 0).sum()} worse, {(sig['Mean'] < 0).sum()} better)", fontsize=9, color=INK)
        ax.set_global()
        ax.axis("off")
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    if len(models) < axes.size:  # colour bar in the first empty panel, else below all
        free = axes.flat[len(models)]
        cax = free.inset_axes([0.1, 0.45, 0.8, 0.08], transform=free.transAxes)
        cax.set_visible(True)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="both")
        free.set_visible(True)
        free.axis("off")
    else:
        cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.03, extend="both",
                          shrink=0.6)
    cb.set_label("Lead-mean ΔRMSE, ablation − full [cm]\nred: ablation worse, blue: better; "
                 "outline: p < 0.05", color=INK, fontsize=9)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    return fig


def main():
    basins = gpd.read_file(BASINS)
    if basins.crs is None:
        basins = basins.set_crs("EPSG:4326")
    plot(basins, pd.read_csv(TABLE), OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

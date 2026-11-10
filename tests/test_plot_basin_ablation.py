"""plot_basin_ablation.py: world maps of the per-basin lead-mean ΔRMSE of each ablation."""
import pathlib
import sys

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

matplotlib.use("Agg")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts" / "plotting"))

import plot_basin_ablation as pba


def _table():
    return pd.DataFrame(dict(model=["No X", "No X", "No Y", "No Y"], basin=[1, 2, 1, 2],
                             Mean=[0.2, -0.5, 0.0, 0.1], p=[0.01, 0.3, 0.9, 0.04]))


def _basins():
    return gpd.GeoDataFrame(dict(HYBAS_ID=[1, 2, 3]),
                            geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)], crs="EPSG:4326")


def test_panel_joins_one_model_onto_the_polygons_and_flags_p_below_005():
    panel = pba.panel(_basins(), _table(), "No X")
    assert list(panel["HYBAS_ID"]) == [1, 2, 3]
    np.testing.assert_allclose(panel["Mean"].to_numpy()[:2], [0.2, -0.5])
    assert np.isnan(panel["Mean"].iloc[2])  # unscored basin stays, without a value
    assert list(panel["significant"]) == [True, False, False]


def test_colour_limit_is_symmetric_and_shared_over_models():
    assert pba.colour_limit(_table()) == pytest.approx(np.percentile([0.2, 0.5, 0.0, 0.1], 98))


def test_zero_change_is_the_gray_midpoint():
    np.testing.assert_allclose(pba.CMAP(0.5)[:3], matplotlib.colors.to_rgb("#f0efec"), atol=0.01)


def test_plot_draws_one_map_per_model(tmp_path):
    out = tmp_path / "map.png"
    fig = pba.plot(_basins(), _table(), str(out))
    assert out.exists()
    assert len([ax for ax in fig.axes if hasattr(ax, "projection")]) == 2

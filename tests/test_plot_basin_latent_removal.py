"""plot_basin_latent_removal.py: world maps of the per-basin lead-mean ΔRMSE of each latent-dim removal."""
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

import plot_basin_latent_removal as pblr

DIMS = ["dim1", "dim2", "dim3", "dim4"]


def _deltas():
    """lead_mean_deltas layout: rows basins 1 and 2, columns removals; TWSA history much larger than the dims."""
    return pd.DataFrame(dict(dim1=[0.2, -0.5], dim2=[0.0, 0.1], dim3=[0.3, 0.05], dim4=[-0.1, 0.4],
                             twsa=[5.0, 8.0], lead=[1.0, 2.0]), index=pd.Index([1, 2], name="basin"))


def _basins():
    return gpd.GeoDataFrame(dict(HYBAS_ID=[1, 2, 3]),
                            geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)], crs="EPSG:4326")


def test_panel_joins_one_removal_onto_the_polygons():
    panel = pblr.panel(_basins(), _deltas(), "dim1")
    assert list(panel["HYBAS_ID"]) == [1, 2, 3]
    np.testing.assert_allclose(panel["Mean"].to_numpy()[:2], [0.2, -0.5])
    assert np.isnan(panel["Mean"].iloc[2])  # unscored basin stays, without a value


def test_colour_limit_is_shared_over_the_four_dims_only():
    dims = _deltas()[DIMS].to_numpy().ravel()
    assert pblr.colour_limit(_deltas()) == pytest.approx(np.percentile(np.abs(dims), 98))


def test_plot_draws_one_map_per_dim_and_none_for_the_references(tmp_path):
    out = tmp_path / "map.png"
    fig = pblr.plot(_basins(), _deltas(), str(out))
    assert out.exists()
    maps = [ax for ax in fig.axes if hasattr(ax, "projection")]
    assert [ax.get_title().split(":")[0] for ax in maps] == ["Dim 1", "Dim 2", "Dim 3", "Dim 4"]


def test_title_counts_basins_that_get_better(tmp_path):
    fig = pblr.plot(_basins(), _deltas(), str(tmp_path / "map.png"))
    titles = [ax.get_title() for ax in fig.axes if hasattr(ax, "projection")]
    assert titles[0] == "Dim 1: Δ < 0 in 1 / 2 basins"
    assert titles[1] == "Dim 2: Δ < 0 in 0 / 2 basins"

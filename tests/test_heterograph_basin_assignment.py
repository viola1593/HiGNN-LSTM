"""`assign_hydrobasins` maps every land point to a hydrobasin, and the points that no
basin covers (Antarctica, Hawaii) fall back to the artificial basin 1030099999.

The fallback used to be accompanied by an `na_gdf` frame that geopandas >= 1.0 refuses
to construct, which made the `landpoints_to_basin.csv` rebuild path raise on every call.
These tests pin down that the join runs, that it returns coords -> basin only, and that
the fallback still catches the uncoverable points.

The synthetic tests use a bare instance so they do not pull in the real shapefile. The
equality test against the cached CSV needs the real data and is skipped without it.
"""
import os
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from datamodules.TWSA_heterograph_dataset import TWSAGraphHeteroData

NA_BASIN_ID = 1030099999

REPO = pathlib.Path(__file__).resolve().parents[1]
SHAPEFILE = REPO / 'data' / 'orderedhydrosheds' / 'hydrosheds_basins.shp'
CACHED_CSV = REPO / 'graphs' / 'full_unitsphere_v2' / 'landpoints_to_basin.csv'


def bare():
    return TWSAGraphHeteroData.__new__(TWSAGraphHeteroData)


def make_basins():
    """Two unit-square basins side by side, plus the degenerate Antarctica basin that
    the real shapefile also carries at (90, -80)."""
    return gpd.GeoDataFrame({
        'HYBAS_ID': [1000000001, 1000000002, NA_BASIN_ID],
        'geometry': [box(0, 0, 1, 1), box(2, 0, 3, 1), Point(90, -80.0)],
    }, crs='EPSG:4326')


# one point well inside each square, and one far from any basin
COORDS = np.array([[0.5, 0.5], [2.5, 0.5], [50.0, -70.0]])
EXPECTED = np.array([1000000001, 1000000002, NA_BASIN_ID])


def test_assign_hydrobasins_runs_and_returns_ids_and_join():
    basin_ids, joined = bare().assign_hydrobasins(COORDS, make_basins())

    assert len(basin_ids) == len(COORDS)
    assert len(joined) == len(COORDS)


def test_covered_points_get_their_basin_and_the_rest_fall_back():
    basin_ids, _ = bare().assign_hydrobasins(COORDS, make_basins())

    np.testing.assert_array_equal(basin_ids.values, EXPECTED)


def test_basin_ids_are_integers():
    basin_ids, _ = bare().assign_hydrobasins(COORDS, make_basins())

    assert basin_ids.dtype == np.int64


def test_the_join_keeps_one_row_per_point_when_a_point_is_equidistant():
    """A point on the shared edge of two touching basins matches both. Only one row
    per input point may survive, or the caller's np.vstack against the point index
    would fail and the written CSV would be longer than the coords it describes."""
    basins = gpd.GeoDataFrame({
        'HYBAS_ID': [1000000001, 1000000002],
        'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1)],
    }, crs='EPSG:4326')
    coords = np.array([[1.0, 0.5]])  # exactly on the shared edge

    basin_ids, joined = bare().assign_hydrobasins(coords, basins)

    assert len(basin_ids) == 1
    assert len(joined) == 1


def test_the_join_geometry_survives_for_the_caller():
    """get_land_to_basin_ids reads joined['geometry'] back out to write lon/lat."""
    _, joined = bare().assign_hydrobasins(COORDS, make_basins())
    written = np.array([[g.x, g.y] for g in joined['geometry'].values])

    np.testing.assert_allclose(written, COORDS)


@pytest.mark.skipif(not (SHAPEFILE.exists() and CACHED_CSV.exists()),
                    reason='needs the real hydrobasins shapefile and the cached CSV')
def test_rebuild_reproduces_the_cached_csv():
    """The acceptance test for the fix: the rebuild path it unblocks must reproduce
    landpoints_to_basin.csv, not replace it. Slow (~45 s) - the sjoin dominates."""
    os.chdir(REPO)
    cache = pd.read_csv(CACHED_CSV)
    coords = cache[['lon', 'lat']].values
    gdf = gpd.read_file(SHAPEFILE)

    basin_ids, joined = bare().assign_hydrobasins(coords, gdf.copy())

    np.testing.assert_array_equal(basin_ids.values, cache['HYBAS_ID'].astype('int64').values)
    written = np.array([[g.x, g.y] for g in joined['geometry'].values])
    np.testing.assert_allclose(written, coords, atol=1e-5)

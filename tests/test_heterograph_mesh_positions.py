"""The mesh nodes are placed at the spherical mean of their members: the mean of the
members' unit-sphere xyz, rescaled onto the sphere. A planar lon/lat mean puts a node that
straddles the antimeridian or surrounds a pole on the wrong side of the globe, and the mesh
positions feed every nearest-neighbour edge set.

Members are the ocean pixels of a cluster (see test_heterograph_clustering.py) and the land
grid cells assigned to a basin (here). Bare instances keep the real data out of the tests.
"""
import numpy as np
import pandas as pd

from datamodules.TWSA_heterograph_dataset import TWSAGraphDataset, TWSAGraphHeteroData


def normalized_mean_xyz(obj, coords):
    mean_xyz = obj.unit_sphere_embedding(coords).mean(axis=0)
    return mean_xyz / np.linalg.norm(mean_xyz)


# --- spherical_mean -------------------------------------------------------------------

def test_spherical_mean_is_the_rescaled_mean_of_the_embedding():
    obj = TWSAGraphDataset.__new__(TWSAGraphDataset)
    coords = np.array([[10.0, 20.0], [30.0, -5.0], [-40.0, 60.0]])

    mean = obj.spherical_mean(coords)

    assert mean.shape == (2,)
    np.testing.assert_allclose(obj.unit_sphere_embedding(mean[None])[0],
                               normalized_mean_xyz(obj, coords), atol=1e-12)


def test_spherical_mean_across_the_antimeridian():
    obj = TWSAGraphDataset.__new__(TWSAGraphDataset)
    coords = np.array([[179.0, -1.0], [179.0, 1.0], [-179.0, -1.0], [-179.0, 1.0]])

    lon, lat = obj.spherical_mean(coords)

    assert abs(abs(lon) - 180.0) < 1e-9
    assert abs(lat) < 1e-9


def test_spherical_mean_of_a_ring_around_the_pole_is_the_pole():
    obj = TWSAGraphDataset.__new__(TWSAGraphDataset)
    lons = np.arange(-180.0, 180.0, 30.0)
    coords = np.stack([lons, np.full_like(lons, 80.0)], axis=1)

    _, lat = obj.spherical_mean(coords)

    assert abs(lat - 90.0) < 1e-9


# --- basin positions ------------------------------------------------------------------

# basin 1 straddles the antimeridian, basin 3 is a ring around the south pole
LAND_COORDS = np.array([[179.5, 10.0], [-179.5, 10.0],
                        [20.0, 0.0], [22.0, 2.0],
                        [0.0, -80.0], [120.0, -80.0], [-120.0, -80.0]])
LAND_TO_BASIN_IDS = pd.Series([1, 1, 2, 2, 3, 3, 3], dtype='int64')
# the basin order of hydro_basins, deliberately not the order of first appearance
BASIN_IDS = pd.Series([2, 3, 1], dtype='int64')


def test_each_basin_is_placed_at_the_spherical_mean_of_its_assigned_cells():
    obj = TWSAGraphHeteroData.__new__(TWSAGraphHeteroData)

    centroids = obj.basin_centroids(LAND_COORDS, LAND_TO_BASIN_IDS, BASIN_IDS)

    assert len(centroids) == len(BASIN_IDS)
    got = np.array([[p.x, p.y] for p in centroids])
    for i, bid in enumerate(BASIN_IDS):
        members = LAND_COORDS[LAND_TO_BASIN_IDS.values == bid]
        np.testing.assert_allclose(obj.unit_sphere_embedding(got[i][None])[0],
                                   normalized_mean_xyz(obj, members), atol=1e-12)


def test_basin_positions_do_not_break_at_the_antimeridian_or_the_pole():
    obj = TWSAGraphHeteroData.__new__(TWSAGraphHeteroData)

    centroids = obj.basin_centroids(LAND_COORDS, LAND_TO_BASIN_IDS, BASIN_IDS)

    across_the_seam, around_the_pole = centroids[2], centroids[1]
    assert abs(abs(across_the_seam.x) - 180.0) < 1e-9
    assert abs(around_the_pole.y + 90.0) < 1e-9

"""A5 - scripts/preprocess_data/get_watersheds.py

Offline checks for the neighbour-list join in main().  getInfo() needs Earth
Engine credentials, so the dataset is replaced by a synthetic stub whose
getInfo() order deliberately differs from its HYBAS_ID order - which is exactly
the condition under which the positional assignment
``gdf['Neigh_HYBAS_ID'] = neighbors_list`` lands neighbour lists on the wrong
rows.
"""
import importlib.util
import pathlib

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box, mapping

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts' / 'preprocess_data' / '03_get_watersheds.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('get_watersheds_a5', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- synthetic Earth Engine stub -------------------------------------------
# A chain of N unit squares along x, each touching its left and right neighbour:
#   position p -> x in [p, p+1].  IDs run backwards along the chain, so the
#   order the collection hands out (position order) is the reverse of HYBAS_ID
#   order.  main() indexes idx_to_basin[100], so the chain needs > 101 basins.
N_BASINS = 105
NATURAL_ORDER = [1000 + (N_BASINS - 1 - p) * 10 for p in range(N_BASINS)]
GEOMS = {
    basin_id: box(p, 0, p + 1, 1) for p, basin_id in enumerate(NATURAL_ORDER)
}
# Basins each basin touches, itself included (find_neighboring_basins does not
# exclude the target - the neq filter is commented out in the script).
EXPECTED_NEIGHBORS = {
    NATURAL_ORDER[p]: sorted(
        NATURAL_ORDER[q] for q in (p - 1, p, p + 1) if 0 <= q < N_BASINS
    )
    for p in range(N_BASINS)
}


class _Feature:
    def __init__(self, basin_id):
        self.properties = {'HYBAS_ID': basin_id, 'SUB_AREA': float(basin_id)}
        self.geom = GEOMS[basin_id]

    def geometry(self):
        return self.geom

    def get(self, key):
        return self.properties[key]


class _Array:
    def __init__(self, values):
        self.values = list(values)

    def getInfo(self):
        return list(self.values)


class _Collection:
    def __init__(self, features):
        self.features = list(features)

    def sort(self, prop):
        return _Collection(sorted(self.features, key=lambda f: f.properties[prop]))

    def aggregate_array(self, prop):
        return _Array([f.properties[prop] for f in self.features])

    def filter(self, predicate):
        return _Collection([f for f in self.features if predicate(f)])

    def first(self):
        return self.features[0]

    def merge(self, other):
        return _Collection(self.features + other.features)

    def getInfo(self):
        return {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'properties': dict(f.properties),
                    'geometry': mapping(f.geom),
                }
                for f in self.features
            ],
        }


class _Filter:
    @staticmethod
    def eq(prop, value):
        return lambda f: f.properties[prop] == value

    @staticmethod
    def inList(prop, values):
        return lambda f: f.properties[prop] in values

    @staticmethod
    def intersects(_field, geometry):
        return lambda f: f.geom.intersects(geometry)

    @staticmethod
    def And(*predicates):
        return lambda f: all(p(f) for p in predicates)


class _EE:
    Filter = _Filter

    @staticmethod
    def Authenticate():
        pass

    @staticmethod
    def Initialize():
        pass

    @staticmethod
    def FeatureCollection(_name):
        return _Collection([_Feature(bid) for bid in NATURAL_ORDER])


@pytest.fixture
def gw(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, 'ee', _EE)
    return module


@pytest.fixture
def dataset():
    return _EE.FeatureCollection('stub')


def test_stub_order_differs_from_hybas_order(dataset):
    """The stub only tests anything if getInfo() order != sorted HYBAS_ID order."""
    natural = [f['properties']['HYBAS_ID'] for f in dataset.getInfo()['features']]
    assert natural != sorted(natural)


def test_neighbors_list_follows_sorted_basin_order(gw, dataset):
    """create_adjacency_matrix returns neighbours indexed by sorted HYBAS_ID."""
    _, _, basin_to_idx, neighbors_list = gw.create_adjacency_matrix(dataset)

    for basin_id, idx in basin_to_idx.items():
        assert sorted(neighbors_list[idx]) == EXPECTED_NEIGHBORS[basin_id]


def test_adjacency_matrix_is_pure_intersection(gw, dataset):
    """Guard for the removed distance loop: edges are 0/1 shared-border edges only."""
    adj, idx_to_basin, basin_to_idx, _ = gw.create_adjacency_matrix(dataset)

    expected = np.zeros((N_BASINS, N_BASINS))
    for basin_id, neighbors in EXPECTED_NEIGHBORS.items():
        for neighbor_id in neighbors:
            expected[basin_to_idx[basin_id], basin_to_idx[neighbor_id]] = 1

    assert np.array_equal(adj, expected)
    assert [idx_to_basin[i] for i in range(N_BASINS)] == sorted(GEOMS)


def test_shapefile_neighbours_match_their_own_row(gw, monkeypatch):
    """Neigh_HYBAS_ID must describe the basin on its own row, not another one."""
    written = []
    monkeypatch.setattr(gw, 'plot_basin_and_neighbors', lambda *a, **k: None)
    monkeypatch.setattr(gw.np, 'save', lambda *a, **k: None)
    monkeypatch.setattr(
        gpd.GeoDataFrame, 'to_file', lambda self, *a, **k: written.append(self.copy())
    )

    gw.main()

    gdf = written[0]
    assert list(gdf['HYBAS_ID']) == sorted(GEOMS)
    for _, row in gdf.iterrows():
        assert sorted(row['Neigh_HYBAS_ID']) == EXPECTED_NEIGHBORS[row['HYBAS_ID']], (
            f"basin {row['HYBAS_ID']} carries the wrong neighbour list"
        )

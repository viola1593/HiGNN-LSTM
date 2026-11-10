"""The heterograph normalization must apply the same std floor as the 2D dataset,
and the two must agree cell for cell on the shared variables.

`normalize_ds_timeslice` is exercised on a bare instance so the test does not pull
in graph construction, clustering or the hydrobasin cache.
"""
import numpy as np
import pytest

from synthetic_grid import (LAND, LOW_VARIANCE_CELL, METEO_VARS, NEGATIVE_RUNOFF_CELL,
                            ORDINARY_CELL, expected_runoff_floor, make_ds)
from test_dataset2D_masks import build as build_2d, normalized as normalized_2d

from datamodules.TWSA_heterograph_dataset import TWSAGraphDataset

LAND_THRESHOLD = 0.8


def build_hetero(scaling_mean_std=None):
    ds = make_ds()
    obj = TWSAGraphDataset.__new__(TWSAGraphDataset)
    obj.ds_timeslice = ds.astype(np.float32)
    obj.land_ocean_mask = ds['lsm'] > LAND_THRESHOLD
    obj.input_vars = [v for v in ds.data_vars if v != 'lsm']  # lsm is not an input var here
    obj.scaling_params = scaling_mean_std
    obj.features_nanfill = 0.0
    obj.only_land_variables = ['ro', 'pev', 'swvl1', 'swvl2', 'swvl3', 'swvl4', 'lai_hv', 'lai_lv']
    obj.only_ocean_variables = ['sst']
    obj.normalize_ds_timeslice()
    return obj


@pytest.fixture
def hetero():
    return build_hetero()


def normalized_hetero(obj, name):
    return obj.ds_timeslice[name].values


def test_negative_runoff_is_not_clipped_before_statistics(hetero):
    raw = make_ds()['ro'].values[:, NEGATIVE_RUNOFF_CELL[0], NEGATIVE_RUNOFF_CELL[1]]
    assert raw.min() < 0, "fixture no longer exercises negative runoff"
    mean, std = hetero.scaling_params[hetero.input_vars.index('ro')]

    assert mean.values[NEGATIVE_RUNOFF_CELL] == pytest.approx(raw.mean(), abs=1e-4)
    assert std.values[NEGATIVE_RUNOFF_CELL] == pytest.approx(raw.std(), rel=1e-4)


def test_small_std_is_clamped_to_the_floor_not_replaced(hetero):
    _, std = hetero.scaling_params[hetero.input_vars.index('ro')]
    floor = expected_runoff_floor()
    raw_std = make_ds()['ro'].values.std(axis=0)

    assert std.values[LOW_VARIANCE_CELL] == pytest.approx(floor, rel=1e-3)
    assert std.values[ORDINARY_CELL] == pytest.approx(raw_std[ORDINARY_CELL], rel=1e-3)


def test_floor_never_amplifies(hetero):
    for var in METEO_VARS:
        per_cell_std = normalized_hetero(hetero, var).std(axis=0)
        assert per_cell_std.max() <= 1.0 + 1e-4, f"{var} amplified beyond unit variance"


def test_matches_the_2d_dataset_cell_for_cell():
    hetero = build_hetero()
    two_d = build_2d()

    for var in METEO_VARS:
        np.testing.assert_allclose(normalized_hetero(hetero, var), normalized_2d(two_d, var),
                                   rtol=1e-4, atol=1e-5,
                                   err_msg=f"{var} differs between the 2D and heterograph datasets")


def test_val_split_reuses_training_statistics(hetero):
    val = build_hetero(scaling_mean_std=hetero.scaling_params)

    for var in METEO_VARS:
        np.testing.assert_allclose(normalized_hetero(val, var), normalized_hetero(hetero, var),
                                   rtol=1e-4, atol=1e-5)

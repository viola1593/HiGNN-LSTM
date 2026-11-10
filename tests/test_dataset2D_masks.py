"""Checks that TWSA2DDataset normalization matches the heterograph dataset.

Covers:
  - land/ocean masks partition the grid (review issue #16)
  - runoff is not clipped at 0: the ERA5 inputs are detrended, so negative values are valid
  - the std floor clamps the divisor instead of replacing it, so a low-variance
    pixel is attenuated rather than amplified or muted
"""
import numpy as np
import pytest

from synthetic_grid import (COASTAL, LAND, LOW_VARIANCE_CELL, LSM, METEO_VARS,
                            NEGATIVE_RUNOFF_CELL, OCEAN, ORDINARY_CELL,
                            expected_runoff_floor, make_ds)

from datamodules.TWSA_dataset2D import TWSA2DDataset


def build(scaling_mean_std=None):
    ds = make_ds()
    return TWSA2DDataset(
        ds,
        time_period=(ds.time.values[0], ds.time.values[-1]),
        target='twsa',
        delay=0,
        input_seq_len=3,
        output_seq_len=1,
        input_vars=list(ds.data_vars.keys()),
        scaling_mean_std=scaling_mean_std,
    )


@pytest.fixture
def dataset():
    return build()


def normalized(dataset, name):
    """Normalized values of one variable, shape (time, lat, lon)."""
    return dataset.features[:, dataset.input_vars.index(name)].numpy()


def test_ocean_mask_is_the_complement_of_the_land_mask(dataset):
    """sst must survive on every non-land cell, coastal cells included."""
    sst = normalized(dataset, 'sst')

    for i, j in np.argwhere(~LAND):
        assert sst[:, i, j].std() > 0, f"sst zeroed at non-land cell ({i}, {j}), lsm={LSM[i, j]}"
    for i, j in np.argwhere(LAND):
        assert np.all(sst[:, i, j] == 0), f"sst kept at land cell ({i}, {j}), lsm={LSM[i, j]}"


def test_land_variables_stay_off_non_land_cells(dataset):
    ro = normalized(dataset, 'ro')

    for i, j in np.argwhere(COASTAL | OCEAN):
        assert np.all(ro[:, i, j] == 0), f"ro kept at non-land cell ({i}, {j})"


def test_negative_runoff_is_not_clipped_before_statistics(dataset):
    raw = make_ds()['ro'].values[:, NEGATIVE_RUNOFF_CELL[0], NEGATIVE_RUNOFF_CELL[1]]
    assert raw.min() < 0, "fixture no longer exercises negative runoff"

    mean, std = dataset.scaling_params[dataset.input_vars.index('ro')]

    assert mean.values[NEGATIVE_RUNOFF_CELL] == pytest.approx(raw.mean(), abs=1e-4)
    assert std.values[NEGATIVE_RUNOFF_CELL] == pytest.approx(raw.std(), rel=1e-4)


def test_small_std_is_clamped_to_the_floor_not_replaced(dataset):
    _, std = dataset.scaling_params[dataset.input_vars.index('ro')]
    floor = expected_runoff_floor()

    raw_std = make_ds()['ro'].values.std(axis=0)
    assert raw_std[LOW_VARIANCE_CELL] < floor, "fixture no longer exercises the floor"

    # clamped to the floor: not left alone (would amplify) and not set to 1 (would mute)
    assert std.values[LOW_VARIANCE_CELL] == pytest.approx(floor, rel=1e-3)
    assert std.values[ORDINARY_CELL] == pytest.approx(raw_std[ORDINARY_CELL], rel=1e-3)


def test_floor_attenuates_and_never_amplifies(dataset):
    """A clamped divisor can only push a pixel's normalized variance below 1."""
    ro = normalized(dataset, 'ro')
    low = ro[:, LOW_VARIANCE_CELL[0], LOW_VARIANCE_CELL[1]].std()

    assert 0 < low < 1, f"low-variance cell should be attenuated, got std {low}"
    assert ro[:, ORDINARY_CELL[0], ORDINARY_CELL[1]].std() == pytest.approx(1.0, rel=1e-3)

    for var in METEO_VARS:
        per_cell_std = normalized(dataset, var).std(axis=0)
        assert per_cell_std.max() <= 1.0 + 1e-4, f"{var} amplified beyond unit variance"


def test_val_split_reuses_training_statistics(dataset):
    """The scaling_params branch must apply the same masking."""
    val = build(scaling_mean_std=dataset.scaling_params)

    for var in METEO_VARS:
        np.testing.assert_allclose(normalized(val, var), normalized(dataset, var),
                                   rtol=1e-4, atol=1e-5)

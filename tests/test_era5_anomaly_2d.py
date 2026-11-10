"""Deseasonalized ERA5 inputs for the ConvLSTM (TWSA_Datamodule2D flag `era5_anomaly`).

Mirrors tests/test_era5_anomaly.py for TWSA2DDataset: with a training climatology C
(xr.Dataset, one [month, lat, lon] variable per ERA5 input) the dataset normalizes
var − C[var][month] instead of var for every variable in C. twsa keeps its own flag.

Unlike the heterograph, lsm is a 2D input (the dataset reads its land mask from it). lsm is
static, so its climatology is lsm itself: the datamodule leaves it out of C, otherwise the
lsm feature would become 0 everywhere.
"""
import numpy as np
import pytest
import torch
import xarray as xr

import datamodules.TWSA_datamodule2D as dm2d_module
from datamodules.TWSA_datamodule2D import TWSA_Datamodule2D
from datamodules.TWSA_dataset2D import TWSA2DDataset
from synthetic_grid import LAND, LSM
from test_era5_anomaly import (ERA5_VARS, H, MASKS, SEQ, TRAIN_STEPS, era5_climatology_24m,
                               expected_z, make_era5_anomaly_ds, twsa_climatology_24m)


def build_2d(**kw):
    ds = make_era5_anomaly_ds()
    return TWSA2DDataset(ds, time_period=(ds.time.values[0], ds.time.values[-1]), delay=0,
                         input_seq_len=SEQ, output_seq_len=H, input_vars=list(ds.data_vars), **kw)


def normalized(obj, var):
    """normalized var as the model sees it, [time, lat, lon]"""
    return obj.features[:, obj.input_vars.index(var)].numpy()


# ---------------------------------------------------------------- E1: ERA5 anomaly normalization


@pytest.mark.parametrize('anomaly', [False, True])
@pytest.mark.parametrize('var', ERA5_VARS)
def test_era5_input_is_normalized_as_an_anomaly(var, anomaly):
    """Flag off builds without the new kwarg, so that case checks the oracle against today's code."""
    C = era5_climatology_24m() if anomaly else None
    obj = build_2d(era5_climatology=C) if anomaly else build_2d()
    np.testing.assert_allclose(normalized(obj, var), expected_z(var, None if C is None else C[var]),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('var', ERA5_VARS)
def test_val_branch_subtracts_the_given_era5_climatology(var):
    """With the training scaling params a +5 shift of C moves the normalized input by −5/std."""
    C = era5_climatology_24m()
    train = build_2d(era5_climatology=C)
    val = build_2d(era5_climatology=C + 5.0, scaling_mean_std=train.scaling_params)
    mask = MASKS[var]
    std = train.scaling_params[train.input_vars.index(var)][1].values[mask]
    np.testing.assert_allclose(normalized(val, var)[:, mask], normalized(train, var)[:, mask] - 5.0 / std,
                               rtol=1e-4, atol=1e-4)
    assert (normalized(val, var)[:, ~mask] == 0).all()


# ---------------------------------------------------------------- E2: twsa and lsm keep their values


@pytest.mark.parametrize('twsa_anomaly', [False, True])
def test_twsa_is_independent_of_era5_anomaly(twsa_anomaly):
    C_twsa = twsa_climatology_24m() if twsa_anomaly else None
    on = build_2d(era5_climatology=era5_climatology_24m(), twsa_climatology=C_twsa)
    if twsa_anomaly:
        np.testing.assert_allclose(normalized(on, 'twsa'), expected_z('twsa', C_twsa), rtol=1e-4, atol=1e-5)
    else:
        off = build_2d()
        np.testing.assert_array_equal(normalized(on, 'twsa'), normalized(off, 'twsa'))
        assert on.twsa_offset is None


def test_lsm_feature_and_land_mask_are_untouched():
    on, off = build_2d(era5_climatology=era5_climatology_24m()), build_2d()
    np.testing.assert_array_equal(normalized(on, 'lsm'), normalized(off, 'lsm'))
    np.testing.assert_array_equal(normalized(on, 'lsm'), np.broadcast_to(LSM, normalized(on, 'lsm').shape))
    np.testing.assert_array_equal(on.land_ocean_mask[0], LAND)


# ---------------------------------------------------------------- E3: flag off changes nothing


def test_flag_off_is_bit_identical():
    before, off = build_2d(), build_2d(era5_climatology=None)
    assert torch.equal(before.features, off.features)
    xr.testing.assert_identical(before.ds_timeslice, off.ds_timeslice)


# ---------------------------------------------------------------- E4: the datamodule owns C

INPUT_VARS = ['twsa'] + ERA5_VARS + ['lsm']
DM_KW = dict(train_val_test_split=(0.5, 0.25, 0.25), input_seq_len=SEQ, output_seq_len=H, delay=0,
             batch_size=2, num_workers=0, input_vars=INPUT_VARS)


@pytest.fixture
def nc_path(tmp_path, monkeypatch):
    """Synthetic grid on disk as float64 (like 02_RecTWSA_era5.nc), the ERA5 inputs +100 after
    the training period, sst NaN in one land cell, and a GRACE file for GRACE_dataloader."""
    ds = make_era5_anomaly_ds()
    for var in ERA5_VARS:
        ds[var] = ds[var].astype(np.float64)
        ds[var].values[TRAIN_STEPS:] += 100.0
    ds['sst'].values[:, 0, 0] = np.nan  # sst is NaN on land
    ds.to_netcdf(tmp_path / 'syn.nc')
    make_era5_anomaly_ds().to_netcdf(tmp_path / 'grace.nc')
    monkeypatch.setattr(dm2d_module, 'GRACE_PATH', str(tmp_path / 'grace.nc'))
    return str(tmp_path / 'syn.nc')


def test_datamodule_owns_the_era5_training_climatology_as_float32(nc_path):
    dm = TWSA_Datamodule2D(ds_path=nc_path, era5_anomaly=True, **DM_KW)
    train = xr.open_dataset(nc_path).sel(time=slice(*dm.train_val_test_periods['train']))
    assert train.time.size == TRAIN_STEPS  # the +100 never enters the training mean
    C = dm.era5_climatology
    assert set(C.data_vars) == set(dm.input_vars) - {'twsa', 'lsm'} == set(ERA5_VARS)
    for var in ERA5_VARS:
        assert C[var].dims == ('month', 'lat', 'lon')
        assert C[var].dtype == np.float32, var  # float64 C would promote the inputs in ds_timeslice
        np.testing.assert_allclose(C[var].values, train[var].groupby('time.month').mean('time').fillna(0).values,
                                   rtol=1e-6, atol=1e-6, err_msg=var)
    assert (C['sst'].values[:, 0, 0] == 0).all()  # NaN filled with 0


def test_twsa_and_lsm_only_inputs_build_without_era5_climatology(nc_path):
    """input_vars=['twsa', 'lsm'] leaves no ERA5 variable to average, so with the flag off the
    datamodule must not try to build the ERA5 climatology."""
    dm = TWSA_Datamodule2D(ds_path=nc_path, era5_anomaly=False, **{**DM_KW, 'input_vars': ['twsa', 'lsm']})
    assert dm.input_vars == ['twsa', 'lsm']
    assert dm.era5_climatology is None


NOT_PASSED = 'not passed'


@pytest.mark.parametrize('flag', [False, True])
def test_every_dataset_construction_site_gets_the_era5_climatology_only_when_on(nc_path, monkeypatch, flag):
    """fit train, fit val, validate, test, predict and GRACE must all pass era5_climatology
    explicitly (None when off)."""
    seen = []

    class Recorder:
        scaling_params = None

        def __init__(self, ds, *args, **kw):
            seen.append(kw.get('era5_climatology', NOT_PASSED))

    monkeypatch.setattr(dm2d_module, 'TWSA2DDataset', Recorder)
    dm = TWSA_Datamodule2D(ds_path=nc_path, era5_anomaly=flag, **DM_KW)
    dm.setup('fit')  # train + val
    dm.data_val = None
    dm.setup('validate')
    dm.setup('test')
    dm.predict_dataloader(period=dm.train_val_test_periods['val'])
    dm.GRACE_dataloader()
    assert len(seen) == 6
    assert not any(c is NOT_PASSED for c in seen), 'a construction site omits era5_climatology'
    expected = dm.era5_climatology if flag else None
    assert all(c is expected for c in seen)

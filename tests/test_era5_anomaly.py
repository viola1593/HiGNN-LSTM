"""Deseasonalized ERA5 inputs for HiGNN (datamodule flag `era5_anomaly`).

With a training climatology C (xr.Dataset, one [month, lat, lon] variable per ERA5 input) the
heterograph dataset normalizes var − C[var][month] instead of var for every variable in C,
before the per-pixel z-score. twsa is not in C and keeps its own flag (`twsa_anomaly`).
Without the flag nothing changes.

The dataset is built with the real TWSAGraphDataset constructor and use_ocean_nodes=False
(no clustering, no graph cache). sst is ocean-only, so it never reaches the land node
features; the checks therefore read the normalized grid in ds_timeslice, from which the
land and ocean features are cut.
"""
import numpy as np
import pytest
import torch
import xarray as xr

import datamodules.TWSA_heterograph_datamodule as dmhetero_module
from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from datamodules.TWSA_heterograph_dataset import TWSAGraphDataset
from synthetic_grid import LAND, LAT, make_ds

SEQ, H = 3, 2
ERA5_VARS = ['sst', 'ro', 't2m']  # every input of make_ds() except twsa (lsm is not an input in the configs)
MASKS = {'sst': ~LAND,  # only_ocean_variables
         'ro': LAND,  # only_land_variables
         't2m': np.ones_like(LAND),  # everything else
         'twsa': LAND & (LAT > -60)[:, None]}


def make_era5_anomaly_ds():
    """make_ds() plus non-seasonal noise in twsa and the ERA5 inputs: their pure 12-month sines
    are removed entirely by a climatology, which would leave a zero anomaly (std 0 -> NaN -> 0)."""
    ds = make_ds()
    rng = np.random.default_rng(0)
    for var in ['twsa'] + ERA5_VARS:
        ds[var] = ds[var] + rng.normal(0, 0.5, ds[var].shape).astype(np.float32)
    return ds


def era5_climatology_24m():
    """C from the first 24 of the 36 months, deliberately not the dataset's own climatology."""
    return (make_era5_anomaly_ds()[ERA5_VARS].isel(time=slice(0, 24))
            .groupby('time.month').mean('time').astype(np.float32))


def twsa_climatology_24m():
    return (make_era5_anomaly_ds()['twsa'].isel(time=slice(0, 24))
            .groupby('time.month').mean('time').astype(np.float32))


def build_hetero(**kw):
    ds = make_era5_anomaly_ds()
    return TWSAGraphDataset(ds, (ds.time.values[0], ds.time.values[-1]), delay_time=0,
                            input_seq_len=SEQ, output_seq_len=H, land_ocean_mask=ds['lsm'] > 0.8,
                            use_ocean_nodes=False, **kw)


def normalized(obj, var):
    """normalized var on the full grid, [time, lat, lon]"""
    return obj.ds_timeslice[var].transpose('time', 'lat', 'lon').values


def expected_z(var, C=None):
    """Independent oracle: (raw − C[month] − mean)/std on the variable's mask and 0 outside it,
    with mean/std of that series and the datasets' std floor (1% of the median std)."""
    ds = make_era5_anomaly_ds()
    mask = MASKS[var]
    a = ds[var].values.astype(np.float64)
    if C is not None:
        a = a - C.sel(month=ds.time.dt.month).values
    a = a[:, mask]
    mean, std = a.mean(0), a.std(0)
    z = np.zeros(ds[var].shape, dtype=np.float32)
    z[:, mask] = (a - mean) / np.maximum(std, np.median(std) * 0.01)
    return z


# ---------------------------------------------------------------- E1: ERA5 anomaly normalization


@pytest.mark.parametrize('anomaly', [False, True])
@pytest.mark.parametrize('var', ERA5_VARS)
def test_era5_input_is_normalized_as_an_anomaly(var, anomaly):
    """Flag off builds without the new kwarg, so that case checks the oracle against today's code.
    atol 1e-4: the dataset works in float32 and sst/t2m sit at ~280-290 K (observed error ~3e-5);
    a missing or wrong C moves z by O(1)."""
    C = era5_climatology_24m() if anomaly else None
    obj = build_hetero(era5_climatology=C) if anomaly else build_hetero()
    np.testing.assert_allclose(normalized(obj, var), expected_z(var, None if C is None else C[var]),
                               rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('var', ERA5_VARS)
def test_val_branch_subtracts_the_given_era5_climatology(var):
    """val, test and GRACE reuse the training scaling params: there a +5 shift of C moves the
    normalized input by −5/std (in the training branch it would cancel in the mean)."""
    C = era5_climatology_24m()
    train = build_hetero(era5_climatology=C)
    val = build_hetero(era5_climatology=C + 5.0, scaling_mean_std=train.scaling_params)
    mask = MASKS[var]
    std = train.scaling_params[train.input_vars.index(var)][1].values[mask]
    np.testing.assert_allclose(normalized(val, var)[:, mask], normalized(train, var)[:, mask] - 5.0 / std,
                               rtol=1e-4, atol=1e-4)
    assert (normalized(val, var)[:, ~mask] == 0).all()


# ---------------------------------------------------------------- E2: twsa keeps its own flag


@pytest.mark.parametrize('twsa_anomaly', [False, True])
def test_twsa_is_independent_of_era5_anomaly(twsa_anomaly):
    C_twsa = twsa_climatology_24m() if twsa_anomaly else None
    on = build_hetero(era5_climatology=era5_climatology_24m(), twsa_climatology=C_twsa)
    if twsa_anomaly:
        np.testing.assert_allclose(normalized(on, 'twsa'), expected_z('twsa', C_twsa), rtol=1e-4, atol=1e-5)
    else:
        off = build_hetero()
        np.testing.assert_array_equal(normalized(on, 'twsa'), normalized(off, 'twsa'))
        assert torch.equal(on.target_series, off.target_series)
        assert on.twsa_offset is None


# ---------------------------------------------------------------- E3: flag off changes nothing


def test_flag_off_is_bit_identical():
    before, off = build_hetero(), build_hetero(era5_climatology=None)
    assert torch.equal(before.features, off.features)
    xr.testing.assert_identical(before.ds_timeslice, off.ds_timeslice)


# ---------------------------------------------------------------- E4: the datamodule owns C

TRAIN_STEPS = 19  # the (0.5, 0.25, 0.25) split of 36 months trains on steps 0..18

DM_KW = dict(train_val_test_split=(0.5, 0.25, 0.25), graph_variant='no_ocean',
             batch_size=2, num_workers=0, input_vars=['twsa'] + ERA5_VARS)


@pytest.fixture
def nc_path(tmp_path, monkeypatch):
    """Synthetic grid on disk as float64 (like 02_RecTWSA_era5.nc), the ERA5 inputs +100 after
    the training period, sst NaN in one land cell, and a GRACE_era5.nc for GRACE_dataloader."""
    monkeypatch.chdir(tmp_path)  # the datamodule creates graphs/ and GRACE_dataloader reads data/ in the cwd
    ds = make_era5_anomaly_ds()
    for var in ERA5_VARS:
        ds[var] = ds[var].astype(np.float64)
        ds[var].values[TRAIN_STEPS:] += 100.0
    ds['sst'].values[:, 0, 0] = np.nan  # sst is NaN on land
    ds.to_netcdf(tmp_path / 'syn.nc')
    (tmp_path / 'data' / 'processed').mkdir(parents=True)
    make_era5_anomaly_ds().to_netcdf(tmp_path / 'data' / 'processed' / 'GRACE_era5.nc')
    return str(tmp_path / 'syn.nc')


def test_datamodule_owns_the_era5_training_climatology_as_float32(nc_path):
    dm = TWSA_HeteroGraphDatamodule(ds_path=nc_path, era5_anomaly=True, **DM_KW)
    train = xr.open_dataset(nc_path).sel(time=slice(*dm.train_val_test_periods['train']))
    assert train.time.size == TRAIN_STEPS  # the +100 never enters the training mean
    C = dm.era5_climatology
    assert set(C.data_vars) == set(dm.input_vars) - {'twsa'} == set(ERA5_VARS)
    for var in ERA5_VARS:
        assert C[var].dims == ('month', 'lat', 'lon')
        assert C[var].dtype == np.float32, var  # float64 C would promote the inputs in ds_timeslice
        np.testing.assert_allclose(C[var].values, train[var].groupby('time.month').mean('time').fillna(0).values,
                                   rtol=1e-6, atol=1e-6, err_msg=var)
    assert (C['sst'].values[:, 0, 0] == 0).all()  # NaN filled with 0


def test_twsa_only_inputs_build_without_era5_climatology(nc_path):
    """input_vars=['twsa'] leaves no ERA5 variable to average, so with the flag off the
    datamodule must not try to build the ERA5 climatology."""
    dm = TWSA_HeteroGraphDatamodule(ds_path=nc_path, era5_anomaly=False,
                                    **{**DM_KW, 'input_vars': ['twsa']})
    assert dm.input_vars == ['twsa']
    assert dm.era5_climatology is None


NOT_PASSED = 'not passed'


@pytest.mark.parametrize('flag', [False, True])
def test_every_dataset_construction_site_gets_the_era5_climatology_only_when_on(nc_path, monkeypatch, flag):
    """fit train, fit val, validate, test, predict and GRACE must all pass era5_climatology
    explicitly (None when off)."""
    seen = []

    class Recorder:
        scaling_params = ocean_clustering = None

        def __init__(self, ds, *args, **kw):
            self.time_period = kw['time_period']  # read by GRACE_dataloader
            seen.append(kw.get('era5_climatology', NOT_PASSED))

    monkeypatch.setattr(dmhetero_module, 'TWSAGraphHeteroData', Recorder)
    dm = TWSA_HeteroGraphDatamodule(ds_path=nc_path, era5_anomaly=flag, **DM_KW)
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

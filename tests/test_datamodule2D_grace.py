"""TWSA_Datamodule2D.GRACE_dataloader: ConvLSTM evaluation on GRACE TWSA.

GRACE_era5.nc is on the same grid as the reconstruction, but it has missing months,
extra variables and its own variable order. The loader must subset and reorder to
the training variables, scale with the training mean/std, use the training
climatology in anomaly mode, and only build windows over consecutive months.
"""
import numpy as np
import pytest
import xarray as xr
from torch.utils.data import DataLoader

import datamodules.TWSA_datamodule2D as dm2d_module
from datamodules.TWSA_datamodule2D import TWSA_Datamodule2D
from synthetic_grid import LAND, LAT, make_ds

INPUT_SEQ_LEN, PRED_LEN = 3, 2
INPUT_VARS = ['twsa', 'sst', 'ro', 't2m', 'lsm']
MISSING = ['2003-08', '2003-09']  # like GRACE 2018-08/09
CELL = (3, 0)  # land, lat 40.5
SCORED = LAND & (LAT > -60)[:, None]  # land outside Antarctica

# GRACE record 2003-01 -> 2005-12 without MISSING: runs of 7 (2003-01..07) and 27 months
# (2003-10..2005-12); a window covers 3 + 0 + 2 = 5 months -> (7-5+1) + (27-5+1) windows
N_WINDOWS = 3 + 23


def expected_target_months():
    """Target months of every gap-free window, in order, by plain date arithmetic."""
    starts = [np.datetime64('2003-01', 'M') + k for k in range(7 - 5 + 1)]
    starts += [np.datetime64('2003-10', 'M') + k for k in range(27 - 5 + 1)]
    return [[s + INPUT_SEQ_LEN + k for k in range(PRED_LEN)] for s in starts]


def make_rec():
    """make_ds() plus non-seasonal noise in twsa: its pure 12-month sine would leave a zero
    training anomaly (std 0), so GRACE anomalies would be divided by zero."""
    ds = make_ds()
    rng = np.random.default_rng(0)
    ds['twsa'] = ds['twsa'] + rng.normal(0, 0.5, ds['twsa'].shape).astype(np.float32)
    return ds


def make_grace():
    """make_ds() moved to 2003-01, with other twsa values, two missing months,
    an extra variable z and the variables in another order than rec.nc."""
    ds = make_ds()
    ds = ds.assign_coords(time=np.arange('2003-01', '2006-01', dtype='datetime64[M]').astype('datetime64[ns]'))
    rng = np.random.default_rng(1)
    ds['twsa'] = ds['twsa'] * 1.5 + 3.0 + rng.normal(0, 0.5, ds['twsa'].shape).astype(np.float32)
    ds['z'] = ds['t2m'] * 0 + 7.0
    ds = ds.drop_sel(time=[np.datetime64(m, 'ns') for m in MISSING])
    return ds[['z', 't2m', 'lsm', 'ro', 'twsa', 'sst']]


@pytest.fixture
def paths(tmp_path, monkeypatch):
    make_rec().to_netcdf(tmp_path / 'rec.nc')
    make_grace().to_netcdf(tmp_path / 'grace.nc')
    monkeypatch.setattr(dm2d_module, 'GRACE_PATH', str(tmp_path / 'grace.nc'))
    return str(tmp_path / 'rec.nc'), str(tmp_path / 'grace.nc')


def build_dm(rec_path, twsa_anomaly=False):
    dm = TWSA_Datamodule2D(ds_path=rec_path, input_vars=list(INPUT_VARS), input_seq_len=INPUT_SEQ_LEN,
                           output_seq_len=PRED_LEN, delay=0, twsa_anomaly=twsa_anomaly,
                           batch_size=2, num_workers=0)
    dm.setup('fit')
    return dm


def raw_grace_twsa():
    return make_grace()['twsa'].values


def test_grace_dataloader_covers_the_gap_free_windows(paths):
    dm = build_dm(paths[0])
    loader = dm.GRACE_dataloader()
    assert isinstance(loader, DataLoader)
    assert loader.dataset is dm.data_pred
    assert len(dm.data_pred) == N_WINDOWS


def test_variables_are_subset_and_in_training_order(paths, capsys):
    dm = build_dm(paths[0])
    capsys.readouterr()
    dm.GRACE_dataloader()
    assert dm.data_pred.input_vars == dm.data_train.input_vars
    assert 'z' not in dm.data_pred.input_vars
    # lsm has (0, 1) as scaling params and is skipped by design, in every prediction dataset
    skipped = [line for line in capsys.readouterr().out.splitlines() if 'skipping normalization' in line]
    assert skipped == ['Variable lsm not found in scaling parameters, skipping normalization']


def test_training_scaling_is_used(paths):
    dm = build_dm(paths[0])
    dm.GRACE_dataloader()
    i = dm.input_vars.index('twsa')
    mean, std = (p.values[CELL] for p in dm.scaling_mean_std[i])
    t = 10
    expected = (raw_grace_twsa()[t][CELL] - mean) / std
    got = dm.data_pred.features[t, dm.data_pred.input_vars.index('twsa')][CELL].item()
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def collect(loader, key):
    return np.concatenate([b[key].numpy() for b in loader])


def test_anomaly_mode_carries_the_training_climatology_of_the_target_months(paths):
    dm = build_dm(paths[0], twsa_anomaly=True)
    offset = collect(dm.GRACE_dataloader(), 'twsa_offset')
    months = [[int(str(m)[5:7]) for m in w] for w in expected_target_months()]
    expected = np.stack([dm.twsa_climatology.sel(month=m).values for m in months])
    np.testing.assert_array_equal(offset, expected)


def test_raw_mode_has_no_offset(paths):
    dm = build_dm(paths[0], twsa_anomaly=False)
    assert all('twsa_offset' not in b for b in dm.GRACE_dataloader())


@pytest.mark.parametrize('twsa_anomaly', [False, True])
def test_round_trip_to_cm_gives_the_raw_grace_twsa(paths, twsa_anomaly):
    dm = build_dm(paths[0], twsa_anomaly=twsa_anomaly)
    loader = dm.GRACE_dataloader()
    y = dm.unnormalize_twsa(collect(loader, 'y_timestep'))
    if twsa_anomaly:
        y = y + collect(loader, 'twsa_offset')

    grace = make_grace()['twsa']
    expected = np.stack([grace.sel(time=np.array(w).astype('datetime64[ns]')).values
                         for w in expected_target_months()])
    np.testing.assert_allclose(y[..., SCORED], expected[..., SCORED], rtol=1e-4, atol=1e-4)

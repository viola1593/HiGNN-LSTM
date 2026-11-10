"""Anomaly-space TWSA (datamodule flag `twsa_anomaly`).

With a training climatology C [month, lat, lon] in cm the datasets normalize
twsa − C[month] instead of twsa, attach C of the target months as `twsa_offset`,
and predict_step adds it back, so predictions stay absolute cm. Evaluation takes
C from the datamodule. Without the flag nothing changes.

The heterograph dataset is built with the real TWSAGraphDataset constructor and
use_ocean_nodes=False (no clustering, no graph cache); TWSAGraphHeteroData's
__getitem__ is borrowed, so graph construction and the hydrobasins are not needed.
"""
import functools
from types import SimpleNamespace

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
import xarray as xr

import datamodules.TWSA_datamodule2D as dm2d_module
import datamodules.TWSA_heterograph_datamodule as dmhetero_module
from datamodules.TWSA_datamodule2D import TWSA_Datamodule2D, collate_fn
from datamodules.TWSA_dataset2D import TWSA2DDataset
from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from datamodules.TWSA_heterograph_dataset import TWSAGraphDataset, TWSAGraphHeteroData
from models.pl_convlstm import ConvLSTMModule
from models.pl_hmmodel import HMModule
from synthetic_grid import LAND, LAT, make_ds
from utils import eval_metrics

SEQ, H = 3, 2
SCORED = (LAND & (LAT > -60)[:, None])[LAND]  # twsa mask of both datasets, over the land nodes


def make_anomaly_ds():
    """make_ds() plus non-seasonal noise in twsa: its pure 12-month sine is removed
    entirely by its own climatology, which would leave a zero anomaly."""
    ds = make_ds()
    rng = np.random.default_rng(0)
    ds['twsa'] = ds['twsa'] + rng.normal(0, 0.5, ds['twsa'].shape).astype(np.float32)
    return ds


def climatology_24m():
    """C from the first 24 of the 36 months, deliberately not the dataset's own climatology."""
    return make_anomaly_ds()['twsa'].isel(time=slice(0, 24)).groupby('time.month').mean('time')


def build_2d(**kw):
    ds = make_anomaly_ds()
    return TWSA2DDataset(ds, time_period=(ds.time.values[0], ds.time.values[-1]), delay=0,
                         input_seq_len=SEQ, output_seq_len=H, input_vars=list(ds.data_vars), **kw)


def build_hetero(**kw):
    ds = make_anomaly_ds()
    obj = TWSAGraphDataset(ds, (ds.time.values[0], ds.time.values[-1]), delay_time=0,
                           input_seq_len=SEQ, output_seq_len=H, land_ocean_mask=ds['lsm'] > 0.8,
                           use_ocean_nodes=False, **kw)
    obj.data = pyg.data.HeteroData()  # the only TWSAGraphHeteroData.__init__ attribute __getitem__ reads
    return obj


# kind -> (builder, normalized twsa as [T, N_land] in row-major land-node order)
KINDS = {
    '2d': (build_2d, lambda o: o.features[:, o.input_vars.index('twsa')].numpy()[:, LAND]),
    'hetero': (build_hetero, lambda o: o.features[:, :, o.target_index].T.numpy()),
}


def expected_z(C=None):
    """Independent oracle: (raw − C[month] − mean)/std on the scored nodes, with mean/std of
    that series and the datasets' std floor (1% of the median std)."""
    ds = make_anomaly_ds()
    a = ds['twsa'].values[:, LAND]
    if C is not None:
        a = a - C.sel(month=ds.time.dt.month).values[:, LAND]
    a = a[:, SCORED]
    mean, std = a.mean(0), a.std(0)
    return (a - mean) / np.maximum(std, np.median(std) * 0.01)


# ---------------------------------------------------------------- A1: anomaly normalization


@pytest.mark.parametrize('anomaly', [False, True])
@pytest.mark.parametrize('kind', KINDS)
def test_twsa_is_normalized_as_an_anomaly(kind, anomaly):
    """Flag off builds without the new kwarg, so that case pins today's behaviour."""
    build, twsa = KINDS[kind]
    C = climatology_24m() if anomaly else None
    obj = build(twsa_climatology=C) if anomaly else build()
    z = twsa(obj)
    np.testing.assert_allclose(z[:, SCORED], expected_z(C), rtol=1e-4, atol=1e-5)
    assert (z[:, ~SCORED] == 0).all()  # Antarctica row stays masked


# ---------------------------------------------------------------- A2: val branch


@pytest.mark.parametrize('kind', KINDS)
def test_val_branch_subtracts_the_given_climatology(kind):
    """With the training scaling params a +5 cm shift of C moves the normalized twsa by
    −5/std (in the training branch it would cancel in the mean)."""
    build, twsa = KINDS[kind]
    C = climatology_24m()
    train = build(twsa_climatology=C)
    val = build(twsa_climatology=C + 5.0, scaling_mean_std=train.scaling_params)
    std = train.scaling_params[0][1].values[LAND][SCORED]  # twsa is variable 0
    np.testing.assert_allclose(twsa(val)[:, SCORED], twsa(train)[:, SCORED] - 5.0 / std,
                               rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- A3: only twsa changes


@pytest.mark.parametrize('kind', KINDS)
def test_flag_off_is_unchanged_and_flag_on_touches_only_twsa(kind):
    build, _ = KINDS[kind]
    before, off, on = build(), build(twsa_climatology=None), build(twsa_climatology=climatology_24m())
    assert torch.equal(before.features, off.features)
    for var in before.ds_timeslice.data_vars:
        if var != 'twsa':
            np.testing.assert_array_equal(on.ds_timeslice[var].values, off.ds_timeslice[var].values,
                                          err_msg=var)


# ---------------------------------------------------------------- A4: twsa_offset per sample


def target_offset(C, t):
    """C of the H target months of the sample whose targets start at time index t (delay 0),
    [H, lat, lon]."""
    return C.sel(month=make_anomaly_ds().time.dt.month.values[t:t + H]).values


def test_hetero_sample_carries_the_climatology_of_its_target_months():
    C = climatology_24m()
    on, off = build_hetero(twsa_climatology=C), build_hetero()
    for idx in range(len(on)):
        sample = TWSAGraphHeteroData.__getitem__(on, idx)
        torch.testing.assert_close(sample['grid'].twsa_offset,  # also checks float32
                                   torch.from_numpy(target_offset(C, on.samples[idx])[:, LAND].T.copy()))
        assert 'twsa_offset' not in TWSAGraphHeteroData.__getitem__(off, idx)['grid']


def test_2d_sample_carries_the_climatology_of_its_target_months():
    C = climatology_24m()
    on, off = build_2d(twsa_climatology=C), build_2d()
    for idx in range(len(on)):
        torch.testing.assert_close(on[idx]['twsa_offset'],  # also checks float32
                                   torch.from_numpy(target_offset(C, on.samples[idx])))
        assert 'twsa_offset' not in off[idx]
    assert collate_fn([on[0], on[1]])['twsa_offset'].shape == (2, H, *LAND.shape)
    assert 'twsa_offset' not in collate_fn([off[0], off[1]])


# ---------------------------------------------------------------- A5: predictions in absolute cm


class PersistenceHM(pl.LightningModule):
    """HMModule's own predict_step and unnormalizer; forward repeats the last normalized twsa."""
    predict_step = HMModule.predict_step
    _twsa_unnormalizer = HMModule._twsa_unnormalizer

    def forward(self, data):
        return data['grid'].x[:, -1, 0:1].expand(-1, H), None, None


class PersistenceConvLSTM(pl.LightningModule):
    """ConvLSTMModule's own predict_step and unnormalizer; forward repeats the last normalized twsa."""
    predict_step = ConvLSTMModule.predict_step
    _twsa_unnormalizer = ConvLSTMModule._twsa_unnormalizer

    def forward(self, x):
        return x[:, -1:, 0].expand(-1, H, -1, -1)


def attach(model, datamodule_cls, **attrs):
    """model.trainer.datamodule is a namespace with the real unnormalize_twsa bound to it."""
    dm = SimpleNamespace(**attrs)
    dm.unnormalize_twsa = functools.partial(datamodule_cls.unnormalize_twsa, dm)
    model.trainer = SimpleNamespace(datamodule=dm)  # LightningModule.trainer's setter only stores it
    return model


def expected_cm(C, samples):
    """truth raw[t:t+H] and persistence C[target] + (raw − C)[t−1] per sample, [B, N_land, H] in cm;
    persistence of raw twsa when C is None."""
    ds = make_anomaly_ds()
    raw = ds['twsa'].values[:, LAND]
    c = np.zeros_like(raw) if C is None else C.sel(month=ds.time.dt.month).values[:, LAND]
    return (np.stack([raw[t:t + H].T for t in samples]),
            np.stack([(c[t:t + H] + (raw - c)[t - 1]).T for t in samples]))


@pytest.mark.parametrize('anomaly', [False, True])
def test_hignn_predict_step_returns_absolute_cm(anomaly):
    C = climatology_24m() if anomaly else None
    obj = build_hetero(twsa_climatology=C) if anomaly else build_hetero()
    model = attach(PersistenceHM(), TWSA_HeteroGraphDatamodule, scaling_mean_std=obj.scaling_params,
                   get_num_nodes=lambda: int(LAND.sum()), get_land_ocean_mask=lambda: obj.land_ocean_mask)
    batch = pyg.data.Batch.from_data_list([TWSAGraphHeteroData.__getitem__(obj, i) for i in range(len(obj))])
    out, y = model.predict_step(batch, 0)[:2]

    truth, persistence = expected_cm(C, obj.samples)
    shape = (len(obj), int(LAND.sum()), H)  # [B·N, H] -> [B, N_land, H]
    np.testing.assert_allclose(y.reshape(shape).numpy()[:, SCORED], truth[:, SCORED], atol=1e-5)
    np.testing.assert_allclose(out.reshape(shape).numpy()[:, SCORED], persistence[:, SCORED], atol=1e-5)


@pytest.mark.parametrize('anomaly', [False, True])
def test_convlstm_predict_step_returns_absolute_cm(anomaly):
    """y == truth alone would pass if the 2D dataset silently ignored C; only the persistence
    check tells "never subtracted" apart from "subtracted and added back"."""
    C = climatology_24m() if anomaly else None
    obj = build_2d(twsa_climatology=C) if anomaly else build_2d()
    model = attach(PersistenceConvLSTM(), TWSA_Datamodule2D, scaling_mean_std=obj.scaling_params)
    out, y = model.predict_step(collate_fn([obj[i] for i in range(len(obj))]), 0)[:2]

    truth, persistence = expected_cm(C, obj.samples)
    nodes = lambda a: a.permute(0, 2, 3, 1)[:, LAND][:, SCORED].numpy()  # [B, H, lat, lon] -> [B, N_scored, H]
    np.testing.assert_allclose(nodes(y), truth[:, SCORED], atol=1e-5)
    np.testing.assert_allclose(nodes(out), persistence[:, SCORED], atol=1e-5)


# ---------------------------------------------------------------- A6: the datamodule owns C


@pytest.fixture
def nc_path(tmp_path, monkeypatch):
    """Synthetic grid on disk as float64 (like 02_RecTWSA_era5.nc), +100 cm after the
    training period, and one all-NaN cell."""
    monkeypatch.chdir(tmp_path)  # the heterograph datamodule creates graphs/ in the cwd
    ds = make_anomaly_ds()
    ds['twsa'] = ds['twsa'].astype(np.float64)
    ds['twsa'].values[24:] += 100.0  # the (0.5, 0.25, 0.25) training period ends at step 18
    ds['twsa'].values[:, 0, 2] = np.nan
    ds.to_netcdf(tmp_path / 'syn.nc')
    return str(tmp_path / 'syn.nc')


DM_KW = dict(train_val_test_split=(0.5, 0.25, 0.25), graph_variant='no_ocean',
             batch_size=2, num_workers=0)

DATAMODULES = [pytest.param(TWSA_HeteroGraphDatamodule, id='hetero'),
               pytest.param(TWSA_Datamodule2D, id='2d')]


@pytest.mark.parametrize('flag', [False, True])
@pytest.mark.parametrize('Datamodule', DATAMODULES)
def test_datamodule_owns_the_raw_training_climatology_as_float32(nc_path, Datamodule, flag):
    dm = Datamodule(ds_path=nc_path, twsa_anomaly=flag, **DM_KW)
    train = xr.open_dataset(nc_path)['twsa'].sel(time=slice(*dm.train_val_test_periods['train']))
    assert train.time.size < 24  # the +100 cm never enters the training mean
    C = dm.twsa_climatology
    assert C.dims == ('month', 'lat', 'lon')
    assert C.dtype == np.float32  # float64 C would promote twsa in ds_timeslice
    np.testing.assert_allclose(C.values, train.groupby('time.month').mean('time').fillna(0).values,
                               rtol=1e-6)
    assert (C.values[:, 0, 2] == 0).all()  # NaN filled with 0


# ---------------------------------------------------------------- A7: evaluation uses the datamodule's C


def test_get_climatology_uses_the_datamodules_climatology_not_a_stale_cache(tmp_path):
    ds = make_anomaly_ds()
    C = ds['twsa'].groupby('time.month').mean('time')
    dm = SimpleNamespace(twsa_climatology=C, land_ocean_mask=ds['lsm'] > 0.8)  # deliberately no data_train
    torch.save(np.full((12, 3), 999.0, dtype=np.float32), tmp_path / 'climatology.pt')  # stale cache
    out = eval_metrics.get_climatology(dm, {'experiment': {'save_dir': str(tmp_path)}})
    assert out.dtype == np.float32 and out.shape == (12, int(LAND.sum()))
    np.testing.assert_array_equal(out, C.values[:, LAND].astype(np.float32))


# ---------------------------------------------------------------- A8: wiring of every construction site

NOT_PASSED = 'not passed'


@pytest.mark.parametrize('flag', [False, True])
@pytest.mark.parametrize('Datamodule, module, name', [
    pytest.param(TWSA_HeteroGraphDatamodule, dmhetero_module, 'TWSAGraphHeteroData', id='hetero'),
    pytest.param(TWSA_Datamodule2D, dm2d_module, 'TWSA2DDataset', id='2d')])
def test_every_dataset_construction_site_gets_the_climatology_only_when_on(
        nc_path, monkeypatch, Datamodule, module, name, flag):
    """fit train, fit val, validate, test and predict must all pass twsa_climatology
    explicitly (None when off). GRACE_dataloader reads a hardcoded file and is checked by review."""
    seen = []

    class Recorder:
        scaling_params = ocean_clustering = None

        def __init__(self, *args, **kw):
            seen.append(kw.get('twsa_climatology', NOT_PASSED))

    monkeypatch.setattr(module, name, Recorder)
    dm = Datamodule(ds_path=nc_path, twsa_anomaly=flag, **DM_KW)
    dm.setup('fit')  # train + val
    dm.data_val = None
    dm.setup('validate')
    dm.setup('test')
    dm.predict_dataloader(period=dm.train_val_test_periods['val'])
    assert len(seen) == 5
    assert not any(c is NOT_PASSED for c in seen), 'a construction site omits twsa_climatology'
    expected = dm.twsa_climatology if flag else None
    assert all(c is expected for c in seen)


def test_grace_dataloader_reads_the_02a_output_and_keeps_its_lat(nc_path, monkeypatch, tmp_path):
    """GRACE_dataloader reads data/processed/GRACE_era5.nc (02a output, already on the
    02b grid with lat descending) and must not overwrite its lat values."""
    grace = make_anomaly_ds().isel(lat=slice(None, None, -1))  # lat descending, top row 40.5
    (tmp_path / 'data' / 'processed').mkdir(parents=True)
    grace.to_netcdf(tmp_path / 'data' / 'processed' / 'GRACE_era5.nc')  # cwd is tmp_path
    seen = []

    class Recorder:
        scaling_params = ocean_clustering = None

        def __init__(self, ds, *args, **kw):
            self.time_period = kw.get('time_period', args[0] if args else None)
            seen.append(ds)

    monkeypatch.setattr(dmhetero_module, 'TWSAGraphHeteroData', Recorder)
    dm = TWSA_HeteroGraphDatamodule(ds_path=nc_path, **DM_KW)
    dm.setup('fit')
    dm.GRACE_dataloader()
    np.testing.assert_array_equal(seen[-1]['lat'].values, grace['lat'].values)

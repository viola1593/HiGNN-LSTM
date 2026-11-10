"""GRACE_dataloader(twsa_climatology=C_grace): on GRACE the anomaly models use the GRACE
climatology instead of the reconstruction training climatology, without retraining.

The datasets use the one twsa_climatology argument both for the input subtraction and for
twsa_offset (tested in test_twsa_anomaly.py), so these tests check what the datamodules pass:
C_grace when given, else self.twsa_climatology; the ERA5 training climatology and the
training scaling stats unchanged.
"""
import numpy as np
import pytest

import datamodules.TWSA_datamodule2D as dm2d_module
import datamodules.TWSA_heterograph_datamodule as dmhetero_module
from datamodules.TWSA_datamodule2D import TWSA_Datamodule2D
from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from test_era5_anomaly import ERA5_VARS, H, SEQ, make_era5_anomaly_ds, twsa_climatology_24m

DM_KW = {
    'hetero': dict(train_val_test_split=(0.5, 0.25, 0.25), graph_variant='no_ocean',
                   batch_size=2, num_workers=0, input_vars=['twsa'] + ERA5_VARS),
    '2d': dict(train_val_test_split=(0.5, 0.25, 0.25), input_seq_len=SEQ, output_seq_len=H, delay=0,
               batch_size=2, num_workers=0, input_vars=['twsa'] + ERA5_VARS + ['lsm']),
}
DATAMODULES = [pytest.param(TWSA_HeteroGraphDatamodule, dmhetero_module, 'TWSAGraphHeteroData', 'hetero', id='hetero'),
               pytest.param(TWSA_Datamodule2D, dm2d_module, 'TWSA2DDataset', '2d', id='2d')]


@pytest.fixture
def nc_path(tmp_path, monkeypatch):
    """Synthetic reconstruction on disk and a GRACE file where each GRACE_dataloader reads it."""
    monkeypatch.chdir(tmp_path)  # the heterograph datamodule creates graphs/ and reads data/ in the cwd
    make_era5_anomaly_ds().to_netcdf(tmp_path / 'syn.nc')
    (tmp_path / 'data' / 'processed').mkdir(parents=True)
    make_era5_anomaly_ds().to_netcdf(tmp_path / 'data' / 'processed' / 'GRACE_era5.nc')
    monkeypatch.setattr(dm2d_module, 'GRACE_PATH', str(tmp_path / 'data' / 'processed' / 'GRACE_era5.nc'))
    return str(tmp_path / 'syn.nc')


def grace_kwargs(nc_path, monkeypatch, Datamodule, module, name, key, dm_kw, **grace_kw):
    """Build the datamodule with recorded datasets, call GRACE_dataloader and return
    (datamodule, kwargs of the GRACE dataset)."""
    seen = []

    class Recorder:
        scaling_params = ocean_clustering = None

        def __init__(self, ds, *args, **kw):
            self.time_period = kw['time_period']  # read by the heterograph GRACE_dataloader
            seen.append(kw)

    monkeypatch.setattr(module, name, Recorder)
    dm = Datamodule(ds_path=nc_path, **{**DM_KW[key], **dm_kw})
    dm.setup('fit')
    dm.GRACE_dataloader(**grace_kw)
    return dm, seen[-1]


@pytest.mark.parametrize('Datamodule, module, name, key', DATAMODULES)
def test_given_climatology_replaces_the_training_climatology_on_grace(nc_path, monkeypatch,
                                                                       Datamodule, module, name, key):
    C_grace = twsa_climatology_24m() + 7.0
    dm, kw = grace_kwargs(nc_path, monkeypatch, Datamodule, module, name, key,
                          dict(twsa_anomaly=True, era5_anomaly=True), twsa_climatology=C_grace)
    assert kw['twsa_climatology'] is C_grace
    assert kw['era5_climatology'] is dm.era5_climatology  # ERA5 is the same product in both datasets
    assert kw['scaling_mean_std'] is dm.scaling_mean_std  # training scaling stats


@pytest.mark.parametrize('Datamodule, module, name, key', DATAMODULES)
def test_default_keeps_the_training_climatology_on_grace(nc_path, monkeypatch, Datamodule, module, name, key):
    dm, kw = grace_kwargs(nc_path, monkeypatch, Datamodule, module, name, key, dict(twsa_anomaly=True))
    assert kw['twsa_climatology'] is dm.twsa_climatology


@pytest.mark.parametrize('Datamodule, module, name, key', DATAMODULES)
def test_raw_models_get_no_climatology_by_default(nc_path, monkeypatch, Datamodule, module, name, key):
    _, kw = grace_kwargs(nc_path, monkeypatch, Datamodule, module, name, key, dict(twsa_anomaly=False))
    assert kw['twsa_climatology'] is None


@pytest.mark.parametrize('Datamodule, module, name, key', DATAMODULES)
def test_climatology_for_a_raw_model_is_an_error(nc_path, monkeypatch, Datamodule, module, name, key):
    """A model trained on raw twsa cannot take anomaly inputs, so a C_grace there is a caller mistake."""
    with pytest.raises(ValueError, match='twsa_anomaly'):
        grace_kwargs(nc_path, monkeypatch, Datamodule, module, name, key, dict(twsa_anomaly=False),
                     twsa_climatology=twsa_climatology_24m())

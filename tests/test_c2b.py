"""Batch C2b: compare_models.py -- redundant test epoch (#20) and the Li et al. forecast
path / read order (#26)."""

import inspect
import pathlib
import re
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import xarray as xr
from torch.utils.data import DataLoader

matplotlib.use("Agg")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_models


# ---------------------------------------------------------------- #20: redundant test epoch


def test_get_predictions_runs_no_test_epoch(monkeypatch, tmp_path):
    calls = []
    trainer = SimpleNamespace(test=lambda *a, **kw: calls.append("test"),
                              predict=lambda model, dl: calls.append("predict") or ["preds"])
    monkeypatch.setattr(compare_models, "build_model_from_checkpoint", lambda *a: "model")
    dm = SimpleNamespace(train_val_test_periods={"test": ("2020", "2021")},
                         predict_dataloader=lambda period: "dl")
    config = {"experiment": {"save_dir": str(tmp_path)}}

    preds = compare_models.get_predictions(dm, trainer, config, str(tmp_path), dataset="test")

    assert calls == ["predict"]
    assert preds == ["preds"]


def test_trainer_datamodule_survives_predict_without_test():
    """The premise of #20: evaluate() assigns trainer.datamodule directly, and a
    dataloader-only predict() must keep it, since predict_step unnormalizes through it."""

    class DM(pl.LightningDataModule):
        def unnormalize_twsa(self, x):
            return x * 100

    class M(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(1, 1)

        def predict_step(self, batch, batch_idx):
            return self.trainer.datamodule.unnormalize_twsa(torch.ones(1))

    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    dm = DM()
    trainer.datamodule = dm
    out = trainer.predict(M(), DataLoader(torch.zeros(2, 1), batch_size=1))
    assert trainer.datamodule is dm
    assert [o.item() for o in out] == [100.0, 100.0]


# ---------------------------------------------------------------- #26: forecast path


def test_fcast_default_path_is_what_prepareFCast_writes(monkeypatch):
    src = (ROOT / "scripts" / "preprocess_data" / "04_prepareFCast.py").read_text()
    produced = re.search(r'outputfolder\s*=\s*"([^"]+)"', src).group(1) + "/combined.nc"

    assert inspect.signature(compare_models.evaluate).parameters["FCast_file_path"].default == produced

    seen = {}
    monkeypatch.setattr(compare_models, "evaluate", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(sys, "argv", ["compare_models.py", "--exp_path", "x"])
    compare_models.start()
    assert seen["FCast_file_path"] == produced


# ---------------------------------------------------------------- #26: forecast read order

T_OUT, T_IN, START = 6, 12, 17  # leads, input months, first window predicting Nov 2019
B = START + 4                    # windows before the crop; 4 survive it
N, H, W = 3, 2, 2                # land nodes, grid


class _Captured(Exception):
    pass


def _fcast_as_read(monkeypatch, tmp_path, var_names):
    """Run evaluate() up to the NaN masking and return the Li et al. tensor it built.
    Every forecast field is filled with the chronological rank of its initialisation."""
    lsm = np.zeros((T_OUT, H, W), dtype=bool)
    lsm.reshape(T_OUT, -1)[:, :N] = True

    ranks = {name: i for i, name in enumerate(sorted(var_names))}
    fcast_vars = {}
    for name in var_names:  # insertion order is the order given
        field = np.full((T_OUT, H, W), np.nan, dtype="float32")
        field.reshape(T_OUT, -1)[:, :N] = ranks[name]
        fcast_vars[name] = (("lead", "lat", "lon"), field)
    ds_fcast = xr.Dataset(fcast_vars)

    x = torch.zeros(B * N, T_IN, 2)
    x[..., 0] = 1.0  # cos=1, sin=0 -> January; the months play no part here
    batch = (torch.zeros(B * N, T_OUT), torch.zeros(B * N, T_OUT), {"grid": SimpleNamespace(x=x)})

    # window w first predicts 2018-06 + w, so window START is the first in the comparison window
    times = np.datetime64("2018-06", "M") - T_IN + np.arange(T_IN + B)
    data_pred = SimpleNamespace(
        time_period=None, samples=np.arange(T_IN, T_IN + B), num_land_nodes=lambda: N,
        ds_timeslice=SimpleNamespace(time=SimpleNamespace(values=times.astype("datetime64[ns]"))))
    datamodule = SimpleNamespace(setup=lambda stage: None, data_pred=data_pred, land_ocean_mask=lsm,
                                 get_antarctica_landmask=lambda: np.zeros(N, dtype=bool))
    config = {"experiment": {"save_dir": str(tmp_path)}, "pl": {},
              "datamodule": {"input_seq_len": T_IN, "output_seq_len": T_OUT, "delay": 0}}

    captured = {}

    def stop_at_valid_mask(pred, fcast, true):
        captured["fcast"] = fcast
        raise _Captured

    monkeypatch.setattr(compare_models, "read_config", lambda path: config)
    monkeypatch.setattr(compare_models, "generate_trainer", lambda config: SimpleNamespace())
    monkeypatch.setattr(compare_models, "TWSA_HeteroGraphDatamodule", lambda **kw: datamodule)
    monkeypatch.setattr(compare_models, "get_climatology", lambda seq, config: np.zeros((12, N)))
    monkeypatch.setattr(compare_models, "get_predictions", lambda *a, **kw: [batch])
    monkeypatch.setattr(compare_models.xr, "open_dataset", lambda path: ds_fcast)
    monkeypatch.setattr(compare_models, "valid_mask", stop_at_valid_mask)

    with pytest.raises(_Captured):
        compare_models.evaluate(str(tmp_path), gpu=0, FCast_file_path="unused.nc")
    return captured["fcast"]


def test_fcast_initialisations_are_read_chronologically(monkeypatch, tmp_path):
    """prepareFCast names the variables YYYY-MM; if they reach the file out of order,
    the forecast windows must still line up with ours instead of following insertion order."""
    n_kept = B - START
    names = [str(np.datetime64("2019-10", "M") + i) for i in range(n_kept + 6)]
    fcast = _fcast_as_read(monkeypatch, tmp_path, names[::-1])

    assert fcast.shape == (n_kept, N, T_OUT)
    np.testing.assert_array_equal(fcast[:, 0, 0].numpy(), np.arange(n_kept))

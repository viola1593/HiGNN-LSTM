"""Batch C3a: evaluation_ConvLSTM.py -- no-checkpoint path (#3), invalid
mean_timestep_scores.json (#4), checkpoint choice (#7), redundant trainer.test (#20)."""

import json
import os
import pathlib
import sys
import warnings
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest
import pytorch_lightning as pl
import torch

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import evaluation_ConvLSTM
from models.pl_convlstm import ConvLSTMModule
from utils.eval_metrics import recover_month_index

T_IN, T_OUT, F, H, W, B = 12, 6, 2, 4, 5, 3
LAT = np.array([-70.0, 0.0, 30.0, 60.0])  # grid rows; row 0 is the "Antarctica" row
TEST_PERIOD = ("2016-01", "2020-06")


class TinyModel(pl.LightningModule):
    """Uses ConvLSTMModule's own predict_step and unnormalizer, without its constructor."""

    predict_step = ConvLSTMModule.predict_step
    _twsa_unnormalizer = ConvLSTMModule._twsa_unnormalizer

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        return x[:, :T_OUT, 0]


def _touch(path, mtime):
    path.touch()
    os.utime(path, (mtime, mtime))


def _harness(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    true = torch.tensor(rng.standard_normal((B, T_OUT, H, W)), dtype=torch.float32)
    pred = true + 2.0
    angle = torch.tensor(2 * np.pi * rng.integers(0, 12, (B, T_IN, 1, H, W)) / 12, dtype=torch.float32)
    x = torch.cat([torch.cos(angle), torch.sin(angle)], dim=2)  # [..., -2] cos, [..., -1] sin
    batch = (pred, true, x)

    calls = SimpleNamespace(loaded=[], tested=0, grace=0, grace_kwargs=[], periods=[])

    def test(model, datamodule):
        calls.tested += 1
        return [{"test_rmse": 2.0}]

    def predict_dataloader(period):
        calls.periods.append(period)

    def GRACE_dataloader(**kwargs):
        calls.grace += 1
        calls.grace_kwargs.append(kwargs)

    trainer = SimpleNamespace(predict=lambda model, dataloader: [batch], test=test)

    class FakeConvLSTMModule:
        @staticmethod
        def load_from_checkpoint(path, **kwargs):
            calls.loaded.append(os.path.basename(path))
            return TinyModel()

    lsm = np.ones((H, W), dtype=bool)
    lsm[0] = False  # "Antarctica" row
    climatology = rng.standard_normal((12, H, W)).astype("float32")
    datamodule = SimpleNamespace(
        setup=lambda stage: None, target_seq_len=T_OUT, train_val_test_periods={"test": TEST_PERIOD},
        predict_dataloader=predict_dataloader, GRACE_dataloader=GRACE_dataloader,
        get_landmask_without_antarctica=lambda: lsm,
        data_train=SimpleNamespace(ds_timeslice={"lat": SimpleNamespace(values=LAT)}),
        twsa_climatology=SimpleNamespace(values=climatology))
    grace_climatology = SimpleNamespace(values=rng.standard_normal((12, H, W)).astype("float32"))
    config = {"experiment": {"save_dir": str(tmp_path)}, "pl": {}, "model": {},
              "datamodule": {"output_seq_len": T_OUT}}

    monkeypatch.setattr(evaluation_ConvLSTM, "load_grace_climatology", lambda: grace_climatology, raising=False)
    monkeypatch.setattr(evaluation_ConvLSTM, "read_config", lambda path: config)
    monkeypatch.setattr(evaluation_ConvLSTM, "generate_trainer", lambda config: trainer)
    monkeypatch.setattr(evaluation_ConvLSTM, "TWSA_Datamodule2D", lambda **kw: datamodule)
    monkeypatch.setattr(evaluation_ConvLSTM, "ConvLSTMModule", FakeConvLSTMModule)
    return calls


def _run(monkeypatch, tmp_path):
    calls = _harness(monkeypatch, tmp_path)
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0)
    return calls


# ---------------------------------------------------------------- #3: no checkpoint


def test_no_checkpoint_fails_loudly(monkeypatch, tmp_path):
    with pytest.raises(AssertionError):
        _run(monkeypatch, tmp_path)


# ---------------------------------------------------------------- #7: checkpoint choice


def test_best_ckpt_is_chosen_on_a_resumed_run(monkeypatch, tmp_path):
    """After a resume the best-val ckpt can be newer than last.ckpt; mtime must not decide."""
    _touch(tmp_path / "last.ckpt", 1_000)
    _touch(tmp_path / "epoch=41-step=420.ckpt", 2_000)
    assert _run(monkeypatch, tmp_path).loaded == ["epoch=41-step=420.ckpt"]


def test_ambiguous_ckpts_are_rejected(monkeypatch, tmp_path):
    _touch(tmp_path / "last.ckpt", 3_000)
    _touch(tmp_path / "epoch=1-step=10.ckpt", 1_000)
    _touch(tmp_path / "epoch=9-step=90.ckpt", 2_000)
    with pytest.raises(AssertionError):
        _run(monkeypatch, tmp_path)


# ---------------------------------------------------------------- #4: valid JSON


def test_timestep_scores_are_valid_json(monkeypatch, tmp_path):
    """pred = true + 2 on every cell: RMSE is exactly 2 at every lead, with zero spread."""
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    _run(monkeypatch, tmp_path)
    out = tmp_path / "eval_on_test"
    with open(out / "mean_timestep_scores.json") as f:
        mean = json.load(f)
    with open(out / "std_timestep_scores.json") as f:
        std = json.load(f)
    np.testing.assert_allclose(mean["rmse"], [2.0] * T_OUT, atol=1e-5)
    np.testing.assert_allclose(std["rmse"], [0.0] * T_OUT, atol=1e-5)
    np.testing.assert_allclose(mean["exp_var"], [1.0] * T_OUT, atol=1e-5)


# ---------------------------------------------------------------- cos(lat) weighting


def _mean_scores(tmp_path):
    with open(tmp_path / "eval_on_test" / "mean_timestep_scores.json") as f:
        return json.load(f)


def test_lat_weighted_rmse_is_written(monkeypatch, tmp_path):
    """pred = true + 2 on every cell: the weighted RMSE is 2 as well."""
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    _run(monkeypatch, tmp_path)
    mean = _mean_scores(tmp_path)
    np.testing.assert_allclose(mean["rmse_latw"], [2.0] * T_OUT, atol=1e-5)
    assert len(mean["rmse_latw_deseas"]) == len(mean["rmse_latw_clim"]) == T_OUT


def test_lat_weighted_rmse_weights_the_rows_by_cos_lat(monkeypatch, tmp_path):
    """Error 1 on the 60° row only: cos(lat) halves that row's weight."""
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    _harness(monkeypatch, tmp_path)
    trainer = evaluation_ConvLSTM.generate_trainer(None)
    _, true, x = trainer.predict(None, None)[0]
    error = torch.tensor(LAT == 60.0, dtype=torch.float32)[:, None].expand(H, W)
    trainer.predict = lambda model, dataloader: [(true + error, true, x)]
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0)

    w, e = np.cos(np.deg2rad(LAT[1:])), (LAT[1:] == 60.0).astype(float)  # row 0 is masked out
    mean = _mean_scores(tmp_path)
    np.testing.assert_allclose(mean["rmse_latw"], [np.sqrt((w * e).sum() / w.sum())] * T_OUT, atol=1e-5)
    np.testing.assert_allclose(mean["rmse"], [np.sqrt(e.mean())] * T_OUT, atol=1e-5)


# ---------------------------------------------------------------- per_timestep_scores.csv


def test_per_timestep_scores_csv_is_written(monkeypatch, tmp_path):
    """pred = true + 2 on every cell: RMSE 2, r = 1, and KGE″ = 1 - 2/σ_t per pixel."""
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    _harness(monkeypatch, tmp_path)
    _, true, _ = evaluation_ConvLSTM.generate_trainer(None).predict(None, None)[0]
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0)

    df = pd.read_csv(tmp_path / "eval_on_test" / "per_timestep_scores.csv", index_col=0)
    assert df.index.name == "Timestep"
    assert list(df.index) == list(range(T_OUT))
    assert list(df.columns[:4]) == ["rmse", "rmse_latw", "clim_rmse", "clim_rmse_latw"]
    assert sorted(df.columns[4:]) == sorted([
        "nse_latw", "nse_deseas_latw", "kge2_latw", "kge2_deseas_latw", "corr_latw", "corr_deseas_latw",
        "clim_nse_latw", "clim_kge2_latw", "clim_corr_latw"])

    np.testing.assert_allclose(df["rmse"], [2.0] * T_OUT, atol=1e-5)
    np.testing.assert_allclose(df["rmse_latw"], [2.0] * T_OUT, atol=1e-5)
    mean = _mean_scores(tmp_path)
    np.testing.assert_allclose(df["clim_rmse"], mean["rmse_clim"])
    np.testing.assert_allclose(df["clim_rmse_latw"], mean["rmse_latw_clim"])
    np.testing.assert_allclose(df["corr_latw"], [1.0] * T_OUT, atol=1e-5)

    sigma = true.double().numpy()[:, :, 1:].std(axis=0)  # [T_OUT, H-1, W] over the windows; row 0 is masked out
    w = np.broadcast_to(np.cos(np.deg2rad(LAT[1:]))[:, None], sigma.shape[1:])
    expected = ((1 - 2.0 / sigma) * w).sum(axis=(1, 2)) / w.sum()
    np.testing.assert_allclose(df["kge2_latw"], expected, rtol=1e-5)


# ---------------------------------------------------------------- climatology source


def test_climatology_is_the_datamodules_not_a_stale_cache(monkeypatch, tmp_path):
    """Truth is the datamodule's C of its target months, so the clim RMSE is 0 --
    unless a stale climatology.pt is read instead."""
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    _harness(monkeypatch, tmp_path)
    clim = evaluation_ConvLSTM.TWSA_Datamodule2D().twsa_climatology.values
    (tmp_path / "eval_on_test").mkdir()
    torch.save(clim + 100.0, tmp_path / "eval_on_test" / "climatology.pt")

    trainer = evaluation_ConvLSTM.generate_trainer(None)
    pred, _, x = trainer.predict(None, None)[0]
    last = recover_month_index(x[:, -1, -1, 0, 0], x[:, -1, -2, 0, 0])  # last input month, decoded as the script does
    true = torch.from_numpy(clim)[(last[:, None] + torch.arange(1, T_OUT + 1)) % 12]  # [B, T_OUT, H, W]
    trainer.predict = lambda model, dataloader: [(pred, true, x)]
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0)

    np.testing.assert_allclose(_mean_scores(tmp_path)["rmse_clim"], [0.0] * T_OUT, atol=1e-5)


# ---------------------------------------------------------------- #20: redundant test epoch


def test_evaluate_runs_no_test_epoch(monkeypatch, tmp_path):
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    assert _run(monkeypatch, tmp_path).tested == 0


# ---------------------------------------------------------------- --dataset grace


def test_grace_uses_the_grace_dataloader(monkeypatch, tmp_path):
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    calls = _harness(monkeypatch, tmp_path)
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0, dataset="grace")
    assert calls.grace == 1
    assert calls.grace_kwargs == [{}]
    assert calls.periods == []
    out = tmp_path / "eval_on_grace"
    for name in ["mean_timestep_scores.json", "std_timestep_scores.json", "per_timestep_scores.csv"]:
        assert (out / name).is_file()


# ---------------------------------------------------------------- --dataset grace_graceclim


def test_grace_graceclim_passes_c_grace_to_the_grace_dataloader(monkeypatch, tmp_path):
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    calls = _harness(monkeypatch, tmp_path)
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0, dataset="grace_graceclim")
    assert calls.grace == 1
    assert calls.grace_kwargs == [{"twsa_climatology": evaluation_ConvLSTM.load_grace_climatology()}]
    assert calls.periods == []
    out = tmp_path / "eval_on_grace_graceclim"
    for name in ["mean_timestep_scores.json", "std_timestep_scores.json", "per_timestep_scores.csv"]:
        assert (out / name).is_file()


def _scores(out):
    with open(out / "mean_timestep_scores.json") as f:
        mean = json.load(f)
    return pd.DataFrame(mean), pd.read_csv(out / "per_timestep_scores.csv", index_col=0)


def test_grace_graceclim_scores_as_grace_with_c_grace_in_place_of_c_rec(monkeypatch, tmp_path):
    """Every climatology-dependent score (clim baseline, deseas) uses C_grace; plain grace keeps C_rec."""
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    _harness(monkeypatch, tmp_path)
    datamodule = evaluation_ConvLSTM.TWSA_Datamodule2D()
    # the harness's pred = true + 2 makes r of the anomalies 1 for any C; a noisy pred does not
    trainer = evaluation_ConvLSTM.generate_trainer(None)
    _, true, x = trainer.predict(None, None)[0]
    pred = true + torch.randn(true.shape, generator=torch.Generator().manual_seed(0))
    trainer.predict = lambda model, dataloader: [(pred, true, x)]

    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0, dataset="grace_graceclim")
    graceclim = _scores(tmp_path / "eval_on_grace_graceclim")
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0, dataset="grace")
    grace = _scores(tmp_path / "eval_on_grace")
    datamodule.twsa_climatology = evaluation_ConvLSTM.load_grace_climatology()
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0, dataset="grace")
    swapped = _scores(tmp_path / "eval_on_grace")

    for got, expected in zip(graceclim, swapped):
        pd.testing.assert_frame_equal(got, expected)
    mean, df = graceclim
    for column in ["rmse_clim", "rmse_latw_clim"]:  # rmse_deseas cancels C: (pred - C) - (true - C)
        assert not np.allclose(mean[column], grace[0][column])
    for column in ["clim_rmse", "nse_deseas_latw", "corr_deseas_latw", "clim_corr_latw"]:
        assert not np.allclose(df[column], grace[1][column])
    np.testing.assert_allclose(df["rmse"], grace[1]["rmse"])  # the model's own scores do not depend on C


def test_test_still_uses_the_test_period(monkeypatch, tmp_path):
    _touch(tmp_path / "epoch=89-step=1890.ckpt", 1_000)
    calls = _harness(monkeypatch, tmp_path)
    evaluation_ConvLSTM.evaluate(str(tmp_path), gpu=0, dataset="test")
    assert calls.periods == [TEST_PERIOD]
    assert calls.grace == 0


class UnnormalizingDatamodule(pl.LightningDataModule):
    def unnormalize_twsa(self, data):
        return data * 10 + 1


def _predict(attach_datamodule, tmp_path):
    x = torch.randn(4, T_IN, F, H, W)
    dataloader = torch.utils.data.DataLoader(
        [{"x_sequence": xi, "y_timestep": xi[:T_OUT, 0]} for xi in x], batch_size=2)
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
                         enable_checkpointing=False, enable_model_summary=False,
                         default_root_dir=str(tmp_path))
    if attach_datamodule:
        trainer.datamodule = UnnormalizingDatamodule()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = trainer.predict(TinyModel(), dataloader)
    no_dm = [w for w in caught if "No DataModule found." in str(w.message)]
    return x, out, no_dm


def test_predict_without_test_epoch_still_unnormalizes(tmp_path):
    """What makes #20 safe: trainer.predict(model, dataloader) keeps the datamodule the
    script assigns by hand, so unnormalization does not depend on a prior trainer.test."""
    x, out, no_dm = _predict(attach_datamodule=True, tmp_path=tmp_path)
    assert no_dm == []
    pred = torch.cat([o[0] for o in out])
    torch.testing.assert_close(pred, x[:, :T_OUT, 0] * 10 + 1)


def test_missing_datamodule_is_detected(tmp_path):
    """Control for the test above: without the assignment the warning does fire."""
    _, _, no_dm = _predict(attach_datamodule=False, tmp_path=tmp_path)
    assert no_dm

"""Batch B2: train_ConvLSTM2D.py resume path and --seed argument (bug #23)."""

import pathlib
import sys

import matplotlib
import pytest
import yaml

matplotlib.use("Agg")

import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import train_ConvLSTM2D


class _StopAfterTrain(Exception):
    """Sentinel raised from save_config, i.e. right after the fit/resume branch."""


class _StubDatamodule:
    input_seq_len = 12
    target_seq_len = 6

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def setup(self, stage=None):
        self.stage = stage


class _StubModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def load_from_checkpoint(cls, path, **kwargs):
        model = cls(**kwargs)
        model.loaded_from = path
        return model


class _RecordingLogger(Logger):
    """Stands in for WandbLogger so no wandb run is needed."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.watched = []

    @property
    def name(self):
        return "stub"

    @property
    def version(self):
        return "0"

    def log_metrics(self, metrics, step=None):
        pass

    def log_hyperparams(self, params, *args, **kwargs):
        pass

    def watch(self, model, **kwargs):
        self.watched.append(model)


def _write_config(tmp_path, checkpoint):
    config = {
        "experiment": {"experiment_name": "b2", "exp_dir": str(tmp_path / "exp")},
        "wandb": {"project": "b2", "mode": "offline"},
        "model": {"input_dim": 12, "hidden_dim": 8, "num_layers": 1},
        "datamodule": {"batch_size": 1, "output_seq_len": 6},
        "pl": {"accelerator": "cpu", "devices": 1, "max_epochs": 1,
               "log_every_n_steps": 1},
        "load_checkpoint": str(checkpoint) if checkpoint else None,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the heavy pieces of train() with CPU-only stubs."""
    monkeypatch.setattr(train_ConvLSTM2D, "TWSA_Datamodule2D", _StubDatamodule)
    monkeypatch.setattr(train_ConvLSTM2D, "ConvLSTMModule", _StubModel)
    monkeypatch.setattr(pl.loggers, "WandbLogger", _RecordingLogger)

    calls = {"seeds": []}

    def fake_fit(self, model, datamodule=None, ckpt_path=None):
        calls["model"] = model
        calls["ckpt_path"] = ckpt_path

    def stop(*args, **kwargs):
        raise _StopAfterTrain

    real_seed_everything = pl.seed_everything

    def fake_seed_everything(seed=None, workers=False):
        calls["seeds"].append(seed)
        return real_seed_everything(seed, workers=workers)

    monkeypatch.setattr(pl.Trainer, "fit", fake_fit)
    monkeypatch.setattr(train_ConvLSTM2D, "save_config", stop)
    monkeypatch.setattr(train_ConvLSTM2D.pl, "seed_everything", fake_seed_everything)
    return calls


def test_resume_from_checkpoint_still_trains(tmp_path, stubbed):
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()
    config_path = _write_config(tmp_path, checkpoint)

    with pytest.raises(_StopAfterTrain):
        train_ConvLSTM2D.train(str(config_path))

    assert "model" in stubbed, "trainer.fit was never called on the resume path"
    assert stubbed["ckpt_path"] == str(checkpoint)
    assert stubbed["model"].loaded_from == str(checkpoint)


def test_fresh_run_trains_without_ckpt_path(tmp_path, stubbed):
    config_path = _write_config(tmp_path, checkpoint=None)

    with pytest.raises(_StopAfterTrain):
        train_ConvLSTM2D.train(str(config_path))

    assert stubbed["ckpt_path"] is None
    assert not hasattr(stubbed["model"], "loaded_from")


def test_seed_is_applied_and_recorded(tmp_path, stubbed):
    config_path = _write_config(tmp_path, checkpoint=None)

    with pytest.raises(_StopAfterTrain):
        train_ConvLSTM2D.train(str(config_path), seed=7)

    assert stubbed["seeds"] == [7]


def test_start_forwards_seed_from_cli(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, checkpoint=None)
    recorded = {}

    monkeypatch.setattr(train_ConvLSTM2D, "train",
                        lambda *args, **kwargs: recorded.update(args=args, kwargs=kwargs))
    monkeypatch.setattr(sys, "argv",
                        ["train_ConvLSTM2D.py", "--config_path", str(config_path),
                         "--seed", "123"])

    train_ConvLSTM2D.start()

    assert 123 in list(recorded["args"]) + list(recorded["kwargs"].values())


def test_start_defaults_to_seed_42(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, checkpoint=None)
    recorded = {}

    monkeypatch.setattr(train_ConvLSTM2D, "train",
                        lambda *args, **kwargs: recorded.update(args=args, kwargs=kwargs))
    monkeypatch.setattr(sys, "argv",
                        ["train_ConvLSTM2D.py", "--config_path", str(config_path)])

    train_ConvLSTM2D.start()

    assert 42 in list(recorded["args"]) + list(recorded["kwargs"].values())

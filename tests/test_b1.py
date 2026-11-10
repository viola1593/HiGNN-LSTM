"""Batch B1: train_HiGNN.py resume path, and dead code around it."""

import ast
import inspect
import pathlib
import sys

import matplotlib
import pytest
import yaml

matplotlib.use("Agg")

import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import train_HiGNN

SCRIPT = pathlib.Path(train_HiGNN.__file__)


class _StopBeforeFit(Exception):
    """Sentinel so a test can stop train() at trainer.fit without training."""


class _StubDatamodule:
    num_features = 3
    num_ocean_features = 2
    use_ocean_nodes = False
    input_seq_len = 4
    target_seq_len = 2

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

    def model_summary(self):
        pass


class _RecordingLogger(Logger):
    """Stands in for WandbLogger so the debug=False path needs no wandb run."""

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


def _write_config(tmp_path, *, debug, checkpoint):
    config = {
        "experiment": {"experiment_name": "b1", "exp_dir": str(tmp_path / "exp")},
        "wandb": {"project": "b1", "mode": "offline"},
        "model": {"hidden_dim": 8},
        "datamodule": {"batch_size": 1, "output_seq_len": 2, "graph_variant": "full"},
        "pl": {"accelerator": "cpu", "devices": 1, "max_epochs": 1},
        "load_checkpoint": str(checkpoint),
        "debug": debug,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return path


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the heavy pieces of train() with CPU-only stubs."""
    monkeypatch.setattr(train_HiGNN, "TWSA_HeteroGraphDatamodule", _StubDatamodule)
    monkeypatch.setattr(train_HiGNN, "HMModule", _StubModel)
    calls = {}

    def fake_fit(self, model, datamodule=None, ckpt_path=None):
        calls["model"] = model
        calls["ckpt_path"] = ckpt_path

    monkeypatch.setattr(pl.Trainer, "fit", fake_fit)
    return calls


def test_resume_in_debug_mode_does_not_index_empty_loggers(tmp_path, stubbed):
    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()
    config_path = _write_config(tmp_path, debug=True, checkpoint=checkpoint)

    train_HiGNN.train(str(config_path))

    assert stubbed["ckpt_path"] == str(checkpoint)
    assert isinstance(stubbed["model"], _StubModel)


def test_resume_outside_debug_mode_still_watches(tmp_path, stubbed, monkeypatch):
    monkeypatch.setattr(pl.loggers, "WandbLogger", _RecordingLogger)

    def fail_fit(self, *args, **kwargs):
        self.loggers[0].fitted = True
        raise _StopBeforeFit

    monkeypatch.setattr(pl.Trainer, "fit", fail_fit)

    checkpoint = tmp_path / "last.ckpt"
    checkpoint.touch()
    config_path = _write_config(tmp_path, debug=False, checkpoint=checkpoint)

    logged = []
    real_init = _RecordingLogger.__init__

    def capture_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        logged.append(self)

    monkeypatch.setattr(_RecordingLogger, "__init__", capture_init)

    with pytest.raises(_StopBeforeFit):
        train_HiGNN.train(str(config_path))

    assert len(logged) == 1
    assert len(logged[0].watched) == 1


def _module_tree():
    return ast.parse(SCRIPT.read_text())


def test_num_ocean_clusters_is_not_assigned():
    names = {
        node.id
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert "num_ocean_clusters" not in names


def test_plot_prediction_has_no_overfitting_parameter():
    assert "overfitting" not in inspect.signature(train_HiGNN.plot_prediction).parameters

    keywords = {
        kw.arg
        for node in ast.walk(_module_tree())
        if isinstance(node, ast.Call)
        for kw in node.keywords
    }
    assert "overfitting" not in keywords

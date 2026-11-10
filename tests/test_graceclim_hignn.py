"""evaluation_HiGNN.py --dataset grace_graceclim: C_grace replaces C_rec in the GRACE
dataloader and in every climatology-dependent score; grace, val and test are unchanged."""

import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import evaluation_HiGNN  # noqa: E402
from utils import eval_metrics  # noqa: E402

NUM_NODES, OUT_LEN = 5, 6


def _clim(seed, lat=3, lon=4):
    values = np.random.default_rng(seed).standard_normal((12, lat, lon))
    return xr.DataArray(values, dims=("month", "lat", "lon"), coords={"month": np.arange(1, 13)})


# ---------------------------------------------------------------- get_predictions


def _get_predictions(tmp_path, monkeypatch, dataset, **kwargs):
    monkeypatch.setattr(evaluation_HiGNN, "build_model_from_checkpoint", lambda *a, **k: "model")
    calls = []
    data_pred = SimpleNamespace(samples=[0], num_land_nodes=lambda: NUM_NODES)
    dm = SimpleNamespace(data_pred=data_pred, GRACE_dataloader=lambda **kw: calls.append(kw))
    y = torch.zeros(NUM_NODES, OUT_LEN)
    trainer = SimpleNamespace(predict=lambda model, dataloader: [(y, y, {})])
    evaluation_HiGNN.get_predictions(dm, trainer, {"experiment": {"save_dir": str(tmp_path)}},
                                     str(tmp_path), dataset=dataset, **kwargs)
    return calls


def test_get_predictions_passes_c_grace_to_the_grace_dataloader(tmp_path, monkeypatch):
    c_grace = _clim(1)
    calls = _get_predictions(tmp_path, monkeypatch, "grace_graceclim", grace_climatology=c_grace)
    assert len(calls) == 1 and calls[0].keys() == {"twsa_climatology"}
    assert calls[0]["twsa_climatology"] is c_grace


def test_get_predictions_on_grace_passes_no_climatology(tmp_path, monkeypatch):
    assert _get_predictions(tmp_path, monkeypatch, "grace") == [{}]


# ---------------------------------------------------------------- get_climatology


def test_get_climatology_returns_the_land_nodes_of_a_given_climatology():
    c_rec, c_grace = _clim(0), _clim(1)
    land = np.random.default_rng(2).random((3, 4)) > 0.5
    lsm = xr.DataArray(np.stack([land, land]), dims=("time", "lat", "lon"))
    dm = SimpleNamespace(twsa_climatology=c_rec, land_ocean_mask=lsm)
    config = {"experiment": {"save_dir": "unused"}}

    out = eval_metrics.get_climatology(dm, config, climatology=c_grace)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, c_grace.values[:, land].astype(np.float32))
    np.testing.assert_array_equal(eval_metrics.get_climatology(dm, config),
                                  c_rec.values[:, land].astype(np.float32))


# ---------------------------------------------------------------- evaluate wiring


class _Stop(Exception):
    pass


def _evaluate(tmp_path, monkeypatch, dataset):
    """Runs evaluate up to get_predictions and records the climatology it hands on."""
    c_grace = _clim(1)
    seen = {}
    config = {"experiment": {"save_dir": str(tmp_path)}, "pl": {}, "datamodule": {"output_seq_len": OUT_LEN}}
    dm = SimpleNamespace(setup=lambda stage: None)

    def get_climatology(datamodule, config, climatology=None):
        seen["climatology"] = climatology
        return np.zeros((12, NUM_NODES), dtype=np.float32)

    def get_predictions(datamodule, trainer, config, experiment_path, dataset="test", gpu=0,
                        grace_climatology=None, save=True):
        seen.update(dataset=dataset, grace_climatology=grace_climatology, save_dir=config["experiment"]["save_dir"])
        raise _Stop

    monkeypatch.setattr(evaluation_HiGNN, "read_config", lambda path: config)
    monkeypatch.setattr(evaluation_HiGNN, "generate_trainer", lambda config: SimpleNamespace())
    monkeypatch.setattr(evaluation_HiGNN, "TWSA_HeteroGraphDatamodule", lambda **kw: dm)
    monkeypatch.setattr(evaluation_HiGNN, "load_grace_climatology", lambda: c_grace, raising=False)
    monkeypatch.setattr(evaluation_HiGNN, "get_climatology", get_climatology)
    monkeypatch.setattr(evaluation_HiGNN, "get_predictions", get_predictions)
    with pytest.raises(_Stop):
        evaluation_HiGNN.evaluate(str(tmp_path), gpu=0, dataset=dataset)
    return seen, c_grace


def test_evaluate_grace_graceclim_uses_c_grace_for_dataloader_and_scores(tmp_path, monkeypatch):
    seen, c_grace = _evaluate(tmp_path, monkeypatch, "grace_graceclim")
    assert seen["climatology"] is c_grace
    assert seen["grace_climatology"] is c_grace
    assert seen["dataset"] == "grace_graceclim"
    assert seen["save_dir"] == str(tmp_path / "eval_on_grace_graceclim")


@pytest.mark.parametrize("dataset", ["grace", "val", "test"])
def test_evaluate_other_datasets_keep_the_datamodules_climatology(tmp_path, monkeypatch, dataset):
    seen, _ = _evaluate(tmp_path, monkeypatch, dataset)
    assert seen["climatology"] is None
    assert seen["grace_climatology"] is None
    assert seen["save_dir"] == str(tmp_path / f"eval_on_{dataset}")

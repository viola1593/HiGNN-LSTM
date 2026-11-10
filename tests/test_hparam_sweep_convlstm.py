"""hparam_sweep_convlstm.py: grid over learning rate x hidden_dim x num_layers for the anomaly ConvLSTM."""

import itertools
import json
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import hparam_sweep_convlstm


def _base():
    return {
        "experiment": {"experiment_name": "baseline2D_anom", "exp_dir": "experiments/baseline2D/anom"},
        "model": {"input_dim": 12, "hidden_dim": 32, "num_layers": 2, "learning_rate": 0.001,
                  "lr_scheduler": None},
        "datamodule": {"twsa_anomaly": True},
        "pl": {"max_epochs": 100, "devices": [0]},
    }


LRS, HIDDEN, LAYERS = [0.0005, 0.001, 0.002], [32, 64], [1, 2]


@pytest.fixture
def configs():
    base = _base()
    out = hparam_sweep_convlstm.build_configs(base, LRS, HIDDEN, LAYERS, epochs=60, gpu=1,
                                              exp_dir="experiments/baseline2D/anom/hparam_sweep")
    return base, out


def test_one_config_per_grid_point(configs):
    _, out = configs
    got = {(c["model"]["learning_rate"], c["model"]["hidden_dim"], c["model"]["num_layers"]) for c in out}
    assert len(out) == 12
    assert got == set(itertools.product(LRS, HIDDEN, LAYERS))


def test_fixed_overrides(configs):
    _, out = configs
    for c in out:
        assert c["pl"]["max_epochs"] == 60
        assert c["pl"]["devices"] == [1]
        assert c["model"]["lr_scheduler"] == "cosine"  # same schedule as the HiGNN anomaly config
        assert c["experiment"]["exp_dir"] == "experiments/baseline2D/anom/hparam_sweep"


def test_only_convlstm_model_keys(configs):
    _, out = configs
    assert all(set(c["model"]) == set(_base()["model"]) for c in out)  # no GNN keys -> no TypeError


def test_names_unique_and_encode_grid_point(configs):
    _, out = configs
    names = [c["experiment"]["experiment_name"] for c in out]
    assert len(set(names)) == len(names)
    c = next(c for c in out if c["model"]["learning_rate"] == 0.002
             and c["model"]["hidden_dim"] == 32 and c["model"]["num_layers"] == 1)
    assert c["experiment"]["experiment_name"] == "baseline2D_anom_h32_d1_lr0.002_ep60"


def test_base_not_mutated(configs):
    base, _ = configs
    assert base == _base()


def test_collect_results(tmp_path):
    # last-epoch val_rmse (json) ranks the runs the other way round than the best checkpoint
    for name, last_rmse, best_rmse, ckpt in [("a_h32_d1_lr0.001_ep60", 5.9, 5.1, "epoch=20-step=441.ckpt"),
                                             ("b_h64_d2_lr0.001_ep60", 5.2, 5.3, "epoch=30-step=651.ckpt")]:
        run = tmp_path / f"{name}_10-16-2026_10:00:00"
        run.mkdir()
        torch.save({"callbacks": {"ModelCheckpoint{'monitor': 'val_rmse'}":
                                  {"best_model_score": torch.tensor(best_rmse)}}}, run / ckpt)
        (run / "eval_on_val.json").write_text(json.dumps({"validation_metrics": [{"val_rmse": last_rmse}]}))
    unfinished = tmp_path / "c_h32_d2_lr0.001_ep60_10-16-2026_11:00:00"  # still training -> skipped
    unfinished.mkdir()
    torch.save({"callbacks": {"ModelCheckpoint{}": {"best_model_score": torch.tensor(1.0)}}},
               unfinished / "epoch=1-step=21.ckpt")
    (tmp_path / "hparam_sweep_configs").mkdir()

    df = hparam_sweep_convlstm.collect_results(str(tmp_path))
    assert list(df["run"]) == ["a_h32_d1_lr0.001_ep60", "b_h64_d2_lr0.001_ep60"]  # sorted by best val_rmse
    assert list(df["best_epoch"]) == [20, 30]
    assert list(df["val_rmse"]) == pytest.approx([5.1, 5.3])

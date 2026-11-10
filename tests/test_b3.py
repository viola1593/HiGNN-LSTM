"""Batch B3: run_seed_study.py manifest key (#24) and eval dir for absolute run dirs (#25)."""

import os
import pathlib
import sys

import matplotlib
import pytest
import yaml

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd

import compare_models
import evaluation_ConvLSTM
import evaluation_HiGNN
import run_seed_study


def _write_config(tmp_path, variant):
    config = {
        "experiment": {"experiment_name": "b3", "exp_dir": str(tmp_path / "exp")},
        "datamodule": {"graph_variant": variant},
    }
    path = tmp_path / f"config_{variant}.yaml"
    path.write_text(yaml.dump(config))
    return path


def _run_main(monkeypatch, tmp_path, config_path, variant, trained):
    """Run main() with training/evaluation/aggregation stubbed out."""

    def fake_train_seed(config_path, seed, exp_dir, run_prefix):
        trained.append((variant, seed))
        run_dir = pathlib.Path(exp_dir) / f"{run_prefix}_seed{seed}_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best.ckpt").touch()
        return str(run_dir)

    tidy = pd.DataFrame([{"dataset": dataset, "variant": variant, "seed": 42, "lead": 1,
                          "metric": metric, "value": 1.0}
                         for dataset in run_seed_study.DATASETS
                         for metric in ("rmse", "clim_rmse")])

    monkeypatch.setattr(run_seed_study, "train_seed", fake_train_seed)
    monkeypatch.setattr(run_seed_study, "evaluate_run", lambda *a, **k: True)
    monkeypatch.setattr(run_seed_study, "collect_scores", lambda runs: tidy)
    monkeypatch.setattr(sys, "argv",
                        ["run_seed_study.py", "--config_path", str(config_path), "--seeds", "42"])
    run_seed_study.main()


def test_second_variant_does_not_reuse_the_first_variants_runs(tmp_path, monkeypatch):
    trained = []
    _run_main(monkeypatch, tmp_path, _write_config(tmp_path, "full"), "full", trained)
    _run_main(monkeypatch, tmp_path, _write_config(tmp_path, "no_ocean"), "no_ocean", trained)

    assert trained == [("full", 42), ("no_ocean", 42)]

    manifests = sorted(p.name for p in (tmp_path / "exp").glob("*_manifest.json"))
    assert manifests == ["b3_full_seed_study_manifest.json",
                         "b3_no_ocean_seed_study_manifest.json"]

    # the second variant's report must not have overwritten the first one's
    reports = sorted(p.name for p in (tmp_path / "exp").glob("*_seed_study.md"))
    assert reports == ["b3_full_seed_study.md", "b3_no_ocean_seed_study.md"]


def test_rerunning_the_same_variant_still_reuses_its_runs(tmp_path, monkeypatch):
    trained = []
    config_path = _write_config(tmp_path, "full")
    _run_main(monkeypatch, tmp_path, config_path, "full", trained)
    _run_main(monkeypatch, tmp_path, config_path, "full", trained)

    assert trained == [("full", 42)]


def test_eval_dir_nests_inside_the_run_dir():
    assert run_seed_study.eval_dir_for("experiments/HiGNN/run", "val") == \
        "experiments/HiGNN/run/eval_on_val"


def test_eval_dir_survives_an_absolute_run_dir(tmp_path):
    run_dir = str(tmp_path / "experiments" / "HiGNN" / "run")

    assert run_seed_study.eval_dir_for(run_dir, "test") == os.path.join(run_dir, "eval_on_test")


@pytest.mark.parametrize("absolute", [False, True])
def test_eval_dir_matches_evaluation_script(tmp_path, monkeypatch, absolute):
    """The harness must look where evaluation_HiGNN actually writes, absolute path included."""
    monkeypatch.chdir(tmp_path)
    run_dir = str(tmp_path / "exp" / "run") if absolute else os.path.join("exp", "run")
    config = {"experiment": {"save_dir": run_dir}}

    updated = evaluation_HiGNN.create_experiment_dir(config, dataset="test")

    assert updated["experiment"]["save_dir"] == run_seed_study.eval_dir_for(run_dir, "test")
    assert (tmp_path / "exp" / "run" / "eval_on_test").is_dir()


@pytest.mark.parametrize("absolute", [False, True])
def test_compare_models_nests_inside_the_run_dir(tmp_path, monkeypatch, absolute):
    monkeypatch.chdir(tmp_path)
    run_dir = str(tmp_path / "exp" / "run") if absolute else os.path.join("exp", "run")
    config = {"experiment": {"save_dir": run_dir}}

    updated = compare_models.create_experiment_dir(config, dataset="test")

    assert updated["experiment"]["save_dir"] == os.path.join(run_dir, "compare_to_fcast")
    assert (tmp_path / "exp" / "run" / "compare_to_fcast").is_dir()


@pytest.mark.parametrize("absolute", [False, True])
def test_convlstm_nests_inside_the_run_dir(tmp_path, monkeypatch, absolute):
    monkeypatch.chdir(tmp_path)
    run_dir = str(tmp_path / "exp" / "run") if absolute else os.path.join("exp", "run")

    updated = evaluation_ConvLSTM.create_experiment_dir({"experiment": {}}, run_dir)

    assert updated["experiment"]["save_dir"] == os.path.join(run_dir, "eval_on_test")
    assert (tmp_path / "exp" / "run" / "eval_on_test").is_dir()

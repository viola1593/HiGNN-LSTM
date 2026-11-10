"""seed_study_convlstm.py: the ConvLSTM seed study mirrors run_seed_study.py (HiGNN)."""

import json
import os
import pathlib
import subprocess
import sys
import time
import types

import matplotlib
import pytest
import yaml

matplotlib.use("Agg")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import evaluation_ConvLSTM
import seed_study_convlstm


def _write_config(tmp_path, name="c2d"):
    config = {"experiment": {"experiment_name": name, "exp_dir": str(tmp_path / "exp")}}
    path = tmp_path / f"config_{name}.yaml"
    path.write_text(yaml.dump(config))
    return path


def _write_scores(run_dir, dataset, rmse, clim):
    """Write a mean_timestep_scores.json as evaluation_ConvLSTM does."""
    eval_dir = pathlib.Path(seed_study_convlstm.eval_dir_for(str(run_dir), dataset))
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "mean_timestep_scores.json").write_text(
        json.dumps({"rmse_latw": rmse, "rmse_latw_deseas": rmse, "rmse_latw_clim": clim}))


# --------------------------------------------------------------------------------------------------
# evaluation_ConvLSTM.py can evaluate on val as well as test
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dataset", ["val", "test"])
@pytest.mark.parametrize("absolute", [False, True])
def test_eval_dir_matches_evaluation_script(tmp_path, monkeypatch, dataset, absolute):
    """The harness must look where evaluation_ConvLSTM actually writes."""
    monkeypatch.chdir(tmp_path)
    run_dir = str(tmp_path / "exp" / "run") if absolute else os.path.join("exp", "run")

    updated = evaluation_ConvLSTM.create_experiment_dir({"experiment": {}}, run_dir, dataset=dataset)

    assert updated["experiment"]["save_dir"] == seed_study_convlstm.eval_dir_for(run_dir, dataset)
    assert (tmp_path / "exp" / "run" / f"eval_on_{dataset}").is_dir()


def test_evaluation_script_takes_a_dataset_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluation_ConvLSTM, "evaluate",
                        lambda exp_path, gpu, dataset="test": calls.append((exp_path, gpu, dataset)))
    monkeypatch.setattr(sys, "argv", ["evaluation_ConvLSTM.py", "--exp_path", "run",
                                      "--gpu", "1", "--dataset", "val"])

    evaluation_ConvLSTM.start()

    assert calls == [("run", 1, "val")]


class _StopAtPrediction(Exception):
    pass


@pytest.mark.parametrize("dataset", ["val", "test"])
def test_evaluation_predicts_on_the_requested_period(tmp_path, monkeypatch, dataset):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.dump(
        {"experiment": {}, "pl": {}, "model": {}, "datamodule": {"output_seq_len": 6}}))
    (run_dir / "epoch=1-step=1.ckpt").touch()

    class FakeDatamodule:
        train_val_test_periods = {"train": "P_train", "val": "P_val", "test": "P_test"}
        target_seq_len = 6

        def __init__(self, **kwargs):
            pass

        def setup(self, stage=None):
            pass

        def predict_dataloader(self, time_step_idx=None, period=None):
            raise _StopAtPrediction(period)

    monkeypatch.setattr(evaluation_ConvLSTM, "generate_trainer", lambda config: types.SimpleNamespace())
    monkeypatch.setattr(evaluation_ConvLSTM, "TWSA_Datamodule2D", FakeDatamodule)
    monkeypatch.setattr(evaluation_ConvLSTM.ConvLSTMModule, "load_from_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr("pytorch_lightning.utilities.model_summary.ModelSummary", lambda model: "")

    with pytest.raises(_StopAtPrediction) as stopped:
        evaluation_ConvLSTM.evaluate(str(run_dir), gpu=0, dataset=dataset)

    assert stopped.value.args[0] == f"P_{dataset}"


# --------------------------------------------------------------------------------------------------
# seed_study_convlstm.py
# --------------------------------------------------------------------------------------------------

def test_train_seed_returns_the_run_dir_the_training_made(tmp_path, monkeypatch):
    exp_dir = tmp_path / "exp"
    old = exp_dir / "c2d_09-01-2026_10:00:00"
    old.mkdir(parents=True)
    os.utime(old, (time.time() - 3600,) * 2)
    new = exp_dir / "c2d_09-11-2026_10:00:00"
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        new.mkdir()
        # coarse fs timestamps can trail time.time(); a real training starts seconds after t0
        os.utime(new, (time.time() + 1,) * 2)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(seed_study_convlstm.subprocess, "run", fake_run)

    assert seed_study_convlstm.train_seed("cfg.yaml", 43, str(exp_dir), "c2d") == str(new)
    assert calls[0][1:] == ["scripts/train_ConvLSTM2D.py", "--config_path", "cfg.yaml", "--seed", "43"]


def test_train_seed_reports_a_failed_training(tmp_path, monkeypatch):
    monkeypatch.setattr(seed_study_convlstm.subprocess, "run",
                        lambda cmd: subprocess.CompletedProcess(cmd, 1))

    assert seed_study_convlstm.train_seed("cfg.yaml", 43, str(tmp_path), "c2d") is None


def test_evaluate_run_asks_for_the_dataset_and_clears_stale_predictions(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    stale = pathlib.Path(seed_study_convlstm.eval_dir_for(str(run_dir), "val")) / "predictions.pt"
    stale.parent.mkdir(parents=True)
    stale.touch()
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        assert not stale.exists()
        _write_scores(run_dir, "val", [1.0], [2.0])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(seed_study_convlstm.subprocess, "run", fake_run)

    assert seed_study_convlstm.evaluate_run(str(run_dir), "val", gpu=3)
    assert calls[0][1:] == ["scripts/evaluation_ConvLSTM.py", "--exp_path", str(run_dir),
                            "--dataset", "val", "--gpu", "3"]

    # the scores are there now, so a second call does not re-run the evaluation
    assert seed_study_convlstm.evaluate_run(str(run_dir), "val", gpu=3)
    assert len(calls) == 1


def test_collect_scores_reads_the_convlstm_scores(tmp_path):
    run_dir = tmp_path / "run42"
    _write_scores(run_dir, "val", [1.0, 2.0], [5.0, 6.0])
    _write_scores(run_dir, "test", [3.0, 4.0], [7.0, 8.0])

    tidy = seed_study_convlstm.collect_scores({"42": str(run_dir)})

    assert list(tidy.columns) == ["dataset", "variant", "seed", "lead", "metric", "value"]
    test_rmse = tidy[(tidy.dataset == "test") & (tidy.metric == "rmse_latw")]
    assert list(test_rmse.lead) == [1, 2]
    assert list(test_rmse.value) == [3.0, 4.0]
    val_clim = tidy[(tidy.dataset == "val") & (tidy.metric == "clim_rmse_latw")]
    assert list(val_clim.value) == [5.0, 6.0]
    assert set(tidy.seed) == {42}
    assert set(tidy.variant) == {"convlstm"}


def test_collect_scores_skips_missing_evaluations(tmp_path):
    run_dir = tmp_path / "run42"
    _write_scores(run_dir, "test", [3.0], [7.0])

    tidy = seed_study_convlstm.collect_scores({"42": str(run_dir)})

    assert set(tidy.dataset) == {"test"}


def _run_main(monkeypatch, config_path, trained, seeds=("42",), extra_args=(), evaluated=None):
    """Run main() with training and evaluation stubbed out."""

    def fake_train_seed(config_path, seed, exp_dir, run_prefix):
        trained.append(seed)
        run_dir = pathlib.Path(exp_dir) / f"{run_prefix}_run{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "epoch=1-step=1.ckpt").touch()
        return str(run_dir)

    def fake_evaluate_run(run_dir, dataset, gpu):
        if evaluated is not None:
            evaluated.append(dataset)
        _write_scores(run_dir, dataset, [1.0, 2.0], [3.0, 3.0])
        return True

    monkeypatch.setattr(seed_study_convlstm, "train_seed", fake_train_seed)
    monkeypatch.setattr(seed_study_convlstm, "evaluate_run", fake_evaluate_run)
    monkeypatch.setattr(sys, "argv", ["seed_study_convlstm.py", "--config_path", str(config_path),
                                      "--seeds", *seeds, *extra_args])
    seed_study_convlstm.main()


def test_main_trains_evaluates_and_reports(tmp_path, monkeypatch):
    trained = []
    _run_main(monkeypatch, _write_config(tmp_path), trained, seeds=("42", "43"))

    exp = tmp_path / "exp"
    assert trained == [42, 43]
    manifest = json.loads((exp / "c2d_seed_study_manifest.json").read_text())
    assert sorted(manifest) == ["42", "43"]
    assert (exp / "c2d_seed_study.csv").exists()
    report = (exp / "c2d_seed_study.md").read_text()
    assert "convlstm/seed42" in report and "convlstm/seed43" in report
    assert "VAL" in report and "TEST" in report
    assert "Noise floor (convlstm, test" in report


def test_rerunning_reuses_the_trained_seeds(tmp_path, monkeypatch):
    trained = []
    config_path = _write_config(tmp_path)
    _run_main(monkeypatch, config_path, trained)
    _run_main(monkeypatch, config_path, trained)

    assert trained == [42]


def test_default_does_not_evaluate_grace(tmp_path, monkeypatch):
    evaluated = []
    _run_main(monkeypatch, _write_config(tmp_path), [], evaluated=evaluated)

    assert evaluated == ["val", "test"]
    assert "GRACE" not in (tmp_path / "exp" / "c2d_seed_study.md").read_text()


def test_datasets_grace_is_evaluated_and_reported(tmp_path, monkeypatch):
    evaluated = []
    _run_main(monkeypatch, _write_config(tmp_path), [], evaluated=evaluated,
              extra_args=("--datasets", "val", "test", "grace"))

    assert evaluated == ["val", "test", "grace"]
    report = (tmp_path / "exp" / "c2d_seed_study.md").read_text()
    assert "### RMSE [cm] - grid cells - GRACE" in report
    assert report.index("- TEST") < report.index("- GRACE")

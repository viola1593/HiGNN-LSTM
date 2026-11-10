"""run_seed_study.py: optional GRACE evaluation via --datasets and a GRACE table in the report."""

import pathlib
import sys
from types import SimpleNamespace

import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import run_seed_study


def _make_trained_study(tmp_path):
    """Config plus a manifest pointing at one trained run, so main() skips training."""
    exp_dir = tmp_path / "exp"
    run_dir = exp_dir / "g_full_seed42_run"
    run_dir.mkdir(parents=True)
    (run_dir / "best.ckpt").touch()
    config = {"experiment": {"experiment_name": "g", "exp_dir": str(exp_dir)},
              "datamodule": {"graph_variant": "full"}}
    (run_dir / "config.yaml").write_text(yaml.dump(config))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    (exp_dir / "g_full_seed_study_manifest.json").write_text(f'{{"42": "{run_dir}"}}')
    return config_path, exp_dir, run_dir


def _fake_evaluate(evaluated, scores_only_seen=None):
    def evaluate_run(run_dir, dataset, gpu, scores_only=False):
        evaluated.append(dataset)
        if scores_only_seen is not None:
            scores_only_seen.append(scores_only)
        eval_dir = pathlib.Path(run_seed_study.eval_dir_for(run_dir, dataset))
        eval_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Timestep": range(6), "rmse_latw": [1.0] * 6,
                      "clim_rmse_latw": [2.0] * 6}).to_csv(eval_dir / "per_timestep_scores.csv")
        return True
    return evaluate_run


def _run_main(monkeypatch, config_path, extra_args):
    evaluated = []
    monkeypatch.setattr(run_seed_study, "evaluate_run", _fake_evaluate(evaluated))
    monkeypatch.setattr(sys, "argv", ["run_seed_study.py", "--config_path", str(config_path),
                                      "--seeds", "42"] + extra_args)
    run_seed_study.main()
    return evaluated


def test_default_does_not_evaluate_grace(tmp_path, monkeypatch):
    config_path, exp_dir, _ = _make_trained_study(tmp_path)

    evaluated = _run_main(monkeypatch, config_path, [])

    assert evaluated == ["val", "test"]
    assert "GRACE" not in (exp_dir / "g_full_seed_study.md").read_text()


def test_datasets_grace_is_evaluated_and_reported(tmp_path, monkeypatch):
    config_path, exp_dir, _ = _make_trained_study(tmp_path)

    evaluated = _run_main(monkeypatch, config_path, ["--datasets", "val", "test", "grace"])

    assert evaluated == ["val", "test", "grace"]
    report = (exp_dir / "g_full_seed_study.md").read_text()
    assert "### RMSE [cm] - grid cells - GRACE" in report
    # GRACE comes after VAL and TEST
    assert report.index("- TEST") < report.index("- GRACE")
    tidy = pd.read_csv(exp_dir / "g_full_seed_study.csv")
    assert set(tidy["dataset"]) == {"val", "test", "grace"}


def test_scores_only_reaches_every_evaluation(tmp_path, monkeypatch):
    config_path, _, _ = _make_trained_study(tmp_path)
    seen = []
    monkeypatch.setattr(run_seed_study, "evaluate_run", _fake_evaluate([], seen))
    monkeypatch.setattr(sys, "argv", ["run_seed_study.py", "--config_path", str(config_path), "--seeds", "42",
                                      "--datasets", "test", "--scores_only"])

    run_seed_study.main()

    assert seen == [True]


def _evaluation_command(tmp_path, monkeypatch, **kwargs):
    commands = []

    def run(command):
        commands.append(command)
        scores = pathlib.Path(run_seed_study.eval_dir_for(tmp_path, "test")) / "per_timestep_scores.csv"
        scores.parent.mkdir(parents=True, exist_ok=True)
        scores.touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_seed_study.subprocess, "run", run)
    assert run_seed_study.evaluate_run(str(tmp_path), "test", 1, **kwargs)
    return commands[0]


def test_evaluate_run_passes_scores_only_to_the_evaluation(tmp_path, monkeypatch):
    assert "--scores_only" in _evaluation_command(tmp_path, monkeypatch, scores_only=True)


def test_evaluate_run_makes_the_full_evaluation_by_default(tmp_path, monkeypatch):
    assert "--scores_only" not in _evaluation_command(tmp_path, monkeypatch)

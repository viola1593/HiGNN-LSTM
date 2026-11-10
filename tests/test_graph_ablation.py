"""run_graph_ablation.py: one-at-a-time graph variants of the base model and their summary table."""

import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import results_tables as rt
import run_graph_ablation as rga
from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule


def _base():
    return {"datamodule": {"num_ocean_clusters": 20, "k_mesh2mesh": 10, "k_mesh2grid": 3, "graph_variant": "full",
                           "graph_embedding_version": "unitsphere_v2", "ds_path": "data/x.nc"},
            "experiment": {"exp_dir": "base_dir", "experiment_name": "V0", "save_dir": "base_run", "seed": 42},
            "model": {"hidden_dim": 32, "residual_from_last": False},
            "pl": {"devices": [0], "max_epochs": 50}}


def test_variants_are_the_requested_one_at_a_time_grid():
    base = _base()["datamodule"]
    varied = {name: {k: v for k, v in overrides.items() if k != "graph_embedding_version"}
              for name, (_, overrides) in rga.VARIANTS.items()}

    assert varied == {"oc10": {"num_ocean_clusters": 10}, "oc50": {"num_ocean_clusters": 50},
                      "m2m5": {"k_mesh2mesh": 5}, "m2m20": {"k_mesh2mesh": 20},
                      "m2g1": {"k_mesh2grid": 1}, "m2g5": {"k_mesh2grid": 5}}
    assert all(base[k] != v for overrides in varied.values() for k, v in overrides.items())


@pytest.mark.parametrize("name", rga.VARIANTS)
def test_build_config_only_changes_the_variant(name):
    base = _base()
    label, overrides = rga.VARIANTS[name]

    config = rga.build_config(base, name, overrides, gpu=1)

    assert base == _base()
    assert config["datamodule"] == {**base["datamodule"], **overrides}
    assert config["model"] == base["model"]
    assert config["pl"] == {**base["pl"], "devices": [1]}
    assert config["experiment"]["exp_dir"] == f"{rga.STUDY_DIR}/{name}"
    assert config["experiment"]["experiment_name"] == f"V0_{name}"


def test_every_variant_passes_the_graph_cache_check(tmp_path, monkeypatch):
    """A cluster count sharing the base graph dir would stop the run at _ensure_graph_meta."""
    monkeypatch.chdir(tmp_path)
    configs = [_base()] + [rga.build_config(_base(), name, overrides, gpu=1)
                           for name, (_, overrides) in rga.VARIANTS.items()]
    for config in configs:
        d = config["datamodule"]
        dm = SimpleNamespace(graph_variant=d["graph_variant"], graph_embedding_version=d["graph_embedding_version"],
                             ds_path=d["ds_path"], num_ocean_clusters=d["num_ocean_clusters"],
                             graph_path=f"graphs/{d['graph_variant']}_{d['graph_embedding_version']}")
        TWSA_HeteroGraphDatamodule._ensure_graph_meta(dm)

    assert (tmp_path / "graphs/full_unitsphere_v2_oc50").is_dir()
    # the k variants share the base graph dir: its cache files are named by k
    assert not (tmp_path / "graphs/full_unitsphere_v2_m2m5").exists()


def test_main_runs_a_test_only_scores_only_seed_study_per_variant(tmp_path, monkeypatch):
    commands = []
    monkeypatch.setattr(rga, "STUDY_DIR", str(tmp_path / "study"))
    monkeypatch.setattr(rga, "base_config", _base)
    monkeypatch.setattr(rga, "write_tables", lambda n_seeds: commands.append(n_seeds))
    monkeypatch.setattr(rga.subprocess, "run", lambda command: commands.append(command))
    monkeypatch.setattr(sys, "argv", ["run_graph_ablation.py", "--gpu", "1"])

    rga.main()

    assert commands[-1] == 4  # the table comes last, for the four default seeds
    studies = commands[:-1]
    assert len(studies) == len(rga.VARIANTS)
    for name, command in zip(rga.VARIANTS, studies):
        path = tmp_path / "study" / name / "config.yaml"
        assert command[1:] == ["scripts/run_seed_study.py", "--config_path", str(path), "--seeds", "42", "43", "44", "45",
                               "--gpu", "1", "--datasets", "test", "--scores_only"]
        config = yaml.safe_load(path.read_text())
        assert config["experiment"]["exp_dir"] == str(tmp_path / "study" / name)
        assert config["pl"]["devices"] == [1]


def _study(study_dir, rmse, with_scores=True):
    """A study with one run per value of `rmse`: its RMSE at the six leads, or at every lead if a number."""
    runs = {}
    for seed, value in enumerate(rmse, start=42):
        run = pathlib.Path(study_dir) / f"run_seed{seed}"
        (run / "eval_on_test").mkdir(parents=True)
        runs[str(seed)] = str(run)
        if with_scores:
            pd.DataFrame({"Timestep": range(6), "rmse_latw": np.broadcast_to(value, 6)}).to_csv(
                run / "eval_on_test" / "per_timestep_scores.csv", index=False)
    (pathlib.Path(study_dir) / "x_seed_study_manifest.json").write_text(json.dumps(runs))


BASE_RMSE = np.array([2.41, 2.42, 2.415, 2.425, 2.41, 2.42, 2.418, 2.422])
OC10_RMSE = np.array([2.43, 2.44, 2.425, 2.445])


@pytest.fixture
def studies(tmp_path, monkeypatch):
    monkeypatch.setattr(rga, "BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setattr(rga, "STUDY_DIR", str(tmp_path / "study"))
    monkeypatch.setattr(rga, "OUT_DIR", str(tmp_path / "tables"))
    _study(tmp_path / "base", BASE_RMSE)
    return tmp_path


def test_write_tables_gives_the_welch_difference_to_the_base_model(studies):
    _study(studies / "study" / "oc10", OC10_RMSE)

    rga.write_tables(n_seeds=4)

    md = (studies / "tables" / "ablation_graph.md").read_text()
    row = next(line for line in md.splitlines() if line.startswith("| 10 ocean clusters"))
    res = stats.ttest_ind(OC10_RMSE, BASE_RMSE, equal_var=False)
    ci = res.confidence_interval(0.95)
    delta = OC10_RMSE.mean() - BASE_RMSE.mean()
    assert f"{OC10_RMSE.mean():.3f} ± {OC10_RMSE.std(ddof=1):.3f}" in row
    assert f"{delta:+.3f} ({ci.low:+.3f}, {ci.high:+.3f})" in row
    assert row.rstrip(" |").endswith(rt.format_p(res.pvalue))
    assert any(line.startswith(f"| {rt.MODELS[0][0]} | {BASE_RMSE.mean():.3f}") for line in md.splitlines())

    tex = (studies / "tables" / "ablation_graph.tex").read_text()
    assert "\\label{tab:ablation_graph}" in tex
    assert "10 ocean clusters" in tex
    assert "4 seeds" in tex and "8 seeds" in tex


def test_write_tables_gives_the_welch_difference_per_lead(studies):
    oc10 = OC10_RMSE[:, None] * (1 + 0.1 * np.arange(6))  # [seeds, leads]: the difference grows with the lead
    _study(studies / "study" / "oc10", oc10)

    rga.write_tables(n_seeds=4)

    expected = [rt.welch(oc10[:, lead], BASE_RMSE) for lead in range(6)]
    md = (studies / "tables" / "ablation_graph.md").read_text().splitlines()
    assert "| Model | Metric | L1 | L2 | L3 | L4 | L5 | L6 |" in md
    assert "| 10 ocean clusters | ΔRMSE [cm] | " + " | ".join(f"{d:+.3f}" for d, *_ in expected) + " |" in md
    assert "|  | p | " + " | ".join(rt.format_p(s[3]) for s in expected) + " |" in md
    assert (f"| {rt.MODELS[0][0]} | RMSE [cm] | " + " | ".join([f"{BASE_RMSE.mean():.3f} ± {BASE_RMSE.std(ddof=1):.3f}"] * 6)
            + " |") in md

    tex = (studies / "tables" / "ablation_graph_per_lead.tex").read_text()
    lines = tex.splitlines()
    assert "\\label{tab:ablation_graph_per_lead}" in tex
    assert "\\begin{tabular}{lcccccc}" in lines
    assert "Model & L1 & L2 & L3 & L4 & L5 & L6 \\\\" in lines
    assert "\\multirow{2}{*}{OC = 10} & " + " & ".join(f"${d:+.3f}$" for d, *_ in expected) + " \\\\" in lines
    assert " & " + " & ".join(f"(${lo:+.3f}$, ${hi:+.3f}$)" for _, lo, hi, _ in expected) + " \\\\" in lines
    assert (f"{rt.MODELS[0][0]} & " + " & ".join([f"${BASE_RMSE.mean():.3f} \\pm {BASE_RMSE.std(ddof=1):.3f}$"] * 6)
            + " \\\\") in lines
    assert "4 seeds" in tex and "8 seeds" in tex


def test_per_lead_latex_has_no_metric_column_and_no_p_rows(studies):
    oc10 = OC10_RMSE[:, None] * (1 + 0.1 * np.arange(6))
    _study(studies / "study" / "oc10", oc10)

    rga.write_tables(n_seeds=4)

    tex = (studies / "tables" / "ablation_graph_per_lead.tex").read_text()
    assert "Metric" not in tex
    assert "$p$" not in tex
    assert "10 ocean clusters" not in tex
    for s in [rt.welch(oc10[:, lead], BASE_RMSE) for lead in range(6)]:
        assert f"${rt.format_p(s[3])}$" not in tex
    # the table no longer says what its rows are, so the caption must
    assert "$\\Delta$RMSE" in tex and "RMSE" in tex and "95\\,\\% confidence interval" in tex


def test_short_labels_of_the_per_lead_table():
    assert {label: rga.SHORT_LABELS[label] for label, _ in rga.VARIANTS.values()} == {
        "10 ocean clusters": "OC = 10", "50 ocean clusters": "OC = 50",
        "5 M2M edges": "M2M = 5", "20 M2M edges": "M2M = 20",
        "1 M2G edge": "M2G = 1", "5 M2G edges": "M2G = 5"}


def test_lead_mean_table_is_the_mean_over_leads(studies):
    oc10 = OC10_RMSE[:, None] * (1 + 0.1 * np.arange(6))
    _study(studies / "study" / "oc10", oc10)

    rga.write_tables(n_seeds=4)

    md = (studies / "tables" / "ablation_graph.md").read_text()
    x = oc10.mean(1)
    assert f"| 10 ocean clusters | {x.mean():.3f} ± {x.std(ddof=1):.3f} | {x.mean() - BASE_RMSE.mean():+.3f} (" in md


@pytest.mark.parametrize("make_study", [
    lambda d: _study(d, OC10_RMSE[:3]),  # a seed failed to train
    lambda d: _study(d, OC10_RMSE, with_scores=False),  # the evaluation failed
    lambda d: None,  # no seed trained, so there is no manifest
])
def test_incomplete_variants_are_left_out_with_a_warning(studies, capsys, make_study):
    _study(studies / "study" / "oc10", OC10_RMSE)
    (studies / "study" / "oc50").mkdir()
    make_study(studies / "study" / "oc50")

    rga.write_tables(n_seeds=4)

    md = (studies / "tables" / "ablation_graph.md").read_text()
    assert "10 ocean clusters" in md
    assert "50 ocean clusters" not in md
    tex = (studies / "tables" / "ablation_graph_per_lead.tex").read_text()
    assert "OC = 10" in tex
    assert "OC = 50" not in tex
    assert "WARNING: 50 ocean clusters" in capsys.readouterr().out

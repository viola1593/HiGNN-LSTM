"""Batch C1b: the live helpers and outputs of evaluation_HiGNN.py survive."""

import ast
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "evaluation_HiGNN.py"
MODULE = ast.parse(SOURCE.read_text())


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _names(node, ctx):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)
            and isinstance(n.ctx, ctx)}


def test_helpers_that_are_still_called_survive():
    """Guard: the cleanup must not take the live helpers with it."""
    for name in ("plot_basintwsa_timesteps", "format_global_plot", "find_month_gap",
                 "plot_prediction", "plot_NSE_or_KGE"):
        assert _function(MODULE, name) is not None


def test_basin_scores_is_still_computed():
    """Guard: basin_scores feeds basin_scores.csv, so it is not dead."""
    evaluate = _function(MODULE, "evaluate")
    assert "basin_scores" in _names(evaluate, ast.Load)


def test_module_still_imports():
    import evaluation_HiGNN

    assert callable(evaluation_HiGNN.evaluate)

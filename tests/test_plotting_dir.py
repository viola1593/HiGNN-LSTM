"""The figure scripts live in scripts/plotting/ and still find their imports from scripts/ and src/."""
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PLOTTING = REPO / "scripts" / "plotting"
SCRIPTS = ["plot_ablation_rmse", "plot_basin_ablation", "plot_basin_latent_removal", "plot_latent_removal"]
# plot_ablation_rmse has no __main__ guard (importing it draws the figure), so it is only checked for its location
IMPORTABLE = ["plot_basin_ablation", "plot_basin_latent_removal", "plot_latent_removal"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_is_in_the_plotting_dir_only(name):
    assert (PLOTTING / f"{name}.py").is_file()
    assert not (REPO / "scripts" / f"{name}.py").exists()


@pytest.mark.parametrize("name", IMPORTABLE)
def test_script_imports_from_the_repo_root(name):
    """Import in a fresh interpreter, run as the docstrings say: from the repo root with PYTHONPATH=src/ only.

    `python scripts/plotting/<name>.py` puts the script's own dir first on sys.path, so the test does the same.
    """
    code = ("import sys; sys.path.insert(0, 'scripts/plotting'); "
            "import importlib.util, matplotlib; matplotlib.use('Agg'); "
            f"spec = importlib.util.spec_from_file_location('{name}', 'scripts/plotting/{name}.py'); "
            "spec.loader.exec_module(importlib.util.module_from_spec(spec))")
    env = {**os.environ, "PYTHONPATH": "src/"}
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr

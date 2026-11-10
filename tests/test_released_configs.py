"""configs/ holds the seed-42 config of each released study, and the scripts only name configs that exist."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
RELEASED = {"hignn_lstm.yaml", "no_mp.yaml", "no_ocean.yaml", "no_era5.yaml", "convlstm.yaml"}


def _scripts():
    return sorted((REPO / "scripts").rglob("*.py"))


def test_configs_dir_holds_the_released_studies_only():
    assert {p.name for p in (REPO / "configs").iterdir()} == RELEASED


def test_scripts_name_only_existing_configs():
    named = {(script.name, m) for script in _scripts() for m in re.findall(r"configs/[\w.-]+\.yaml", script.read_text())}
    assert named, "no script names a config"
    assert [(s, m) for s, m in named if not (REPO / m).is_file()] == []


def test_no_script_names_the_unreported_no_mp_no_era5_study():
    hits = [script.name for script in _scripts()
            if re.search(r"skipgnn_noera5|no MP \+ no ERA5", script.read_text(), re.IGNORECASE)]
    assert hits == []

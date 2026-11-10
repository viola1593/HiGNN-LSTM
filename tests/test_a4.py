"""Batch A4: join_GRACE-REC_era5.py area weighting, time-range selection, dead code."""

import ast
import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_data"
SCRIPT = SCRIPTS / "02b_join_GRACE-REC_era5.py"
REFERENCE = SCRIPTS / "01_read_GRACE-REC.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def join_mod():
    return _load(SCRIPT, "join_grace_rec_era5")


@pytest.fixture(scope="module")
def reference_mod():
    return _load(REFERENCE, "read_grace_rec")


def _global_quarter_degree_field(seed=None):
    """0.25 deg cell-centred grid: 720 lat rows from 89.875 down to -89.875."""
    lat = np.arange(89.875, -90.0, -0.25)
    lon = np.arange(0.125, 2.0, 0.25)  # 8 columns, divisible by COARSE_FACTOR
    assert lat.size == 720 and lon.size == 8
    if seed is None:
        data = np.ones((lat.size, lon.size))
    else:
        data = np.random.default_rng(seed).normal(size=(lat.size, lon.size))
    return xr.DataArray(data, coords={"lat": lat, "lon": lon}, dims=["lat", "lon"])


# --- item 1: coarsen_with_lat_weight must keep every latitude row ---------------


def test_coarsen_keeps_all_latitude_rows(join_mod):
    coarse = join_mod.coarsen_with_lat_weight(_global_quarter_degree_field(), COARSE_FACTOR=4)

    assert coarse.sizes["lat"] == 180
    assert coarse["lat"].values[0] == pytest.approx(89.5)
    assert coarse["lat"].values[-1] == pytest.approx(-89.5)


def test_coarsen_preserves_a_constant_field(join_mod):
    """Area-weighted mean of a constant field is that constant in every coarse cell."""
    coarse = join_mod.coarsen_with_lat_weight(_global_quarter_degree_field(), COARSE_FACTOR=4)

    assert np.isfinite(coarse.values).all()
    np.testing.assert_allclose(coarse.values, 1.0)


def test_coarsen_matches_read_grace_rec_reference(join_mod, reference_mod):
    """Same maths as the reference implementation (its cm<->m scaling cancels out)."""
    field = _global_quarter_degree_field(seed=0)

    got = join_mod.coarsen_with_lat_weight(field, COARSE_FACTOR=4)
    expected = reference_mod.coarsen_with_lat_weight(field, COARSE_FACTOR=4)

    np.testing.assert_allclose(got.values, expected.values, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["lat"].values, expected["lat"].values)


# --- item 2: overlapping time range must not depend on filesystem order --------


def _era5_file(path, varnames, start, periods):
    time = pd.date_range(start, periods=periods, freq="MS")
    lat = np.arange(1.875, 0.0, -0.25)
    lon = np.arange(0.125, 2.0, 0.25)
    shape = (time.size, lat.size, lon.size)
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    ds = xr.Dataset(
        {name: (("valid_time", "latitude", "longitude"), rng.normal(size=shape)) for name in varnames},
        coords={"valid_time": time, "latitude": lat, "longitude": lon},
    )
    ds.to_netcdf(path)
    ds.close()


def _twsa_file(path):
    # float times, as produced by read_GRACE-REC.py: year + (month-1)/12
    float_time = np.array([2003 + i / 12 + 1e-6 for i in range(12)])
    lat = np.array([0.5, 1.5])
    lon = np.array([0.5, 1.5])
    shape = (float_time.size, lat.size, lon.size)
    rng = np.random.default_rng(7)
    ds = xr.Dataset(
        {"twsa": (("time", "lat", "lon"), rng.normal(size=shape))},
        coords={"time": float_time, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    ds.close()


@pytest.fixture
def era5_workdir(tmp_path, monkeypatch):
    """A throwaway cwd holding the ERA5 tree the script globs ('data/raw/era5/...', relative to the repo root)."""
    cwd = tmp_path / "work"
    era5 = cwd / "data" / "raw" / "era5"
    era5_1, era5_2, static = era5 / "era5_1", era5 / "era5_2", era5 / "era5_static"
    for folder in (era5_1, era5_2, static):
        folder.mkdir(parents=True)

    # two files in the same directory -> their glob order is the filesystem's choice
    _era5_file(era5_1 / "a_evap.nc", ["e"], "2003-01-01", 6)      # ends 2003-06
    _era5_file(era5_1 / "b_runoff.nc", ["ro"], "2003-03-01", 10)  # starts 2003-03
    _era5_file(era5_2 / "pev.nc", ["pev"], "2003-01-01", 12)
    _era5_file(static / "static.nc", ["lsm", "z"], "2003-01-01", 12)

    twsa = cwd / "twsa.nc"
    _twsa_file(twsa)

    monkeypatch.chdir(cwd)
    return cwd, twsa


def _run_main(join_mod, monkeypatch, tmp_path, twsa, reverse):
    import glob as glob_module

    real_glob = glob_module.glob
    monkeypatch.setattr(join_mod.glob, "glob", lambda p: sorted(real_glob(p), reverse=reverse))
    monkeypatch.setattr(join_mod, "TWSA_FILE_PATH", str(twsa))
    out_dir = tmp_path / f"out_{'rev' if reverse else 'fwd'}"
    monkeypatch.setattr(join_mod, "OUTPUT_DIR", out_dir)

    join_mod.main()

    with xr.open_dataset(out_dir / "02_RecTWSA_era5.nc") as ds:
        return ds.time.values.copy()


@pytest.mark.parametrize("reverse", [False, True])
def test_time_range_is_independent_of_glob_order(join_mod, era5_workdir, monkeypatch, reverse):
    """Overlap is 2003-03 (latest start) .. 2003-06 (earliest end), whatever the file order."""
    tmp_path, twsa = era5_workdir

    times = _run_main(join_mod, monkeypatch, tmp_path, twsa, reverse=reverse)

    assert times[0] == np.datetime64("2003-03-01")
    assert times[-1] == np.datetime64("2003-06-01")


def test_drop_vars_tolerates_missing_expver_and_number(join_mod, era5_workdir, monkeypatch):
    """ERA5 files without 'expver'/'number' must not blow up the merge."""
    tmp_path, twsa = era5_workdir

    _run_main(join_mod, monkeypatch, tmp_path, twsa, reverse=False)


# --- item 3: dead statement ----------------------------------------------------


def test_no_discarded_astype_call():
    tree = ast.parse(SCRIPT.read_text())
    bare_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "astype"
    ]
    assert bare_calls == [], "result of .astype(...) is computed and thrown away"

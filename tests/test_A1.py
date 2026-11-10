"""Batch A1: prepareFCast.py must not leave dead code or unclosed datasets."""

import ast
import runpy
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_data" / "04_prepareFCast.py"


def _write_forecast_file(path, seed):
    time = np.arange(12)
    lat = np.array([-1.0, 0.0, 1.0])
    lon = np.array([0.0, 90.0, 200.0, 300.0])  # >180 exercises the eastings conversion
    dims = ("time", "lat", "lon")
    shape = (time.size, lat.size, lon.size)
    rng = np.random.default_rng(seed)
    ds = xr.Dataset(
        {
            "TWSC_full": (dims, rng.normal(size=shape)),
            "TWSC_linear": (dims, rng.normal(size=shape)),
            "TWSC_seasonal": (dims, rng.normal(size=shape)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    ds.close()


@pytest.fixture
def fcast_workdir(tmp_path, monkeypatch):
    """A throwaway cwd holding data/Li_csr_fcast/, the relative path the script uses."""
    folder = tmp_path / "data" / "Li_csr_fcast"
    folder.mkdir(parents=True)
    for seed, ym in enumerate(("2003-01", "2003-02")):
        _write_forecast_file(folder / f"TWSC_fcast_{ym}.nc", seed)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_no_unused_assignments():
    tree = ast.parse(SCRIPT.read_text())
    stored, loaded = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (stored if isinstance(node.ctx, ast.Store) else loaded).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                stored.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Attribute):
            loaded.add(node.attr)
    assert sorted(stored - loaded) == []


def test_opened_datasets_are_closed(fcast_workdir, monkeypatch):
    opened = []
    real_open_dataset = xr.open_dataset

    def tracking_open_dataset(*args, **kwargs):
        ds = real_open_dataset(*args, **kwargs)
        opened.append(ds)
        return ds

    monkeypatch.setattr(xr, "open_dataset", tracking_open_dataset)
    runpy.run_path(str(SCRIPT), run_name="__main__")

    assert len(opened) == 2
    # xarray sets Dataset._close to None once close() has run.
    assert [i for i, ds in enumerate(opened) if ds._close is not None] == []


def test_combined_output_unchanged(fcast_workdir):
    runpy.run_path(str(SCRIPT), run_name="__main__")

    folder = fcast_workdir / "data" / "Li_csr_fcast"
    with xr.open_dataset(folder / "combined.nc") as combined:
        assert set(combined.data_vars) == {"2003-01", "2003-02"}
        assert combined["2003-01"].dims == ("time", "lat", "lon")
        assert combined.sizes["time"] == 6
        assert combined["lon"].max() <= 180
        np.testing.assert_array_equal(
            combined["lon"].values, np.sort(combined["lon"].values)
        )

        with xr.open_dataset(folder / "TWSC_fcast_2003-01.nc") as src:
            expected = (src["TWSC_full"] - src["TWSC_linear"]).isel(time=slice(0, 6))
        expected = expected.assign_coords(
            lon=xr.where(expected["lon"] > 180, expected["lon"] - 360, expected["lon"])
        ).sortby("lon")
        np.testing.assert_allclose(combined["2003-01"].values, expected.values)

"""Batch A2: join_graceTWSA_era5.py — coarsening, metadata, time overlap, detrending."""

import ast
import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.signal import detrend

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_data" / "02a_join_graceTWSA_era5.py"


def _load_script():
    """Import the script from a throwaway cwd — it mkdirs data/ at import time."""
    spec = importlib.util.spec_from_file_location("join_graceTWSA_era5", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


mod = _load_script()

R = 6371000.0


def _quarter_degree_grid(n_lat=720, n_lon=8, n_time=2):
    """Descending 0.25 deg latitudes, as in the GRACE mascon file."""
    lat = 89.875 - 0.25 * np.arange(n_lat)
    lon = 0.25 * np.arange(n_lon)
    field = np.broadcast_to(lat[None, :, None], (n_time, n_lat, n_lon)).copy()
    return xr.DataArray(
        field,
        dims=("time", "lat", "lon"),
        coords={"time": pd.date_range("2018-01-01", periods=n_time, freq="MS"), "lat": lat, "lon": lon},
    )


def _band_areas(lat, dlon_deg):
    """Reference area weights: the band between the two edges of each lat row."""
    lat_edges = np.deg2rad(
        np.concatenate([
            [lat[0] - 0.5 * (lat[1] - lat[0])],
            0.5 * (lat[1:] + lat[:-1]),
            [lat[-1] + 0.5 * (lat[1] - lat[0])],
        ])
    )
    return R**2 * np.deg2rad(dlon_deg) * np.abs(np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))


# --- Finding 1: the southernmost latitude row must survive coarsening ------------------


def test_coarsening_keeps_every_latitude_row():
    da = _quarter_degree_grid()
    coarse = mod.coarsen_with_lat_weight(da, COARSE_FACTOR=4)

    assert coarse.sizes["lat"] == 180
    assert float(coarse["lat"][0]) == pytest.approx(89.5)
    assert float(coarse["lat"][-1]) == pytest.approx(-89.5)


def test_coarsening_uses_the_band_centred_on_each_row():
    da = _quarter_degree_grid()
    coarse = mod.coarsen_with_lat_weight(da, COARSE_FACTOR=4)

    lat = da["lat"].values
    weights = _band_areas(lat, dlon_deg=0.25).reshape(180, 4)
    expected = (lat.reshape(180, 4) * weights).sum(axis=1) / weights.sum(axis=1)

    np.testing.assert_allclose(coarse.isel(time=0, lon=0).values, expected, rtol=1e-12)


# --- Finding 2: the netCDF description attribute must be complete ---------------------


def test_description_attribute_round_trips(tmp_path):
    ds = xr.Dataset({"twsa": ("time", [0.0, 1.0])}, coords={"time": [0, 1]})
    ds.attrs["description"] = mod.DESCRIPTION
    path = tmp_path / "meta.nc"
    ds.to_netcdf(path)

    with xr.open_dataset(path) as written:
        description = written.attrs["description"]

    for fragment in (
        "e: evaporation",
        "sw4: volumetric soil water layer 4",
        "lai_high: leaf area index high vegetation",
        "lsm: land sea mask, z: geopotential",
        "Temporal matching:",
        f"±{mod.TOLERANCE_DAYS} days tolerance",
        'coordinate "GRACE_time"',
    ):
        assert fragment in description
    assert "TOLERANCE_DAYS" not in description


# --- Finding 3: overlap window and drop_vars must not depend on the filesystem --------


def _era5_stub(start, periods):
    time = pd.date_range(start, periods=periods, freq="MS")
    return xr.Dataset({"e": ("valid_time", np.zeros(periods))}, coords={"valid_time": time})


def test_era5_overlap_is_independent_of_file_order():
    early = _era5_stub("2018-01-01", 36)   # ends 2020-12
    late = _era5_stub("2019-01-01", 48)    # starts 2019-01, ends 2022-12
    middle = _era5_stub("2018-06-01", 40)

    expected = (np.datetime64("2019-01-01"), np.datetime64("2020-12-01"))
    for order in ([early, late, middle], [late, middle, early], [middle, early, late]):
        start, end = mod.era5_time_overlap(order)
        assert (np.datetime64(start, "D"), np.datetime64(end, "D")) == expected


def test_expver_drop_tolerates_missing_variables():
    tree = ast.parse(SCRIPT.read_text())
    drops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "drop_vars"
    ]
    assert drops
    for call in drops:
        assert any(kw.arg == "errors" and kw.value.value == "ignore" for kw in call.keywords)


# --- Finding 4: detrending must not poison NaN pixels or touch static fields ----------


def _era5_like():
    time = pd.date_range("2018-01-01", periods=24, freq="MS")
    lat = np.array([10.0, 9.0])
    lon = np.array([0.0, 1.0])
    rng = np.random.default_rng(0)
    values = rng.normal(size=(24, 2, 2)) + np.arange(24)[:, None, None]
    values[3, 0, 0] = np.nan
    return xr.Dataset(
        {
            "e": (("time", "lat", "lon"), values),
            # static fields carry a time dim in the raw ERA5 files, just like the dynamic ones
            "lsm": (("time", "lat", "lon"), np.ones((24, 2, 2))),
            "z": (("time", "lat", "lon"), np.full((24, 2, 2), 500.0)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )


def test_detrend_keeps_pixels_that_contain_a_nan():
    ds = _era5_like()
    detrended = mod.detrend_preserving_nans(ds["e"])

    pixel = detrended.isel(lat=0, lon=0).values
    assert np.isnan(pixel[3])
    assert np.isfinite(np.delete(pixel, 3)).all()
    assert detrended.dims == ("time", "lat", "lon")


def test_detrend_leaves_clean_pixels_identical():
    ds = _era5_like()
    detrended = mod.detrend_preserving_nans(ds["e"])

    clean = ds["e"].isel(lat=1, lon=1).values
    np.testing.assert_allclose(
        detrended.isel(lat=1, lon=1).values, detrend(clean, axis=0, type="linear")
    )


# --- Dead code: no unused imports, no import-time side effects -----------------------


def test_every_function_is_reachable_from_main():
    """No orphans: each helper must be called by main() or by something main() calls."""
    tree = ast.parse(SCRIPT.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def calls_made_in(node):
        return {
            n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in functions
        }

    reachable, frontier = {"main"}, ["main"]
    while frontier:
        for name in calls_made_in(functions[frontier.pop()]) - reachable:
            reachable.add(name)
            frontier.append(name)

    assert sorted(set(functions) - reachable) == []


def test_no_unused_imports():
    tree = ast.parse(SCRIPT.read_text())
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported[alias.asname or alias.name.split(".")[0]] = node.lineno

    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)

    assert sorted(name for name in imported if name not in used) == []


def test_detrend_skips_lsm_and_z_by_name():
    ds = _era5_like()
    before = {name: ds[name].values.copy() for name in ("lsm", "z", "e")}
    detrended = mod.detrend_time_varying(ds)

    for name in ("lsm", "z"):
        np.testing.assert_array_equal(detrended[name].values, before[name])
    np.testing.assert_allclose(
        detrended["e"].isel(lat=1, lon=1).values,
        detrend(before["e"][:, 1, 1], axis=0, type="linear"),
    )
    assert np.isnan(detrended["e"].isel(time=3, lat=0, lon=0).values)


# --- Regridding: output must be on the GRACE grid, like 02b's --------------------------


def _write_era5(path, name_values, lat, lon, times):
    dims = ("valid_time", "latitude", "longitude")
    ds = xr.Dataset({name: (dims, np.broadcast_to(values, (times.size, lat.size, lon.size)).astype("float32"))
                     for name, values in name_values.items()},
                    coords={"valid_time": times, "latitude": lat, "longitude": lon})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_output_is_on_the_grace_grid_without_an_extra_pole_row(tmp_path, monkeypatch):
    months = pd.date_range("2018-01-01", periods=24, freq="MS")

    # raw GRACE: 0.25 deg, lat ascending like the real file, time in days since 2002-01
    grace_lat = np.arange(-89.875, -85.0, 0.25)
    grace_lon = np.arange(0.125, 4.0, 0.25)
    grace_days = (months + pd.Timedelta(days=14) - pd.Timestamp("2002-01-01")).days.values.astype(float)
    cell_lat = np.floor(grace_lat) + 0.5   # lat of the 1 deg cell each row ends up in
    lwe = np.broadcast_to(cell_lat[None, :, None], (months.size, grace_lat.size, grace_lon.size))
    grace_file = tmp_path / "grace.nc"
    xr.Dataset({"lwe_thickness": (("time", "lat", "lon"), lwe)},
               coords={"time": grace_days, "lat": grace_lat, "lon": grace_lon}).to_netcdf(grace_file)

    # ERA5-like 0.25 deg grid that, like the real 721-row grid, ends with a lone pole row at -90
    era5_lat = np.arange(-85.0, -90.25, -0.25)   # 21 rows = 5 blocks of 4 + the -90 row
    era5_lon = np.arange(0.0, 4.0, 0.25)
    rows = np.arange(era5_lat.size)
    block_centre = np.where(rows < 20, -85.375 - rows // 4, -90.0)[:, None] * np.ones(era5_lon.size)

    work = tmp_path / "work"
    era5_dir = work / "data" / "raw" / "era5"
    rng = np.random.default_rng(0)
    _write_era5(era5_dir / "era5_1" / "a.nc", {"e": rng.normal(size=(months.size, era5_lat.size, era5_lon.size))},
                era5_lat, era5_lon, months)
    _write_era5(era5_dir / "era5_static" / "s.nc", {"lsm": 1.0, "z": block_centre}, era5_lat, era5_lon, months)

    monkeypatch.chdir(work)
    monkeypatch.setattr(mod, "GRACE_file_path", str(grace_file))
    monkeypatch.setattr(mod, "output_path", str(tmp_path / "GRACE_era5.nc"))
    monkeypatch.setattr(mod, "DETREND", False)

    mod.main()

    with xr.open_dataset(tmp_path / "GRACE_era5.nc") as out:
        np.testing.assert_array_equal(out.lat.values, [-85.5, -86.5, -87.5, -88.5, -89.5])
        np.testing.assert_array_equal(out.lon.values, [0.5, 1.5, 2.5, 3.5])
        assert not out.lsm.isnull().any()
        assert not out.z.isnull().any()
        twsa, lat = xr.broadcast(out.twsa, out.lat)
        np.testing.assert_allclose(twsa.values, lat.values)            # orientation kept
        z, lat = xr.broadcast(out.z, out.lat)
        np.testing.assert_allclose(z.values, lat.values + 0.125)       # each GRACE row gets the ERA5 block it overlaps


# --- One ERA5 month per GRACE solution ---------------------------------------------------


def _grace_solutions(times):
    """GRACE on the given timestamps, 1 x 1 grid, twsa = solution index."""
    times = pd.DatetimeIndex(times)
    return xr.DataArray(
        np.arange(times.size, dtype=float).reshape(-1, 1, 1),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [0.0], "lon": [0.0]},
    )


def _era5_months(year):
    time = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    return xr.Dataset({"e": (("time", "lat", "lon"), np.zeros((12, 1, 1)))},
                      coords={"time": time, "lat": [0.0], "lon": [0.0]})


def test_match_timesteps_uses_each_era5_month_once():
    grace_times = pd.DatetimeIndex(["2018-06-16", "2018-07-10", "2018-10-31T12", "2018-11-16", "2018-12-16"])
    ds = mod.match_timesteps(_grace_solutions(grace_times), _era5_months(2018), tolerance_days=20)

    time = ds.indexes["time"]
    assert time.is_unique and time.is_monotonic_increasing
    assert time.equals(pd.DatetimeIndex(["2018-06-01", "2018-07-01", "2018-10-01", "2018-11-01", "2018-12-01"]))
    assert pd.DatetimeIndex(ds.GRACE_time.values).equals(grace_times)
    np.testing.assert_array_equal(ds.twsa.values.ravel(), [0, 1, 2, 3, 4])


@pytest.mark.parametrize("year, grace_times, expected_months", [
    (2011, ["2011-10-16T12", "2011-10-31T12", "2011-12-28"], ["2011-10-01", "2011-11-01", "2011-12-01"]),
    (2015, ["2015-03-16T12", "2015-04-16", "2015-04-27", "2015-07-15T12"],
     ["2015-03-01", "2015-04-01", "2015-05-01", "2015-07-01"]),
    (2017, ["2017-01-21", "2017-03-31T12", "2017-04-24T12", "2017-05-13", "2017-06-10T12"],
     ["2017-01-01", "2017-03-01", "2017-04-01", "2017-05-01", "2017-06-01"]),
])
def test_short_solution_takes_free_neighbour_month(year, grace_times, expected_months):
    ds = mod.match_timesteps(_grace_solutions(grace_times), _era5_months(year), tolerance_days=20)

    assert ds.indexes["time"].equals(pd.DatetimeIndex(expected_months))


def test_solution_without_free_month_is_dropped():
    # April and May are taken by 04-16 and 05-16; March is 43 days from 04-27
    grace = _grace_solutions(["2015-04-16", "2015-04-27", "2015-05-16"])
    with pytest.warns(UserWarning, match="2015-04-27"):
        ds = mod.match_timesteps(grace, _era5_months(2015), tolerance_days=20)

    assert ds.indexes["time"].equals(pd.DatetimeIndex(["2015-04-01", "2015-05-01"]))
    np.testing.assert_array_equal(ds.twsa.values.ravel(), [0, 2])


# --- ERA5 is detrended over its full record, like in 02b ----------------------------------


def test_era5_is_detrended_over_the_full_record_before_the_time_cut(tmp_path, monkeypatch):
    era5_months = pd.date_range("2010-01-01", "2023-12-01", freq="MS")
    grace_months = pd.date_range("2018-01-01", "2019-12-01", freq="MS")

    # raw GRACE: 0.25 deg, lat ascending like the real file, time in days since 2002-01, one solution per month
    grace_lat = np.arange(-89.875, -85.0, 0.25)
    grace_lon = np.arange(0.125, 4.0, 0.25)
    grace_days = (grace_months + pd.Timedelta(days=14) - pd.Timestamp("2002-01-01")).days.values.astype(float)
    grace_file = tmp_path / "grace.nc"
    xr.Dataset({"lwe_thickness": (("time", "lat", "lon"), np.ones((grace_months.size, grace_lat.size, grace_lon.size)))},
               coords={"time": grace_days, "lat": grace_lat, "lon": grace_lon}).to_netcdf(grace_file)

    # quadratic + noise: a linear series would detrend to ~0 over any period
    rng = np.random.default_rng(0)
    x = np.linspace(-1.0, 1.0, era5_months.size)
    series = (10.0 * x**2 + rng.normal(scale=0.5, size=x.size)).astype("float32")

    # same value everywhere in space, so coarsening and regridding cannot change it
    work = tmp_path / "work"
    _write_era5(work / "data" / "raw" / "era5" / "era5_1" / "a.nc", {"swvl1": series[:, None, None]},
                np.arange(-85.0, -90.25, -0.25), np.arange(0.0, 4.0, 0.25), era5_months)

    monkeypatch.chdir(work)
    monkeypatch.setattr(mod, "GRACE_file_path", str(grace_file))
    monkeypatch.setattr(mod, "output_path", str(tmp_path / "GRACE_era5.nc"))
    monkeypatch.setattr(mod, "DETREND", True)

    mod.main()

    expected = xr.DataArray(detrend(series.astype(float)), dims="time", coords={"time": era5_months})
    with xr.open_dataset(tmp_path / "GRACE_era5.nc") as out:
        assert out.sizes["time"] > 0
        swvl1, expected = xr.broadcast(out.swvl1, expected.sel(time=out.time))
        np.testing.assert_allclose(swvl1.values, expected.values, rtol=1e-5, atol=1e-4)


# --- GRACE is detrended against time, over the full record, before the GRACE-FO cut --------


def test_grace_detrend_fits_against_time_not_sample_index():
    # irregular timestamps with single missing months and an 11-month gap, as in the real record
    months = pd.date_range("2002-04-01", "2025-06-01", freq="MS")
    keep = np.ones(months.size, dtype=bool)
    keep[[5, 17, 40, 41, 90, 133, 150]] = False
    keep[(months >= "2017-07-01") & (months < "2018-06-01")] = False
    rng = np.random.default_rng(0)
    times = months[keep] + pd.to_timedelta(rng.integers(5, 25, keep.sum()), unit="D")

    days = ((times - times[0]) / pd.Timedelta(days=1)).values
    a = np.array([[1.0, -2.0], [0.5, 3.0]])
    b = np.array([[0.01, -0.004], [0.0, 0.02]])
    linear = xr.DataArray(a + b * days[:, None, None], dims=("time", "lat", "lon"),
                          coords={"time": times, "lat": [10.0, 9.0], "lon": [0.0, 1.0]})

    detrended = mod.detrend_against_time(linear)

    assert detrended.dims == ("time", "lat", "lon")
    np.testing.assert_allclose(detrended.values, 0.0, atol=1e-8)
    # the index-based fit leaves a residual on the same series
    assert np.abs(detrend(linear.values, axis=0, type="linear")).max() > 0.1


def test_grace_is_detrended_over_the_full_record_and_cut_to_grace_fo(tmp_path, monkeypatch):
    # full GRACE + GRACE-FO record: mid-month, some missing months, 11-month gap 2017-07 ... 2018-05
    months = pd.date_range("2002-04-01", "2019-12-01", freq="MS")
    keep = np.ones(months.size, dtype=bool)
    keep[[5, 17, 40, 90, 133]] = False
    keep[(months >= "2017-07-01") & (months < "2018-06-01")] = False
    grace_times = months[keep] + pd.Timedelta(days=15)
    days = ((grace_times - grace_times[0]) / pd.Timedelta(days=1)).values

    # trend over 2002-2019 differs from the trend over GRACE-FO only: slope flips sign in 2018
    split = days[grace_times >= "2018-01-01"][0]
    series = np.where(days < split, 0.01 * days, 0.01 * split - 0.03 * (days - split))

    grace_lat = np.arange(-89.875, -85.0, 0.25)
    grace_lon = np.arange(0.125, 4.0, 0.25)
    grace_days = (grace_times - pd.Timestamp("2002-01-01")).days.values.astype(float)
    grace_file = tmp_path / "grace.nc"
    lwe = np.broadcast_to(series[:, None, None], (grace_times.size, grace_lat.size, grace_lon.size))
    xr.Dataset({"lwe_thickness": (("time", "lat", "lon"), lwe)},
               coords={"time": grace_days, "lat": grace_lat, "lon": grace_lon}).to_netcdf(grace_file)

    work = tmp_path / "work"
    _write_era5(work / "data" / "raw" / "era5" / "era5_1" / "a.nc", {"e": 0.0},
                np.arange(-85.0, -90.25, -0.25), np.arange(0.0, 4.0, 0.25),
                pd.date_range("2016-01-01", "2019-12-01", freq="MS"))

    monkeypatch.chdir(work)
    monkeypatch.setattr(mod, "GRACE_file_path", str(grace_file))
    monkeypatch.setattr(mod, "output_path", str(tmp_path / "GRACE_era5.nc"))
    monkeypatch.setattr(mod, "DETREND", True)

    mod.main()

    full = pd.Series(series - np.polyval(np.polyfit(days, series, 1), days), index=grace_times)
    fo = grace_times >= "2018-01-01"
    fo_only = pd.Series(series[fo] - np.polyval(np.polyfit(days[fo], series[fo], 1), days[fo]),
                        index=grace_times[fo])
    with xr.open_dataset(tmp_path / "GRACE_era5.nc") as out:
        written = pd.DatetimeIndex(out.GRACE_time.values)
        assert written.equals(grace_times[fo])                                   # cut to GRACE-FO
        twsa = out.twsa.transpose("time", ...).values.reshape(written.size, -1)
        np.testing.assert_allclose(twsa, np.broadcast_to(full[written].values[:, None], twsa.shape), atol=1e-8)
        assert np.abs(twsa[:, 0] - fo_only[written].values).max() > 1.0          # not a GRACE-FO-only fit


# --- Edge months: no time cut that compares GRACE mid-month with ERA5 month starts ---------


def test_edge_months_within_tolerance_are_kept(tmp_path, monkeypatch):
    # GRACE mid-month (06-16 ... 02-16) vs ERA5 month starts (01-01 ... 12-01): June and Dec are
    # 1 day from the ERA5 month midpoint, but lie outside [first GRACE date, last ERA5 date]
    era5_months = pd.date_range("2018-01-01", "2018-12-01", freq="MS")
    grace_times = pd.date_range("2018-06-01", "2019-02-01", freq="MS") + pd.Timedelta(days=15)

    grace_lat = np.arange(-89.875, -85.0, 0.25)
    grace_lon = np.arange(0.125, 4.0, 0.25)
    grace_days = (grace_times - pd.Timestamp("2002-01-01")).days.values.astype(float)
    grace_file = tmp_path / "grace.nc"
    xr.Dataset({"lwe_thickness": (("time", "lat", "lon"), np.ones((grace_times.size, grace_lat.size, grace_lon.size)))},
               coords={"time": grace_days, "lat": grace_lat, "lon": grace_lon}).to_netcdf(grace_file)

    work = tmp_path / "work"
    _write_era5(work / "data" / "raw" / "era5" / "era5_1" / "a.nc", {"e": 0.0},
                np.arange(-85.0, -90.25, -0.25), np.arange(0.0, 4.0, 0.25), era5_months)

    monkeypatch.chdir(work)
    monkeypatch.setattr(mod, "GRACE_file_path", str(grace_file))
    monkeypatch.setattr(mod, "output_path", str(tmp_path / "GRACE_era5.nc"))
    monkeypatch.setattr(mod, "DETREND", False)

    mod.main()

    with xr.open_dataset(tmp_path / "GRACE_era5.nc") as out:
        assert out.indexes["time"].equals(era5_months[5:])                       # 2018-06 ... 2018-12
        assert pd.DatetimeIndex(out.GRACE_time.values).equals(grace_times[:7])   # 06-16 ... 12-16


def test_grace_file_is_in_the_project_data_dir():
    """The mascon file is read from the repo's data/raw/, not from a directory outside the repository."""
    assert mod.GRACE_file_path == "data/raw/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"

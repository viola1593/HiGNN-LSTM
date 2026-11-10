"""The preprocessed dataset must live on the grid and land/ocean mask of the TWSA reconstruction."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_data"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_").removesuffix(".py"), SCRIPTS / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


read_rec = _load("01_read_GRACE-REC.py")
join_rec = _load("02b_join_GRACE-REC_era5.py")


def test_read_step_feeds_the_join_step_with_the_detrended_reconstruction():
    assert read_rec.variable == "grace_rec_detrended"
    assert read_rec.output_file_path == join_rec.TWSA_FILE_PATH


def test_coarsening_keeps_the_twsa_ocean_mask_and_does_not_dilute_coasts():
    lat = np.array([1.25, 0.75, 0.25, -0.25])
    lon = np.array([0.25, 0.75, 1.25, 1.75])
    nan = np.nan
    twsa = np.array([[nan, nan, 2.0, 4.0],
                     [nan, nan, nan, nan],
                     [5.0, 5.0, 1.0, 1.0],
                     [5.0, 5.0, 1.0, 1.0]])
    ds = xr.Dataset({"twsa": (("time", "lat", "lon"), twsa[None])},
                    coords={"time": [2000.0], "lat": lat, "lon": lon})

    out = read_rec.coarsen_with_lat_weight(ds, COARSE_FACTOR=2).twsa.isel(time=0).values

    assert np.isnan(out[0, 0])                      # pure ocean block stays ocean
    np.testing.assert_allclose(out[0, 1], 3.0)      # coast: mean of the valid cells (same row -> same area)
    np.testing.assert_allclose(out[1, 0], 5.0)
    np.testing.assert_allclose(out[1, 1], 1.0)


def _write_era5(path, name_values, lat, lon, times):
    dims = ("valid_time", "latitude", "longitude")
    ds = xr.Dataset({name: (dims, np.broadcast_to(values, (times.size, lat.size, lon.size)).astype("float32"))
                     for name, values in name_values.items()},
                    coords={"valid_time": times, "latitude": lat, "longitude": lon})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def test_join_output_is_on_the_twsa_grid_without_an_extra_pole_row(tmp_path, monkeypatch):
    # ERA5-like 0.25 deg grid that, like the real 721-row grid, ends with a lone pole row at -90
    era5_lat = np.arange(-85.0, -90.25, -0.25)   # 21 rows = 5 blocks of 4 + the -90 row
    era5_lon = np.arange(0.0, 4.0, 0.25)         # 16 columns = 4 blocks of 4
    times = pd.date_range("1979-01-01", periods=6, freq="MS")
    rows = np.arange(era5_lat.size)
    block_id = np.where(rows < 20, rows // 4, 99).astype(float)[:, None] * np.ones(era5_lon.size)

    era5_dir = tmp_path / "work" / "data" / "raw" / "era5"
    rng = np.random.default_rng(0)
    _write_era5(era5_dir / "era5_1" / "a.nc", {"t2m": rng.normal(size=(times.size, era5_lat.size, era5_lon.size))},
                era5_lat, era5_lon, times)
    _write_era5(era5_dir / "era5_static" / "s.nc", {"lsm": 1.0, "z": block_id}, era5_lat, era5_lon, times)

    # TWSA on its own 1 deg grid (cell centres at x.5), ocean cells are NaN
    twsa_lat = np.array([-85.5, -86.5, -87.5, -88.5, -89.5])
    twsa_lon = np.array([0.5, 1.5, 2.5, 3.5])
    twsa = rng.normal(size=(times.size, twsa_lat.size, twsa_lon.size))
    twsa[:, 0, 0] = np.nan
    float_time = 1979 + (np.arange(times.size) + 0.5) / 12
    twsa_file = tmp_path / "twsa.nc"
    xr.Dataset({"twsa": (("time", "lat", "lon"), twsa)},
               coords={"time": float_time, "lat": twsa_lat, "lon": twsa_lon}).to_netcdf(twsa_file)

    work = tmp_path / "work"
    monkeypatch.chdir(work)
    monkeypatch.setattr(join_rec, "TWSA_FILE_PATH", str(twsa_file))
    monkeypatch.setattr(join_rec, "OUTPUT_DIR", work / "out")

    join_rec.main()

    with xr.open_dataset(work / "out" / "02_RecTWSA_era5.nc") as out:
        np.testing.assert_array_equal(out.lat.values, twsa_lat)
        np.testing.assert_array_equal(out.lon.values, twsa_lon)
        assert not out.lsm.isnull().any()
        np.testing.assert_allclose(out.twsa.values, twsa)          # target untouched, ocean NaN kept
        # each TWSA row gets the ERA5 block it overlaps, the lone pole row is not used
        np.testing.assert_allclose(out.z.isel(time=0).values,
                                   np.arange(twsa_lat.size)[:, None] * np.ones(twsa_lon.size))

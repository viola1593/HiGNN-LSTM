"""C_grace: per-calendar-month mean of the GRACE months before 2008-03, ocean set to 0 as in GRACE_dataloader."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "preprocess_data" / "05_grace_climatology.py"
spec = importlib.util.spec_from_file_location("grace_climatology", SCRIPT)
grace_clim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grace_clim)


def test_climatology_uses_only_months_before_2008_03_and_zeroes_ocean():
    time = pd.date_range("2006-01-01", "2008-12-01", freq="MS")   # 2006-01..2008-02 in, 2008-03.. out
    lat = np.array([0.5, -0.5])                                    # float64 like the coarsened GRACE grid
    lon = np.array([0.5, 1.5])
    values = np.zeros((time.size, 2, 2))
    values[:] = (time.year * 100 + time.month).values[:, None, None].astype(float)
    values[time >= "2008-03-01"] = 1e6                             # must not leak into the mean
    twsa = xr.DataArray(values, dims=("time", "lat", "lon"), coords={"time": time, "lat": lat, "lon": lon})
    land = xr.DataArray([[True, False], [True, True]], dims=("lat", "lon"),
                        coords={"lat": lat.astype(np.float32), "lon": lon.astype(np.float32)})  # GRACE_era5.nc grid

    clim = grace_clim.monthly_climatology(twsa, land)

    assert clim.dims == ("month", "lat", "lon")
    assert clim.dtype == np.float32
    np.testing.assert_array_equal(clim.month.values, np.arange(1, 13))
    expected = np.array([(2006 + 2007 + 2008) * 100 / 3 + m if m <= 2 else (2006 + 2007) * 100 / 2 + m
                         for m in range(1, 13)], dtype=np.float32)
    np.testing.assert_allclose(clim.values[:, 0, 0], expected)
    np.testing.assert_allclose(clim.values[:, 1, 1], expected)
    np.testing.assert_array_equal(clim.values[:, 0, 1], 0.0)      # ocean

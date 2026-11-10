
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
import xarray as xr

'''
This script builds C_grace, the GRACE TWSA climatology over the months before GRACE_era5.nc starts
(2002-04 to 2008-02). It is used on GRACE in place of the reconstruction's training climatology.
It runs the TWSA steps of 02a in 02a's order, on the full GRACE/GRACE-FO record:
1. Detrend the full 2002-2025 record against time.
2. Convert longitudes to eastings/westings and sort them.
3. Coarsen by 4 with area weighting.
4. Sort latitude descending.
5. Assign each GRACE solution to a month with match_timesteps against a dummy monthly axis.
Then it takes the mean per calendar month over the months before 2008-03, sets ocean cells to 0
(as GRACE_dataloader does for the TWSA) and saves [month=12, lat, lon] in float32.
Check: the same pipeline on the months from 2008-03 onward must reproduce GRACE_era5.nc['twsa'].
'''

_spec = importlib.util.spec_from_file_location("join_graceTWSA_era5",
                                               Path(__file__).with_name("02a_join_graceTWSA_era5.py"))
join = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(join)

CLIM_END = np.datetime64('2008-03-01')  # first month of GRACE_era5.nc
GRACE_ERA5_PATH = join.output_path
OUTPUT_PATH = join.OUTPUT_DIR / "GRACE_climatology_pre2008.nc"


def grace_on_months(dataset_twsa: xr.Dataset) -> xr.Dataset:
    """02a's TWSA steps on the full record. Returns twsa and GRACE_time on a monthly axis (1st of month)."""
    start_date = pd.to_datetime('200201', format='%Y%m')
    dates = start_date + dataset_twsa.time.values.astype('timedelta64[D]')
    dataset_twsa = dataset_twsa.assign_coords(time=dates)

    dataset_twsa['twsa'] = join.detrend_against_time(dataset_twsa.lwe_thickness)
    dataset_twsa = dataset_twsa[['twsa']]  # the other variables are not used

    dataset_twsa['lon'] = xr.where(dataset_twsa['lon'] > 180, dataset_twsa['lon'] - 360, dataset_twsa['lon'])
    dataset_twsa = dataset_twsa.sortby('lon')
    dataset_twsa = join.coarsen_with_lat_weight(dataset_twsa, COARSE_FACTOR=join.COARSE_FACTOR_TWSA)
    ds_twsa = dataset_twsa.sortby('lat', ascending=False)

    # dummy monthly axis in place of ERA5, one month of margin on both sides of the record
    months = pd.date_range(dates.min() - pd.DateOffset(months=2), dates.max() + pd.DateOffset(months=1), freq='MS')
    return join.match_timesteps(ds_twsa.twsa, xr.Dataset(coords={'time': months}), join.TOLERANCE_DAYS)


def monthly_climatology(twsa: xr.DataArray, land: xr.DataArray) -> xr.DataArray:
    """Mean per calendar month over the months before CLIM_END, ocean -> 0 as in GRACE_dataloader."""
    clim = twsa.sel(time=twsa.time < CLIM_END).groupby('time.month').mean('time')
    return clim.where(land, 0.0).astype(np.float32)


def main():
    matched = grace_on_months(xr.open_dataset(join.GRACE_file_path, engine='netcdf4'))
    ref = xr.open_dataset(GRACE_ERA5_PATH)

    # pipeline check: the months from 2008-03 onward, cut as in 02a (by solution date, and ERA5 ends at ref's
    # last month), must reproduce GRACE_era5.nc
    gt = matched.GRACE_time
    in_02a = (((gt >= np.datetime64('2008-03-01')) & (gt <= np.datetime64('2010-07-31')))
              | ((gt >= np.datetime64('2018-01-01')) & (gt <= np.datetime64('2025-12-31'))))
    after = matched.sel(time=in_02a & (matched.time <= ref.time[-1]))
    np.testing.assert_array_equal(after.time.values, ref.time.values)
    np.testing.assert_array_equal(after.GRACE_time.values, ref.GRACE_time.values)
    np.testing.assert_array_equal(after.lat.values, ref.lat.values)
    np.testing.assert_array_equal(after.lon.values, ref.lon.values)
    diff = np.abs(after.twsa.values - ref.twsa.values).max()
    print(f"Pipeline check: {after.time.size} months from 2008-03 on, max |twsa - GRACE_era5.nc twsa| = {diff:.3e} cm")
    np.testing.assert_allclose(after.twsa.values, ref.twsa.values, rtol=0, atol=1e-6)

    land = ref.lsm.isel(time=0, drop=True) > 0  # land/ocean as in GRACE_dataloader (lsm is constant in time)
    clim = monthly_climatology(matched.twsa, land)

    pre = matched.sel(time=matched.time < CLIM_END)
    print(f"Climatology months {str(pre.time.values[0])[:7]} to {str(pre.time.values[-1])[:7]}, "
          f"{pre.time.size} solutions. Solutions per calendar month:")
    print(pre.time.dt.month.to_series().value_counts().sort_index().to_string())

    np.testing.assert_array_equal(clim.lat.values, ref.lat.values)
    np.testing.assert_array_equal(clim.lon.values, ref.lon.values)
    assert clim.sizes == {'month': 12, 'lat': ref.lat.size, 'lon': ref.lon.size}
    assert np.isfinite(clim.values[:, land.values]).all(), "climatology not finite on land"

    clim.name = 'twsa'
    clim.attrs['description'] = ('GRACE TWSA climatology (cm), mean per calendar month over the GRACE solutions '
                                 'matched to 2002-04..2008-02, detrended and coarsened as in 02a; ocean set to 0.')
    clim.attrs['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    clim.to_netcdf(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

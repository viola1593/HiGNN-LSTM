
import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr
from scipy.signal import detrend
import warnings

'''
This script prepares the GRACE TWSA and ERA5 data for training and evaluation. 
It performs the following steps:
1. Load GRACE TWSA data and ERA5 data from netCDF files.
2. Coarsen the data to a common spatial resolution (e.g., 1°x1°) using area-weighted averaging.
3. Detrend the data if specified (GRACE over its full record against time, then cut to 2008-03..2010-07 + GRACE-FO).
4. Match the GRACE and ERA5 time steps based on their timestamps, allowing for a specified tolerance in days.
5. Reindex ERA5 onto the GRACE grid.
6. Save the processed dataset to a new netCDF file for use in training and evaluation.

'''


COARSE_FACTOR_ERA5 = 4 #20
COARSE_FACTOR_TWSA = 4 # original 
DETREND = True
STATIC_VARS = ('lsm', 'z')  # have a time dim in the raw ERA5 files, but must not be detrended
TOLERANCE_DAYS= 20 
OUTPUT_DIR = Path("data/processed")
output_path = os.path.join(OUTPUT_DIR, "GRACE_era5.nc")
GRACE_file_path = "data/raw/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc"


DESCRIPTION = (
    'GRACE TWSA and selection of variables from ERA5.'
    '\n e: evaporation, pev: potential evaporation, ro: runoff, '
    'sw1: volumetric soil water layer 1, sw2: volumetric soil water layer 2, sw3: volumetric soil water layer 3, sw4: volumetric soil water layer 4,'
    '\n lai_high: leaf area index high vegetation, '
    # 'lai_low: leaf area index low vegetation, '
    'lsm: land sea mask, z: geopotential'
    f'\n Temporal matching: GRACE timestamps matched to one ERA5 month by midpoint distance (nearest pairs first, each month used once), within ±{TOLERANCE_DAYS} days tolerance.'
    '\n Original GRACE timestamps retained in coordinate "GRACE_time" for traceability.'
)


def coarsen_with_lat_weight(lwe, COARSE_FACTOR=4):
    # --- 2. Compute grid-cell area (exact on a sphere) ---
    R = 6371000.0  # mean Earth radius (m)
    lat = lwe["lat"].values
    lon = lwe["lon"].values


    # Convert to radians
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))


    # Compute latitude edges to get equal-area bands
    lat_edges = np.deg2rad(
        np.concatenate([
            [lat[0] - 0.5 * (lat[1] - lat[0])],
            0.5 * (lat[1:] + lat[:-1]),
            [lat[-1] + 0.5 * (lat[1] - lat[0])]
        ])
    )
    area_per_lat = R**2 * dlon * np.abs((np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])))  # m² per lon cell per lat row

    # Expand to 2D array
    area = xr.DataArray(
        np.repeat(area_per_lat[:, np.newaxis], len(lon), axis=1),
        coords={"lat": lwe["lat"], "lon": lwe["lon"]},
        dims=["lat", "lon"]
    )
    
    # --- 3. Compute mass (thickness * area) ---
    volume = lwe * area # volume in m³

    # --- 4. Coarsen from 0.25° to 1° in both lat/lon ---
    vol_coarse = volume.coarsen(lat=COARSE_FACTOR, lon=COARSE_FACTOR, boundary="pad").sum()
    area_coarse = area.coarsen(lat=COARSE_FACTOR, lon=COARSE_FACTOR, boundary="pad").sum()

    # --- 5. Area-weighted mean thickness (mass-conserving) ---
    lwe_1deg_m = vol_coarse / area_coarse
    

    return lwe_1deg_m


def detrend_preserving_nans(da: xr.DataArray) -> xr.DataArray:
    """Linearly detrend along the leading (time) axis, keeping NaN pixels NaN.

    scipy.signal.detrend propagates a single NaN over the whole time series of a
    pixel, so gaps are filled with 0 for the fit and masked out again afterwards.
    """
    mask = da.isnull()
    detrended = xr.DataArray(
        detrend(da.fillna(0).values, axis=0, type='linear'),
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs,
    )
    return detrended.where(~mask)


def detrend_against_time(da: xr.DataArray) -> xr.DataArray:
    """Remove a per-pixel least-squares fit a + b*t, with t in days since the first solution.

    GRACE is unevenly spaced (missing months, the 2017/18 gap), so the trend is fitted
    against time, not against sample index as scipy.signal.detrend does. Assumes no NaNs.
    """
    days = (da.time - da.time[0]) / np.timedelta64(1, 'D')
    X = np.stack([np.ones(len(days)), days.values], axis=1)       # [T, 2]
    Y = da.values.reshape(len(days), -1)                           # [T, pixels]
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return da - (X @ coef).reshape(da.shape)


def detrend_time_varying(ds: xr.Dataset) -> xr.Dataset:
    """Detrend every variable except the static fields, which are selected by name (see STATIC_VARS)."""
    for var in ds.data_vars:
        if var not in STATIC_VARS:
            ds[var] = detrend_preserving_nans(ds[var])
    return ds


def era5_time_overlap(datasets: list[xr.Dataset]) -> tuple:
    """Latest start and earliest end over all ERA5 files, independent of glob order."""
    starts = sorted(ds.valid_time.min().values for ds in datasets)
    ends = sorted(ds.valid_time.max().values for ds in datasets)
    return starts[-1], ends[0]


def match_timesteps(grace: xr.DataArray, era5: xr.Dataset,tolerance_days: int = TOLERANCE_DAYS, delta_days:int=14) -> xr.Dataset:
    """
    Match GRACE timestamps one-to-one to ERA5 months by the distance to the month
    midpoint (1st of month + 14 days), within tolerance_days. Nearest pairs are
    matched first and each GRACE solution and each ERA5 month is used at most once,
    so a short solution next to an already taken month moves to the free neighbour.

    Returns an xarray Dataset on the matched GRACE time axis containing
    TWSA and all ERA5 variables. Unmatched GRACE timesteps are dropped.
    """
    grace_times = pd.DatetimeIndex(grace.time.values)
    era5_times  = pd.DatetimeIndex(era5.time.values)
    era5_mids   = era5_times + pd.Timedelta(days=delta_days)
    # distance in days for every (GRACE, ERA5 month) pair; nearest pairs claim their month
    # first, so each GRACE solution and each ERA5 month is used at most once
    deltas = np.abs((grace_times.values[:, None] - era5_mids.values[None, :]) / np.timedelta64(1, 'D'))
    pairs = sorted(zip(*np.nonzero(deltas <= tolerance_days)), key=lambda p: deltas[p])
    matched = {}  # GRACE index -> ERA5 index
    for gi, ei in pairs:
        if gi not in matched and ei not in matched.values():
            matched[gi] = ei
    for gi in sorted(set(range(len(grace_times))) - set(matched)):
        warnings.warn(f"GRACE timestep {grace_times[gi].date()} unmatched "
                      f"(no free ERA5 month within ±{tolerance_days} days)")
    order = sorted(matched, key=matched.get)  # chronological by ERA5 month
    matched_grace_times = grace_times[order]
    matched_era5_times  = era5_times[[matched[gi] for gi in order]]
    n_dropped = len(grace_times) - len(matched_grace_times)
    if n_dropped:
        print(f"Dropped {n_dropped} GRACE timestep(s) with no ERA5 match within ±{tolerance_days} days.")
    print(f"Matched {len(matched_grace_times)}/{len(grace_times)} GRACE timesteps.")
    
    # Select GRACE slices and re-index them onto ERA5 timestamps -> nice and regular
    grace_sel = grace.sel(time=pd.DatetimeIndex(matched_grace_times))
    era5_sel  = era5.sel(time=pd.DatetimeIndex(matched_era5_times))
    grace_reindexed = grace_sel.assign_coords(time=pd.DatetimeIndex(matched_era5_times))
    ds = xr.merge([grace_reindexed.to_dataset(name="twsa"), era5_sel])
    # Keep the original GRACE timestamps as a non-dimension coordinate for traceability
    ds = ds.assign_coords(GRACE_time=("time", pd.DatetimeIndex(matched_grace_times)))
    return ds


def reindex_to_target( source: xr.Dataset | xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> xr.Dataset | xr.DataArray:
    target_ds = xr.Dataset({
            "lat": xr.DataArray(target_lat, dims="lat"),
            "lon": xr.DataArray(target_lon, dims="lon"),
        })
    reind = source.reindex_like(target_ds, method="nearest", tolerance=0.25)
    return reind


def main():

    # Path to the netCDF file for GRACE-derived TWSA

    # output_path = os.path.join(output_dir, "GRACE_era5_1deg_detrend_latavg_tempmatch.nc")
    # Load the netCDF file as an xarray dataset
    dataset_twsa = xr.open_dataset(GRACE_file_path, engine='netcdf4')

    # Print the dataset
    print(dataset_twsa)
    start_date = pd.to_datetime('200201', format='%Y%m')
    dates = start_date + dataset_twsa.time.values.astype('timedelta64[D]')

    dataset_twsa=dataset_twsa.assign_coords(time=dates)

    if DETREND:
        ### detrend the twsa data over the full GRACE/GRACE-FO record, against time
        dataset_twsa['twsa'] = detrend_against_time(dataset_twsa.lwe_thickness)
    else:
        dataset_twsa['twsa'] = dataset_twsa.lwe_thickness

    # GRACE 2008-03..2010-07 (2008-03..2009-02 are the inputs for the first target 2009-03) + GRACE-FO
    dataset_twsa = xr.concat([dataset_twsa.sel(time=slice(np.datetime64('2008-03-01'), np.datetime64('2010-07-31'))),
                              dataset_twsa.sel(time=slice(np.datetime64('2018-01-01'), np.datetime64('2025-12-31')))],
                             dim='time')
    print(dataset_twsa)
    #Comvert from degrees_east to eastings and westings
    dataset_twsa['lon'] = xr.where(dataset_twsa['lon'] > 180, dataset_twsa['lon'] - 360, dataset_twsa['lon'])
    # Sort the dataset by longitude to reorder it
    dataset_twsa = dataset_twsa.sortby('lon')

    if COARSE_FACTOR_TWSA:
        dataset_twsa = coarsen_with_lat_weight(dataset_twsa, COARSE_FACTOR=COARSE_FACTOR_TWSA)
        # dataset_twsa = dataset_twsa.coarsen(lon=COARSE_FACTOR_TWSA, lat=COARSE_FACTOR_TWSA, boundary='exact').sum()
        


    #--------------------------------------------------------------------------------------------------
    ### Get all ERA5 nc files 

    # Path to the netCDF files
    dirs = ['data/raw/era5/era5_1','data/raw/era5/era5_2', 'data/raw/era5/era5_static']
    nc_files = []
    # Loop through each directory and get all .nc file paths
    for dir in dirs:
        nc_files.extend(sorted(glob.glob(os.path.join(dir, '*.nc'))))
    
    # nc_files = ['data/era5/era5_25/data_stream-moda_stepType-avgad.nc', 'data/era5/era5_25/data_stream-moda_stepType-avgua.nc']

    # open all era5 datasets
    dataset_era5 = [xr.open_dataset(file_path, engine='netcdf4') for file_path in nc_files] #

    print(dataset_era5[0])

    # Find the overlapping time range (sorted by the time coordinate, not by glob order).
    # Only ERA5 is cut; match_timesteps drops GRACE solutions without an ERA5 month. The pad keeps
    # every ERA5 month whose midpoint (1st + 14 days) is within the tolerance of a GRACE date.
    era5_start, era5_end = era5_time_overlap(dataset_era5)
    pad = np.timedelta64(TOLERANCE_DAYS + 14, 'D')
    time_start = max(dataset_twsa.time.min().values - pad, era5_start)
    time_end = min(dataset_twsa.time.max().values + pad, era5_end)

    ds_twsa = dataset_twsa.sortby('lat', ascending=False)  # 02b convention: lat descending

    

    datasets_to_merge = []
    # vars_to_sum = ['tp', 'e', 'ro', 'pev']  # variables to be summed when coarsening

    for ds in dataset_era5:
        ds=ds.rename({'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'})
        ds = ds.assign_coords(time=ds.time.dt.floor('D')) # make sure all timestamps are at same hour
        # detrend ERA5 variables except the static fields lsm and z (over the full record, as in 02b)
        if DETREND:
            ds = detrend_time_varying(ds)
        # Select time range overlapping with TWSA
        ds = ds.sel(time=slice(time_start, time_end))



        #Comvert from degrees_east to eastings and westings
        ds['lon'] = xr.where(ds['lon'] > 180, ds['lon'] - 360, ds['lon'])
        # Sort the dataset by longitude to reorder it
        ds = ds.sortby('lon')
        
        if COARSE_FACTOR_ERA5:
                ds = coarsen_with_lat_weight(ds, COARSE_FACTOR=COARSE_FACTOR_ERA5)

        # if COARSE_FACTOR_ERA5:
        #     if np.array(a in ds.data_vars for a in vars_to_sum).any():
        #         ds = ds.coarsen(lon=COARSE_FACTOR_ERA5, lat=COARSE_FACTOR_ERA5, boundary='pad').sum()
        #     else:
        #         ds = ds.coarsen(lon=COARSE_FACTOR_ERA5, lat=COARSE_FACTOR_ERA5, boundary='pad').mean()
       
        # ds = ds.interp_like(ds_twsa)
        datasets_to_merge.append(ds)
   
    merged_era5 = xr.merge(datasets_to_merge).drop_vars(['expver', 'number'], errors='ignore')
    # regrid ERA5 onto the GRACE grid (as in 02b); the lone ERA5 pole row at -90 is dropped
    regridded_era5 = reindex_to_target(merged_era5, ds_twsa.lat.values, ds_twsa.lon.values)
    # Temporal matching
    match_df = match_timesteps(ds_twsa.twsa, regridded_era5, TOLERANCE_DAYS)

    # Add metadata to the dataset
    match_df.attrs['title'] = 'GRACE-derived TWSA and static variables of ERA5 regridded data'
    match_df.attrs['description'] = DESCRIPTION

    # combined_ds.attrs['source'] = 'TWSA: Li et al. 2021  https://doi.org/10.5061/dryad.z612jm6bt; ERA5: DOI: 10.24381/cds.f17050d7'
    match_df.attrs['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save the combined dataset if needed
    match_df.to_netcdf(output_path)




if __name__ == "__main__":
    main()

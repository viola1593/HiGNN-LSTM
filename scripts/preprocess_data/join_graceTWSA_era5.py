
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import xarray as xr
from scipy.signal import detrend
from scipy.interpolate import RegularGridInterpolator
import warnings

'''
This script prepares the GRACE TWSA and ERA5 data for training and evaluation. 
It performs the following steps:
1. Load GRACE TWSA data and ERA5 data from netCDF files.
2. Coarsen the data to a common spatial resolution (e.g., 1°x1°) using area-weighted averaging.
3. Detrend the data if specified.
4. Match the GRACE and ERA5 time steps based on their timestamps, allowing for a specified tolerance in days.
5. Reindex or regrid the data to a common lat/lon grid if necessary.
6. Save the processed dataset to a new netCDF file for use in training and evaluation.

'''


COARSE_FACTOR_ERA5 = 4 #20
COARSE_FACTOR_TWSA = 4 # original 
DETREND = True
TOLERANCE_DAYS= 20 
OUTPUT_DIR = Path("data/processed_GRACE")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def coarsen_with_lat_weight(lwe, COARSE_FACTOR=4):
    # --- 2. Compute grid-cell area (exact on a sphere) ---
    R = 6371000.0  # mean Earth radius (m)
    lat = lwe["lat"].values
    lon = lwe["lon"].values


    # Convert to radians
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))


    area_per_lat = R**2 * dlon * np.abs((np.sin(np.deg2rad(lat[1:])) - np.sin(np.deg2rad(lat[:-1]))))  # m² per lon cell per lat row

    # Expand to 2D array
    area = xr.DataArray(
        np.repeat(area_per_lat[:, np.newaxis], len(lon), axis=1),
        coords={"lat": lwe["lat"][:-1], # adjust for having one less lat row after differencing
                "lon": lwe["lon"]},
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

def assign_monthly_timecoord(ds1, time_var='time'):
    # Create year-month string coordinate for both
    ym = ds1[time_var].dt.strftime('%Y-%m').astype('datetime64[M]')
    # and assign as new time coordinate
    ds1_monthly = ds1.assign_coords(time=ym)
    
    return ds1_monthly

    
# Convert the time coordinate of ds1
def convert_float_to_datetime(float_time):
    year = float_time.astype(int)
    month = ((float_time - year) * 12 + 1).astype(int)
    return pd.to_datetime(year * 100 + month, format='%Y%m')


def match_grace_to_era5_times(
    grace_times: pd.DatetimeIndex,
    era5_times: pd.DatetimeIndex,
    tolerance_days: int = TOLERANCE_DAYS,
) -> pd.DataFrame:
    """
    For each GRACE observation, find the ERA5 month it belongs to.

    Strategy: match to the ERA5 timestamp whose month midpoint is
    closest to the GRACE date, within tolerance_days.

    GRACE CSR mascon products give a time stamp that is the center of
    the accumulation period. ERA5 monthly means are stamped at the
    first of the month (or sometimes the 15th depending on the product).
    We compute the midpoint of each ERA5 month and match against that.

    Returns a DataFrame with columns:
        grace_time      — original GRACE timestamp
        era5_time       — matched ERA5 timestamp
        delta_days      — difference in days (signed)
        matched         — bool, False if no match within tolerance
    """
    # Compute ERA5 month midpoints
    era5_mid = pd.DatetimeIndex([
        t + pd.Timedelta(days=14) for t in era5_times
    ])

    records = []
    for gt in grace_times:
        deltas = np.abs((era5_mid - gt).days)
        best_idx = int(np.argmin(deltas))
        best_delta = int((era5_mid[best_idx] - gt).days)

        if abs(best_delta) <= tolerance_days:
            records.append({
                "grace_time": gt,
                "era5_time": era5_times[best_idx],
                "delta_days": best_delta,
                "matched": True,
            })
        else:
            records.append({
                "grace_time": gt,
                "era5_time": pd.NaT,
                "delta_days": np.nan,
                "matched": False,
            })

    df = pd.DataFrame(records)
    n_matched = df["matched"].sum()
    n_total = len(df)
    print(f"Temporal matching: {n_matched}/{n_total} GRACE obs matched "
          f"(tolerance ±{tolerance_days} days). "
          f"Mean |delta| = {df.loc[df.matched, 'delta_days'].abs().mean():.1f} days")

    unmatched = df[~df["matched"]]
    if len(unmatched):
        print(f"  Unmatched GRACE times: {unmatched['grace_time'].tolist()}")

    return df


def match_timesteps(grace: xr.DataArray, era5: xr.Dataset,tolerance_days: int = TOLERANCE_DAYS, delta_days:int=14) -> xr.Dataset:
    """
    For each GRACE timestamp, find the ERA5 month whose midpoint
    (1st of month + 14 days) is closest, within tolerance_days.

    Returns an xarray Dataset on the matched GRACE time axis containing
    TWSA and all ERA5 variables. Unmatched GRACE timesteps are dropped.
    """
    grace_times = pd.DatetimeIndex(grace.time.values)
    era5_times  = pd.DatetimeIndex(era5.time.values)
    era5_mids   = era5_times + pd.Timedelta(days=delta_days)
    matched_grace_times = []
    matched_era5_times  = []
    for gt in grace_times:
        deltas   = np.abs((era5_mids - gt).days)
        best_idx = int(np.argmin(deltas))
        if deltas[best_idx] <= tolerance_days:
            matched_grace_times.append(gt)
            matched_era5_times.append(era5_times[best_idx])
        else:
            warnings.warn(
                f"GRACE timestep {gt.date()} unmatched "
                f"(nearest ERA5 midpoint {era5_mids[best_idx].date()} "
                f"is {deltas[best_idx]} days away, tolerance={tolerance_days})"
            )
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


def regrid_to_target(
    source: xr.Dataset | xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> xr.Dataset | xr.DataArray:
    """
    Bilinear interpolation of source onto target lat/lon grid.
    Uses xesmf if available (conservative regridding preferred for area-integrated
    quantities like TWSA), otherwise falls back to scipy bilinear.

    Conservative regridding is more physically correct for TWSA because it
    preserves the area-weighted integral of the field. Bilinear is acceptable
    when grids are close in resolution.
    """
    try:
        import xesmf as xe
        print("Using xesmf conservative regridding.")

        # xesmf needs a Dataset with lat/lon as 2D or 1D coords
        if isinstance(source, xr.DataArray):
            source_ds = source.to_dataset(name=source.name or "var")
        else:
            source_ds = source

        target_ds = xr.Dataset({
            "lat": xr.DataArray(target_lat, dims="lat"),
            "lon": xr.DataArray(target_lon, dims="lon"),
        })

        regridder = xe.Regridder(source_ds, target_ds, method="conservative")
        result = regridder(source_ds)
        return result if isinstance(source, xr.Dataset) else result[source.name or "var"]

    except ImportError:
        print("xesmf not available — falling back to scipy bilinear interpolation.")
        return _scipy_regrid(source, target_lat, target_lon)


def _scipy_regrid(
    source: xr.Dataset | xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> xr.Dataset | xr.DataArray:
    """Bilinear regrid using scipy. Loops over time steps."""

    def _regrid_one_var(da: xr.DataArray) -> xr.DataArray:
        src_lat = da.lat.values
        src_lon = da.lon.values
        out = np.full((len(da.time), len(target_lat), len(target_lon)), np.nan)

        for t_idx in range(len(da.time)):
            data_slice = da.isel(time=t_idx).values
            # Mask NaNs: scipy can't interpolate through them
            if np.all(np.isnan(data_slice)):
                continue
            # Replace NaN with 0 for interpolation (ocean fill); refine if needed
            data_filled = np.where(np.isnan(data_slice), 0.0, data_slice)
            interp_fn = RegularGridInterpolator(
                (src_lat, src_lon),
                data_filled,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            lon_grid, lat_grid = np.meshgrid(target_lon, target_lat)
            pts = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
            out[t_idx] = interp_fn(pts).reshape(len(target_lat), len(target_lon))

        return xr.DataArray(
            out,
            dims=["time", "lat", "lon"],
            coords={"time": da.time, "lat": target_lat, "lon": target_lon},
            name=da.name,
            attrs=da.attrs,
        )

    if isinstance(source, xr.DataArray):
        return _regrid_one_var(source)
    else:
        return xr.Dataset({v: _regrid_one_var(source[v]) for v in source.data_vars})
    
def diagnose_matching(match_df: pd.DataFrame, output_dir: Path) -> None:
    """Save matching diagnostics to CSV for inspection."""
    match_df.to_csv(output_dir / "grace_era5_temporal_match.csv", index=False)

    # Distribution of delta days
    matched = match_df[match_df["matched"]]
    print(f"\nDelta days statistics:")
    print(matched["delta_days"].describe())

    # Flag large offsets
    large_offset = matched[matched["delta_days"].abs() > 10]
    if len(large_offset):
        print(f"\nGRACE observations with >10 day offset ({len(large_offset)}):")
        print(large_offset[["grace_time", "era5_time", "delta_days"]].to_string())



def main():

    # Path to the netCDF file for GRACE-derived TWSA
    file_path = "data/10_2025_GRACEmasconCSR/CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections(1).nc"
    output_dir = Path("data/latest_data0326")
    output_dir.mkdir(parents=True, exist_ok=True)
    # output_path = os.path.join(output_dir, "GRACE_era5_1deg_detrend_latavg_tempmatch.nc")
    output_path = os.path.join(output_dir, "GRACE-FO_era5_detrend_1deg_latavg_tempmatch.nc")
    # Load the netCDF file as an xarray dataset
    dataset_twsa = xr.open_dataset(file_path, engine='netcdf4')

    # Print the dataset
    print(dataset_twsa)
    start_date = pd.to_datetime('200201', format='%Y%m')
    dates = start_date + dataset_twsa.time.values.astype('timedelta64[D]')

    dataset_twsa=dataset_twsa.assign_coords(time=dates)

    dataset_twsa = dataset_twsa.sel(time=slice(np.datetime64('2018-01-01'), np.datetime64('2025-12-31'))) # limit to GRACE-FO
    import pdb; pdb.set_trace()
    print(dataset_twsa)
    if DETREND:
        ### detrend the twsa data
        detrend_twsa = detrend(dataset_twsa.lwe_thickness.values, axis=0, type='linear')
        dataset_twsa['twsa'] = (('time', 'lat', 'lon'), detrend_twsa)
    else: 
        dataset_twsa['twsa'] = dataset_twsa.lwe_thickness
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
    dirs = ['data/era5/era5_1','data/era5/era5_2', 'data/era5/era5_static']
    nc_files = []
    # Loop through each directory and get all .nc file paths
    for dir in dirs:
        nc_files.extend(glob.glob(os.path.join(dir, '*.nc')))
    
    # nc_files = ['data/era5/era5_25/data_stream-moda_stepType-avgad.nc', 'data/era5/era5_25/data_stream-moda_stepType-avgua.nc']

    # open all era5 datasets
    dataset_era5 = [xr.open_dataset(file_path, engine='netcdf4') for file_path in nc_files] #

    print(dataset_era5[0])

    # Find the overlapping time range
    time_start = max(dataset_twsa.time.min(), dataset_era5[0].valid_time.min()).values
    time_end = min(dataset_twsa.time.max(), dataset_era5[1].valid_time.max()).values # dataset_era5[1] is the one with the earliest end time among the ERA5 datasets, adjust if needed

    ds_twsa=dataset_twsa.sel(time=slice(time_start, time_end))

    

    datasets_to_merge = []
    # vars_to_sum = ['tp', 'e', 'ro', 'pev']  # variables to be summed when coarsening

    for ds in dataset_era5:
        ds=ds.rename({'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'})
        # Select time range overlapping with TWSA
        ds = ds.sel(time=slice(time_start, time_end))
        ds = ds.assign_coords(time=ds.time.dt.floor('D')) # make sure all timestamps are at same hour
        # detrend ERA5 variables
        if DETREND:
            for var in ds.data_vars:
                if var != 'lsm' and var != 'z': # don't detrend land sea mask or geopotential

                    # mask nans out before detrending, then put nans back in
                    mask = ds[var].isnull()
                    ds[var] = (('time', 'lat', 'lon'), detrend(ds[var].fillna(0).values, axis=0, type='linear'))
                    ds[var] = ds[var].where(~mask) # put nans back in
        


        # ds = assign_monthly_timecoord(ds)
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
   
    merged_era5 = xr.merge(datasets_to_merge).drop_vars(['expver', 'number'])
    target_lat = merged_era5.lat.values
    target_lon = merged_era5.lon.values
    regridded_twsa = reindex_to_target(ds_twsa, target_lat, target_lon)
    # regridded_era5 = reindex_to_target(merged_era5, target_lat, target_lon)
    # Temporal matching
    match_df = match_timesteps(regridded_twsa.twsa, merged_era5, TOLERANCE_DAYS)


    

    import pdb; pdb.set_trace()

    # Add metadata to the dataset
    match_df.attrs['title'] = 'GRACE-derived TWSA and static variables of ERA5 regridded data'
    match_df.attrs['description'] = 'GRACE TWSA and selection of variables from ERA5.' \
    '\n e: evaporation, pev: potential evaporation, ro: runoff, ' \
    'sw1: volumetric soil water layer 1, sw2: volumetric soil water layer 2, sw3: volumetric soil water layer 3, sw4: volumetric soil water layer 4,' \
    '\n lai_high: leaf area index high vegetation, ' \
    # 'lai_low: leaf area index low vegetation, ' \
    'lsm: land sea mask, z: geopotential'\
    '\n Temporal matching: GRACE timestamps matched to ERA5 month whose midpoint is closest, within ±{TOLERANCE_DAYS} days tolerance.' \
    '\n Original GRACE timestamps retained in coordinate "GRACE_time" for traceability.'

    # combined_ds.attrs['source'] = 'TWSA: Li et al. 2021  https://doi.org/10.5061/dryad.z612jm6bt; ERA5: DOI: 10.24381/cds.f17050d7'
    match_df.attrs['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save the combined dataset if needed
    match_df.to_netcdf(output_path)




if __name__ == "__main__":
    main()

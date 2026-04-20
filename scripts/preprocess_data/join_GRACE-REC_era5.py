
# In this script we check the data modalities (spatial/temporal resolution) of the ERA5 data and GRACE TWSA. 
# Then we join them in a common xarray Dataset to populate the graph nodes with those features
import glob
from scipy.signal import detrend
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
import xarray as xr

'''
This script reads in the reconstructed TWSA data from Li et al. (https://datadryad.org/dataset/doi:10.5061/dryad.z612jm6bt) and the ERA5 data, checks their spatial and temporal resolution, and joins them into a common xarray Dataset.
The TWSA data is coarsened to 1 degree latitude resolution using an area-weighted average to ensure that the total water storage anomaly (TWSA) values are conserved when aggregating to a coarser grid. 
The ERA5 data is also coarsened to match the TWSA resolution. 
The script then saves the combined dataset as a netCDF file for use in training and evaluation of the model.
'''

COARSE_FACTOR_ERA5 = 4 #20
COARSE_FACTOR_TWSA = None # already coarsened in loading step
SHORTCUT = False # shortcut for exchanging the TWSA reconstruction
# Path to the netCDF file for GRACE TWSA
FILE_PATH = "data/grace_rec_data_global_1deglatAVG_detrended.nc"
OUTPUT_DIR = Path("data/latest_data0326")

def coarsen_with_lat_weight(lwe, COARSE_FACTOR=4):
    # --- 2. Compute grid-cell area (exact on a sphere) ---
    R = 6371000.0  # mean Earth radius (m)
    lat = lwe["lat"].values
    lon = lwe["lon"].values


    # Convert to radians
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
    dlon = np.deg2rad(np.abs(lon[1] - lon[0]))

    # formula for spherical surface area element 
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


# Convert the time coordinate of ds1
def convert_float_to_datetime(float_time):
    year = float_time.astype(int)
    month = ((float_time - year) * 12 + 1).astype(int)
    return pd.to_datetime(year * 100 + month, format='%Y%m')



def main():
    # Path to the netCDF file for GRACE TWSA

    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file_path = os.path.join(OUTPUT_DIR, "TWSA-REC_era5_detrend_latavg.nc")
    
    # Load the netCDF file as an xarray dataset
    dataset_twsa = xr.open_dataset(FILE_PATH, engine='netcdf4')
    if COARSE_FACTOR_TWSA:
        dataset_twsa = dataset_twsa.coarsen(lon=COARSE_FACTOR_TWSA, lat=COARSE_FACTOR_TWSA, boundary='exact').sum()


    # Print the dataset
    print(dataset_twsa)


    dataset_twsa.time.data.astype(int)
    new_time=convert_float_to_datetime(dataset_twsa.time.data)
    dataset_twsa=dataset_twsa.assign_coords(time=new_time)

    print(dataset_twsa.twsa)

    if SHORTCUT:
        ### short cut here to join new grace_rec with old dataset
        old_dataset_twsa = xr.open_dataset("data/twsa_rec_era5_global_1deglatAVG_detrend.nc", engine='netcdf4')
        old_dataset_twsa = old_dataset_twsa.drop_vars(['twsa'])

        combined_ds = xr.merge([dataset_twsa, old_dataset_twsa])
    else:

        # # Path to the netCDF files
        dirs = ['data/era5/era5_1','data/era5/era5_2', 'data/era5/era5_static']
        # Initialize a list to store the file paths
        nc_files = []
        # Loop through each directory and get all .nc file paths
        for dir in dirs:
            nc_files.extend(glob.glob(os.path.join(dir, '*.nc')))

        dataset_era5 = [xr.open_dataset(file_path, engine='netcdf4') for file_path in nc_files] #

        print(dataset_era5)

        time_start = max(dataset_twsa.time.min(), dataset_era5[0].valid_time.min()).values
        time_end = min(dataset_twsa.time.max(), dataset_era5[1].valid_time.max()).values # dataset_era5[1] is the one with the earliest end time among the ERA5 datasets, adjust if needed



        # Initialize a list to store the datasets
        datasets_to_merge = []
        
        for ds in dataset_era5:
                
            ds=ds.rename({'latitude': 'lat', 'longitude': 'lon', 'valid_time': 'time'})
            ds = ds.assign_coords(time=ds.time.dt.floor('D')) # make sure all timestamps are at same hour
            for var in ds.data_vars:
                if var != 'lsm' and var != 'z': # don't detrend land sea mask or geopotential
                    # mask nans out before detrending, then put nans back in
                    mask = ds[var].isnull()
                    ds[var] = (('time', 'lat', 'lon'), detrend(ds[var].fillna(0).values, axis=0, type='linear'))
                    ds[var] = ds[var].where(~mask) # put nans back in
            
            
            #Comvert from degrees_east to eastings and westings
            ds['lon'] = xr.where(ds['lon'] > 180, ds['lon'] - 360, ds['lon'])
            ds = ds.sortby('lon')
            if COARSE_FACTOR_ERA5:
                ds = coarsen_with_lat_weight(ds, COARSE_FACTOR=COARSE_FACTOR_ERA5)
                #ds = ds.coarsen(lon=COARSE_FACTOR_ERA5, lat=COARSE_FACTOR_ERA5, boundary='pad').mean()
       
            
            datasets_to_merge.append(ds)

        
        combined_era5 = xr.merge(datasets_to_merge).drop_vars(['expver', 'number'])
        # regrid everything to era5 
        regridded_twsa = reindex_to_target(dataset_twsa, target_lat=combined_era5.lat, target_lon=combined_era5.lon)
        combined_ds = xr.merge([regridded_twsa, combined_era5])
        combined_ds = combined_ds.sel(time=slice(time_start, time_end))




    # Add metadata to the dataset
    combined_ds.attrs['title'] = 'Rec TWSA (detrended) and static variables of ERA5 regridded data'
    combined_ds.attrs['description'] = 'Reconstructed TWSA and selection of variables from ERA5 (interpolated to match the TWSA resolution).\n e: evaporation, pev: potential evaporation, ro: runoff, sw1: volumetric soil water layer 1, sw2: volumetric soil water layer 2, sw3: volumetric soil water layer 3, sw4: volumetric soil water layer 4, lai_high: leaf area index high vegetation, lai_low: leaf area index low vegetation, lsm: land sea mask, z: geopotential'
    combined_ds.attrs['source'] = 'TWSA: Li et al. 2021  https://doi.org/10.5061/dryad.z612jm6bt; ERA5: DOI: 10.24381/cds.f17050d7'
    combined_ds.attrs['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    # Save the combined dataset if needed
    combined_ds.to_netcdf(output_file_path)


    # import pdb; pdb.set_trace()
    combined_ds.isel(time=0).z.plot()
    plt.savefig('geopotential_coarse.png')
    plt.close()


if __name__ == "__main__":
    main()

# This script reads the GRACE data from the .mat file and saves it as a .nc file
# decide if you want to extract only the data for Spain or the global data
# decide if you want to coarse the resolution of the data

import scipy.io
import xarray as xr
import h5py
import numpy as np
import matplotlib.pyplot as plt


'''This script reads the .mat file of the reconstruced TWSA data from Li et al. (https://datadryad.org/dataset/doi:10.5061/dryad.z612jm6bt)
and saves it as a .nc file. 
It also allows for coarsening the data and extracting only the data for Spain if desired.
The coarsening is done with an area-weighted average to ensure that the total water storage anomaly (TWSA) values are conserved when aggregating to a coarser grid. '''

# ------------------------------------------------------------------------------------------------------------------------------
# set variables

COARSE_FACTOR = 2 #10
file_path = 'data/raw/GRID_CSR_GRACE_REC.mat'
variable = 'grace_rec_detrended'
output_file_path = "data/processed/01_reconstructedTWSA_data.nc"
# ------------------------------------------------------------------------------------------------------------------


def coarsen_with_lat_weight(lwe, COARSE_FACTOR=4):
    # --- 2. Compute grid-cell area (exact on a sphere) ---
    R = 6371000.0  # mean Earth radius (m)
    lat = lwe["lat"].values
    lon = lwe["lon"].values
    lwe = lwe * 0.01 # convert cm -> m if needed

    # Convert to radians
    dlat = np.deg2rad(np.abs(lat[1] - lat[0]))
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
    # only count the area of valid (land) cells: pure ocean blocks stay NaN (0/0) and coastal cells are not diluted
    area = area.where(lwe.notnull())

    # --- 4. Coarsen from 0.25° to 1° in both lat/lon ---
    vol_coarse = volume.coarsen(lat=COARSE_FACTOR, lon=COARSE_FACTOR, boundary="exact").sum()
    area_coarse = area.coarsen(lat=COARSE_FACTOR, lon=COARSE_FACTOR, boundary="exact").sum()

    # --- 5. Area-weighted mean thickness (mass-conserving) ---
    lwe_1deg_m = vol_coarse / area_coarse
    lwe_1deg_cm = lwe_1deg_m * 100.0  # back to cm

    # Done!
    lwe_1deg_cm.attrs["long_name"] = "Total Water Storage Anomaly (1° area-weighted)"
    lwe_1deg_cm.attrs["units"] = "cm"
    return lwe_1deg_cm

# ------------------------------------------------------------------------------------------------------------------------------
def main():
    # Open the .mat file
    with h5py.File(file_path, 'r') as mat_file:
        # List all groups
        print("Keys: %s" % mat_file.keys())
        
        # Get the data
        data = mat_file[variable][:]
        
        # Get the coordinates
        lon = mat_file['long'][:]
        lat = mat_file['lat'][:]
        time = mat_file['time'][:]  # Assuming time is also stored in the file

    # Convert the data and coordinates to NumPy arrays
    data_array = np.array(data)
    lon_array = np.array(lon)
    lat_array = np.array(lat)
    time_array = np.array(time)

    # check if the time data is evenly spaced

    time_diff = np.diff(time)
    print(time_diff.max())
    print(time_diff.min())



    # Create an xarray dataset from the NumPy arrays
    dataset = xr.Dataset(
        {
            'twsa': (['time', 'lat', 'lon'], np.transpose(data_array,(0,2, 1)))
        },
        coords={
            'time': time_array[0],
            'lon': lon_array[0],
            'lat': lat_array[0]
            
        }
    )


    #Comvert from degrees_east to eastings and westings
    # Extract the longitude values
    longitudes = dataset['lon'].values

    # Apply the transformation
    longitudes = [(lon - 360) if lon > 180 else lon for lon in longitudes]

    # Assign the transformed longitude values back to the dataset
    dataset = dataset.assign_coords(lon=longitudes)
    # Sort the dataset by longitude to reorder it
    dataset = dataset.sortby('lon')


    dataset=coarsen_with_lat_weight(dataset, COARSE_FACTOR=COARSE_FACTOR)

    mean = dataset.mean(dim=['lon', 'lat'])

    mean.twsa.plot(label='twsa_mean')
    
    plt.legend()
    plt.savefig(f'{variable}.png')
    plt.close()

    dataset.to_netcdf(output_file_path)


if __name__ == "__main__":    
    main()
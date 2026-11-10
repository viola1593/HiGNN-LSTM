import xarray as xr
import re
from pathlib import Path
'''

This script prepares the forecast data from Li et al 2026 (https://doi.org/10.1029/2025WR041710) for evaluation and comparison with our model. 
It reads in the netcdf files, processes the data, and saves it in a format that can be easily used for evaluation.'''


outputfolder = "data/Li_csr_fcast"

folder = Path("data/Li_csr_fcast")

dataset = xr.Dataset()

for file in sorted(folder.glob("*.nc")):
    match = re.search(r"\d{4}-\d{2}", file.name)
    if match is None:
        continue
    ds = xr.open_dataset(file)
    ds_diff = ds['TWSC_full']-ds['TWSC_linear']

    dataset[match.group()] = ds_diff.isel(time=slice(0,6)).assign_coords(time=range(0,6))
    ds.close()



#Comvert from degrees_east to eastings and westings
dataset['lon'] = xr.where(dataset['lon'] > 180, dataset['lon'] - 360, dataset['lon'])
# Sort the dataset by longitude to reorder it
dataset = dataset.sortby('lon')

dataset = dataset.transpose('time', 'lat', 'lon')

dataset.attrs['time'] = 'Lead time from initialization date'
dataset.attrs['year_month'] = 'Initialization date'
dataset.attrs['description']='TWSC forecast from Li et al 2026:https://doi.org/10.1029/2025WR041710. Data: TWSC_full-TWSC_linear'
dataset.attrs['data']='TWSC_full-TWSC_linear'
dataset.to_netcdf(path= outputfolder+"/combined.nc")
print('done')
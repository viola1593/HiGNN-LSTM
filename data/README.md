# Data

This document describes the datasets required to train and evaluate the HiGNN-LSTM and ConvLSTM2D models, how to download them, and how to preprocess them into the format expected by the training pipeline.

---

## Preprocessed Datasets

If you do not wish to run the preprocessing pipeline from scratch, the preprocessed datasets of the revised
results are available on Zenodo: [10.5281/zenodo.24135096](https://doi.org/10.5281/zenodo.24135096). The earlier version of the record
([10.5281/zenodo.19664592](https://doi.org/10.5281/zenodo.19664592)) holds the data of the previous version, with
different file names and an older preprocessing; it does not match the configs in this repository.
The GRACE climatology (`GRACE_climatology_pre2008.nc`) is not part of the record: build them
from the CSR mascons (see below) with `05_grace_climatology.py`; 

### Expected layout

```
data/
├── raw/
│   ├── GRID_CSR_GRACE_REC.mat                  # GRACE-REC, input of 01_read_GRACE-REC.py
│   └── era5/{era5_1,era5_2,era5_static}/*.nc   # ERA5 monthly means, input of 02a/02b
├── processed/
│   ├── 01_reconstructedTWSA_data.nc            # output of 01, input of 02b
│   ├── 02_RecTWSA_era5.nc                      # training/val/test data (datamodule.ds_path)
│   ├── GRACE_era5.nc                           # GRACE/GRACE-FO observations, --dataset grace and grace_graceclim
│   └── GRACE_climatology_pre2008.nc            # GRACE climatology 2002-04 to 2008-02, --dataset grace_graceclim
├── orderedhydrosheds/
│   ├── hydrosheds_basins.shp                   # basin polygons (+ .dbf/.shx/.cpg), read by the graph build
│   └── hydrosheds_basins.gpkg                  # the same polygons, read by the basin maps in scripts/plotting/
└── Li_csr_fcast/
    ├── CSR_FCast_gridded_initialized_in_<YYYY-MM>.nc  # Li & Kusche (2026) forecast, input of 04_prepareFCast.py
    └── combined.nc                                    # its output, read by compare_models.py
```

`processed/02_RecTWSA_era5.nc` is the file the configs point at. `GRACE_era5.nc` and
`GRACE_climatology_pre2008.nc` are only needed for the evaluation on GRACE, `Li_csr_fcast/combined.nc` only for
the model comparison. These three paths are hardcoded: `GRACE_era5.nc` in
`src/datamodules/TWSA_heterograph_datamodule.py` and `src/datamodules/TWSA_datamodule2D.py`,
`GRACE_climatology_pre2008.nc` in `src/utils/eval_metrics.py`, and `combined.nc` is the default of
`compare_models.py --fcast_file_path`.

---

## Raw Data Sources

### Training Data — GRACE Reconstructed TWSA (GRACE-REC)

The primary training and test target is a long-term TWSA reconstruction derived from GRACE observations and climate reanalysis, covering the period 1979 to present [1]. Download the dataset from:

> https://datadryad.org/dataset/doi:10.5061/dryad.z612jm6bt

### Input Features — ERA5 Monthly Means

Climate input features are taken from the ERA5 monthly averaged reanalysis on single levels [2]. Download the data via the [Copernicus Climate Data Store (CDS)](https://doi.org/10.24381/cds.f17050d7) using the following API request:

```python
import cdsapi

dataset = "reanalysis-era5-single-levels-monthly-means"
years = [str(year) for year in range(1979, 2024)]

request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_temperature",
        "sea_surface_temperature",
        "surface_pressure",
        "total_precipitation",
        "10m_wind_speed",
        "evaporation",
        "potential_evaporation",
        "runoff",
        "land_sea_mask",
    ],
    "year": years,
    "month": ["01", "02", "03", "04", "05", "06",
              "07", "08", "09", "10", "11", "12"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

client = cdsapi.Client()
client.retrieve(dataset, request).download("era5_monthly.nc")
```

The preprocessing scripts read every `.nc` file in `data/raw/era5/era5_1/`, `era5_2/` and `era5_static/`. The data
used in the paper was split over three requests, one per folder, with the static fields (land-sea mask and
geopotential) in `era5_static/`.

### Graph Construction — HydroSHEDS Basin Polygons

Basin delineations for the heterogeneous graph are derived from the HydroSHEDS HydroBASINS dataset [3]. The shapefiles can be downloaded directly from:

> https://www.hydrosheds.org

Alternatively, use the provided script to fetch the data via Google Earth Engine:

```bash
python scripts/preprocess_data/03_get_watersheds.py
```

The script saves the basin polygons to `data/orderedhydrosheds/hydrosheds_basins.shp` and `.gpkg`. The `.shp` path is hardcoded in `src/datamodules/TWSA_heterograph_dataset.py` and is read on every run, including when the precomputed graph in `graphs/` is used.

### Evaluation — CSR GRACE/GRACE-FO RL06.03 Mascons

To evaluate model performance against observed GRACE data, we use the RL06.03 monthly mascon solutions from the Center for Space Research (CSR) [4, 5]. Download from:

> https://www2.csr.utexas.edu/grace/RL06_mascons.html

Download the grid with all corrections applied and set `GRACE_file_path` at the top of
`scripts/preprocess_data/02a_join_graceTWSA_era5.py` to it. The paper uses the version updated on 2025-09-25,
spanning Apr 2002 to Jul 2025. `05_grace_climatology.py` reads the same file through 02a.

### Comparison — Li & Kusche (2026) TWSA Forecast

Model predictions are compared against the observation-driven TWSA forecast of Li & Kusche (2026) [6]. Download from:

> https://doi.pangaea.de/10.1594/PANGAEA.973113

Put the `CSR_FCast_gridded_initialized_in_<YYYY-MM>.nc` files into `data/Li_csr_fcast/` and run
`scripts/preprocess_data/04_prepareFCast.py`.

---

## Preprocessing

Preprocessing scripts are located in `scripts/preprocess_data/`. The input and output paths at the top of each script follow the layout above (except `GRACE_file_path` in 02a, see above); run them from the repository root. See the main `README.md` for the full step-by-step instructions.

---

## References

[1] Li, F., Kusche, J., Chao, N., Wang, Z., and Löcher, A.: Long-Term (1979–Present) Total Water Storage Anomalies Over the Global Land Derived by Reconstructing GRACE Data, *Geophysical Research Letters*, 48, e2021GL093492, https://doi.org/10.1029/2021GL093492, 2021.

[2] Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., and Thépaut, J.-N.: ERA5 monthly averaged data on single levels from 1940 to present, Copernicus Climate Change Service (C3S) Climate Data Store (CDS), https://doi.org/10.24381/cds.f17050d7, 2023.

[3] Lehner, B. and Grill, G.: Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems, *Hydrological Processes*, 27, 2171–2186, https://doi.org/10.1002/hyp.9740, 2013.

[4] Save, H.: CSR GRACE and GRACE-FO RL06 Mascon Solutions v02, https://doi.org/10.15781/cgq9-nh24, 2025.

[5] Save, H., Bettadpur, S., and Tapley, B. D.: High-resolution CSR GRACE RL05 mascons, *Journal of Geophysical Research: Solid Earth*, 121, 7547–7569, https://doi.org/10.1002/2016JB013007, 2016.

[6] Li, F. and Kusche, J.: Observation-Driven Forecast of Global Terrestrial Water Storage and Evaluation for 2010–2024, *Water Resources Research*, 62, e2025WR041710, https://doi.org/10.1029/2025WR041710, 2026.

# TWSA HiGNN-LSTM

**Global Terrestrial Water Storage Forecasting with Hierarchical Graph Neural Networks and LSTMs**

---

## Overview

HiGNN-LSTM is a deep learning model for global Terrestrial Water Storage Anomaly (TWSA) forecasting. It combines a multi-scale heterogeneous graph neural network (GNN) with an LSTM decoder to predict TWSA six months ahead from a twelve-month history of reconstructed GRACE/GRACE-FO observations and ERA5 climate reanalysis fields.

The heterogeneous graph represents the global hydrological system at three scales: 1° land grid cells, basin-level mesh nodes (derived from HydroSHEDS catchments), and ocean clusters.

In the revised version of the model (the one in `configs/`), the target and all ERA5 inputs are anomalies with respect to the monthly climatology of the training period (`datamodule.twsa_anomaly`, `datamodule.era5_anomaly`), and the model predicts the anomaly directly instead of a residual on top of the last input month (`model.residual_from_last: false`). The results are reported as the mean over 8 seeds (42–49), with retrained ablations (no message passing, no ocean nodes, no ERA5 inputs) and anomaly persistence and climatology as references.

A **ConvLSTM2D** model operating directly on the global 1° grid is provided as a spatial-convolutional baseline under identical training and evaluation conditions.

---

## Repository Structure

```
TWSA_HiGNN-LSTM/
├── configs/                             # the saved config.yaml of the seed-42 run of each study
│   ├── hignn_lstm.yaml                  # HiGNN-LSTM (main model)
│   ├── no_mp.yaml                       # ablation: no message passing (model.skip_gnn)
│   ├── no_ocean.yaml                    # ablation: no ocean nodes (graph_variant: no_ocean)
│   ├── no_era5.yaml                     # ablation: TWSA input only
│   └── convlstm.yaml                    # ConvLSTM2D baseline
├── data/                                # Data directory — see data/README.md
├── scripts/
│   ├── preprocess_data/
│   │   ├── 01_read_GRACE-REC.py         # GRACE-REC .mat → NetCDF
│   │   ├── 02a_join_graceTWSA_era5.py   # CSR GRACE/GRACE-FO mascons + ERA5 (evaluation on GRACE)
│   │   ├── 02b_join_GRACE-REC_era5.py   # GRACE-REC + ERA5 (training/test dataset)
│   │   ├── 03_get_watersheds.py         # HydroSHEDS basin polygons via Google Earth Engine
│   │   ├── 04_prepareFCast.py           # Li & Kusche (2026) forecast for the comparison
│   │   └── 05_grace_climatology.py      # GRACE climatology before 2008-03 (C_grace)
│   ├── plotting/
│   │   ├── plot_ablation_rmse.py        # ablation ΔRMSE per lead
│   │   ├── plot_basin_ablation.py       # per-basin ablation maps
│   │   ├── plot_latent_removal.py       # latent removal ΔRMSE per lead
│   │   └── plot_basin_latent_removal.py # per-basin latent removal maps
│   ├── train_HiGNN.py                   # Train HiGNN-LSTM (one seed)
│   ├── train_ConvLSTM2D.py              # Train ConvLSTM2D (one seed)
│   ├── run_seed_study.py                # Train + evaluate HiGNN-LSTM over several seeds
│   ├── seed_study_convlstm.py           # Train + evaluate ConvLSTM2D over several seeds
│   ├── hparam_sweep.py                  # Hyperparameter sweep, HiGNN-LSTM (validation only)
│   ├── hparam_sweep_convlstm.py         # Hyperparameter sweep, ConvLSTM2D (validation only)
│   ├── evaluation_HiGNN.py              # Evaluate one HiGNN-LSTM run (test, val, grace, grace_graceclim)
│   ├── evaluation_ConvLSTM.py           # Evaluate one ConvLSTM2D run
│   ├── compare_models.py                # HiGNN-LSTM vs the Li & Kusche (2026) forecast
│   ├── results_tables.py                # Per-lead results and ablation tables over all seeds
│   ├── basin_ablation.py                # Per-basin ablation table
│   ├── basin_anomaly_skill.py           # Per-basin anomaly correlation and skill scores
│   ├── run_graph_ablation.py            # Graph-parameter ablation (clusters, k_mesh2mesh, k_mesh2grid)
│   ├── run_latent_removal.py            # LSTM input removal on trained models
│   ├── basin_latent_removal.py          # The same, scored per basin
│   └── GNNfeatures_analysis.py          # Gradient-based attribution (not part of the revised results)
├── src/
│   ├── datamodules/
│   │   ├── TWSA_datamodule2D.py         # PyTorch Lightning datamodule (2D grid)
│   │   ├── TWSA_dataset2D.py            # Dataset class for ConvLSTM2D
│   │   ├── TWSA_heterograph_datamodule.py  # Datamodule for the heterogeneous graph
│   │   └── TWSA_heterograph_dataset.py     # Dataset class and graph construction
│   ├── models/
│   │   ├── hierarchical_model.py        # HiGNN-LSTM core architecture
│   │   ├── pl_hmmodel.py                # PyTorch Lightning module (HiGNN-LSTM)
│   │   ├── pl_convlstm.py               # PyTorch Lightning module (ConvLSTM2D)
│   │   └── layers/
│   │       ├── convlstm_model.py        # ConvLSTM cell and stack
│   │       ├── embedder.py              # Node and edge embedding MLPs
│   │       ├── grid2mesh.py             # Grid-to-mesh message passing layer
│   │       ├── mesh2mesh.py             # Mesh-to-mesh message passing layer
│   │       ├── mesh2grid.py             # Mesh-to-grid message passing layer
│   │       ├── output.py                # Output projection layer
│   │       └── mlp.py                   # Shared MLP module
│   ├── losses/
│   │   └── kge.py                       # Kling-Gupta Efficiency (KGE) metric
│   └── utils/
│       ├── eval_metrics.py              # Area-weighted scores and climatologies used by the evaluation
│       └── utils.py                     # Configuration I/O and plotting utilities
├── graphs/                              # Precomputed graphs: graphs/<graph_variant>_<graph_embedding_version>/
│   ├── full_unitsphere_v2/              # main model and No ERA5 / No MP (+ k_mesh2mesh, k_mesh2grid variants)
│   ├── no_ocean_unitsphere_v2/          # No ocean nodes
│   ├── full_unitsphere_v2_oc10/         # 10 ocean clusters (graph-parameter ablation)
│   └── full_unitsphere_v2_oc50/         # 50 ocean clusters (graph-parameter ablation)
├── checkpoints/                         # Model checkpoints — see checkpoints/README.md
├── tests/                               # pytest suite
├── LICENSE
└── requirements.txt
```

---

## Installation

**Requirements:** Python ≥ 3.8, CUDA ≥ 11.0

```bash
git clone <repository-url>
cd TWSA_HiGNN-LSTM
```

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`torch-scatter` and `torch-sparse` are not installed by `requirements.txt` because the wheels
must match your PyTorch and CUDA build:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

The last two entries of `requirements.txt` (`earthengine-api`, `cdsapi`) are only needed to
download the raw data in `scripts/preprocess_data/`, not to train or evaluate.

---

## Data

### Sources

| Dataset | Description | Reference | Download |
|---------|-------------|-----------|----------|
| GRACE-REC | Reconstructed TWSA time series, 1979–present | Li et al. (2021), *GRL* | [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.z612jm6bt) |
| ERA5 | Monthly mean single-level climate fields | Hersbach et al. (2023) | [CDS](https://doi.org/10.24381/cds.f17050d7) |
| CSR GRACE RL06.03 mascons | Observed TWSA for evaluation | Save et al. (2025) | [CSR](https://www2.csr.utexas.edu/grace/RL06_mascons.html) |
| Li et al. (2026) forecast | Observation-driven TWSA forecast for comparison | Li & Kusche (2026), *WRR* | [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.973113) |

GRACE-REC and ERA5 are used for model training and testing. The CSR mascon product and the Li et al. forecast are used for independent evaluation and comparison only.

**ERA5 variables required:** 2m temperature, sea surface temperature, surface pressure, total precipitation, 10m wind speed, evaporation, potential evaporation, runoff, land-sea mask.

See `data/README.md` for the ERA5 CDS API request and the expected directory layout. The preprocessed data used for the revised results is available at [10.5281/zenodo.24135096](https://doi.org/10.5281/zenodo.24135096).

### Preprocessing

All preprocessing scripts are in `scripts/preprocess_data/`. Check the input/output file paths at the top of each script before running. All outputs are NetCDF files on a common 1° × 1° global grid in `data/processed/`.

```bash
python scripts/preprocess_data/01_read_GRACE-REC.py         # → data/processed/01_reconstructedTWSA_data.nc
python scripts/preprocess_data/02b_join_GRACE-REC_era5.py   # → data/processed/02_RecTWSA_era5.nc
python scripts/preprocess_data/02a_join_graceTWSA_era5.py   # → data/processed/GRACE_era5.nc
python scripts/preprocess_data/03_get_watersheds.py         # → data/orderedhydrosheds/hydrosheds_basins.{shp,gpkg}
python scripts/preprocess_data/04_prepareFCast.py           # → data/Li_csr_fcast/combined.nc
python scripts/preprocess_data/05_grace_climatology.py      # → data/processed/GRACE_climatology_pre2008.nc
```

- **01** reads the detrended reconstruction (`variable = 'grace_rec_detrended'`) and coarsens it with area weighting.
- **02b** merges it with ERA5: latitude-weighted coarsening to 1°, linear detrending of the ERA5 fields (not of `lsm` and `z`), temporal alignment. Its output is `datamodule.ds_path` in all configs.
- **02a** does the same for the CSR GRACE/GRACE-FO mascons (detrended over the full record, then cut to 2008-03 onward), with GRACE solutions matched to ERA5 months within ±20 days. Used by `--dataset grace` and `grace_graceclim`.
- **03** is only needed if you do not download the basin polygons yourself; the path is read on every run.
- **04** is only needed for `compare_models.py`.
- **05** builds the GRACE climatology over 2002-04 to 2008-02 (the months before `GRACE_era5.nc` starts), used by `--dataset grace_graceclim` and `results_tables.py`.

---

## Training

All scripts must be run from the repository root with `src/` on the Python path:

```bash
export PYTHONPATH=src/
```

**HiGNN-LSTM (one seed):**
```bash
python scripts/train_HiGNN.py --config_path configs/hignn_lstm.yaml --seed 42
```

**ConvLSTM2D baseline (one seed):**
```bash
python scripts/train_ConvLSTM2D.py --config_path configs/convlstm.yaml --seed 42
```

Training progress is logged via [Weights & Biases](https://wandb.ai). To run without W&B logging, set `wandb.mode: offline` in the configuration file. The best checkpoint (lowest `val_rmse`) and `last.ckpt` are saved in a new run directory under `experiment.exp_dir`.

Key configuration parameters:

| Parameter | Description |
|-----------|-------------|
| `datamodule.ds_path` | Path to the preprocessed NetCDF training file |
| `datamodule.twsa_anomaly` / `era5_anomaly` | Target and TWSA input / ERA5 inputs as anomalies w.r.t. the training-period monthly climatology |
| `datamodule.graph_variant` / `graph_embedding_version` | Select the graph directory `graphs/<graph_variant>_<graph_embedding_version>`. Missing files are recomputed and cached there, which yields a different graph than the released checkpoints were trained on |
| `datamodule.num_ocean_clusters`, `k_mesh2mesh`, `k_mesh2grid` | Graph parameters |
| `datamodule.input_seq_len` | Input sequence length in months |
| `datamodule.output_seq_len` | Forecast horizon in months |
| `model.hidden_dim` | Hidden representation size |
| `model.skip_gnn` | Skip mesh → mesh and mesh → grid message passing (No MP ablation) |
| `model.residual_from_last` | Predict a residual on top of the last input month (`false` in the revised model) |
| `pl.max_epochs` | Maximum number of training epochs |
| `pl.devices` | GPU device IDs (list) |

The `experiment.save_dir` entry in `configs/` is the directory of the original seed-42 run; the training and seed-study scripts overwrite it (and the seed).

---

## Evaluation

The evaluation scripts expect a directory containing a `config.yaml` and exactly one best checkpoint (a `last.ckpt` next to it is ignored).

```bash
# Evaluate HiGNN-LSTM
python scripts/evaluation_HiGNN.py --exp_path <run_dir> --dataset test

# Evaluate ConvLSTM2D
python scripts/evaluation_ConvLSTM.py --exp_path <run_dir> --dataset test

# Compare HiGNN-LSTM to the forecast of Li & Kusche (2026)
python scripts/compare_models.py --exp_path <run_dir> --fcast_file_path data/Li_csr_fcast/combined.nc
```

`--dataset` selects the evaluation period: `test`, `val`, `grace` (CSR mascon observations, with the reconstruction's training climatology) or `grace_graceclim` (CSR mascons, with the GRACE climatology from `05_grace_climatology.py`). `--gpu` selects the device and `--scores_only` (HiGNN only) skips the figures and `predictions.pt`.
Outputs go to `<experiment.save_dir>/eval_on_<dataset>/` (`<experiment.save_dir>/compare_to_fcast/` for `compare_models.py`), where `experiment.save_dir` is read from the run's `config.yaml`. They include per-lead scores (`per_timestep_scores.csv`), basin-wide summaries and figures.

Predictions and the climatology are cached as `predictions.pt` and `climatology.pt` inside the
output directory and reused on subsequent runs — delete them to force recomputation.

---

## Reproducing results

Run from the repository root with `export PYTHONPATH=src/`. The analysis scripts find the seed studies by the
`experiment.exp_dir` paths in `configs/` (see `MODELS` in `scripts/results_tables.py`), so keep those paths.
The hyperparameters in `configs/` were chosen with `hparam_sweep.py` and `hparam_sweep_convlstm.py` on the
validation period; the test period was not used for tuning.

1. **Preprocessing** — the six steps above (or download the preprocessed data and run only 02a and 05; 02a also needs the raw ERA5 files).
2. **Seed studies** (8 seeds each):
   ```bash
   python scripts/run_seed_study.py --config_path configs/hignn_lstm.yaml --seeds 42 43 44 45 46 47 48 49 --datasets val test grace
   python scripts/run_seed_study.py --config_path configs/no_mp.yaml      --seeds 42 43 44 45 46 47 48 49
   python scripts/run_seed_study.py --config_path configs/no_ocean.yaml   --seeds 42 43 44 45 46 47 48 49
   python scripts/run_seed_study.py --config_path configs/no_era5.yaml    --seeds 42 43 44 45 46 47 48 49
   python scripts/seed_study_convlstm.py --config_path configs/convlstm.yaml --seeds 42 43 44 45 46 47 48 49 --datasets val test grace
   ```
   Then, for every seed run of HiGNN-LSTM and ConvLSTM2D, `evaluation_HiGNN.py` / `evaluation_ConvLSTM.py`
   with `--dataset grace_graceclim`.
3. **Tables** — `python scripts/results_tables.py` (per-lead results and ablation tables, written to
   `experiments/HiGNN/tables_02/`), then `python scripts/basin_ablation.py` and
   `python scripts/basin_anomaly_skill.py` (and `--dataset grace`).
4. **Graph-parameter ablation** — `python scripts/run_graph_ablation.py` (4 seeds per variant).
5. **Latent removal** — `python scripts/run_latent_removal.py`, then `python scripts/basin_latent_removal.py --seed 45`.
6. **Model comparison** — `python scripts/compare_models.py --exp_path <seed-45 run of HiGNN-LSTM>`; the figures of
   `evaluation_HiGNN.py` come from the same evaluation runs as step 2.
7. **Figures** — the four scripts in `scripts/plotting/`, after the steps whose output they read.

---

## Tests

```bash
PYTHONPATH=src/ python -m pytest tests/
```

Several tests read the preprocessed files in `data/` and the graphs in `graphs/`.

---

## Model Architecture

### HiGNN-LSTM

The model processes each time step through a four-stage hierarchical encoder before passing the resulting sequence to an LSTM decoder:

1. **Embedder** — independent MLPs project grid, mesh, and ocean node features into a shared latent space.
2. **Grid → Mesh** — message passing aggregates 1° land cell features into basin-level mesh nodes.
3. **Mesh → Mesh** — message passing propagates information across basin nodes and ocean clusters. Timestep embeddings (FiLM conditioning) allow the network to differentiate forecast steps during autoregressive decoding.
4. **Mesh → Grid** — message passing distributes basin-scale representations back to the 1° grid.
5. **LSTM decoder** — receives the sequence of per-timestep GNN outputs and the TWSA history and predicts the TWSA anomaly of each lead (with `residual_from_last: true`, a residual on top of the last input month instead).

With `model.skip_gnn: true` (No MP ablation) stages 3 and 4 are skipped.

### ConvLSTM2D

A stack of convolutional LSTM cells operating on the full 1° global grid. Each cell applies 3×3 convolutions within its gating mechanism, allowing the model to capture spatially local dependencies.

---

## Acknowledgements

- GRACE and GRACE-FO mission teams (NASA/DLR)
- ERA5 reanalysis provided by the Copernicus Climate Change Service
- PyTorch Lightning and PyTorch Geometric open-source communities
- The implementation of the hierarchical GNN is deduced from https://github.com/corentin-dfg/graph4sen2dwater: "Forecasting water resources from satellite image time series using a graph-based learning strategy" by Corentin Dufourg, Charlotte Pelletier, Stéphane May and Sébastien Lefèvre, at ISPRS Technical Commission II Symposium 2024
- Claude Code was used to assist in implementing and revising data processing, analysis pipelines and visualisation. All scientific content, experimental design, analysis, and interpretation are the responsibility of the authors. 

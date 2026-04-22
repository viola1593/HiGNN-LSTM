# TWSA HiGNN-LSTM

**Global Terrestrial Water Storage Forecasting with Hierarchical Graph Neural Networks and LSTMs**

---

## Overview

HiGNN-LSTM is a deep learning model for global Terrestrial Water Storage Anomaly (TWSA) forecasting. It combines a multi-scale heterogeneous graph neural network (GNN) with an LSTM decoder to predict TWSA six months ahead from a twelve-month history of reconstructed GRACE/GRACE-FO observations and ERA5 climate reanalysis fields.

The heterogeneous graph represents the global hydrological system at three scales: 1° land grid cells, basin-level mesh nodes (derived from HydroSHEDS catchments), and ocean clusters. This enables the model to capture both local land–atmosphere interactions and long-range teleconnections across river basins.

A **ConvLSTM2D** model operating directly on the global 1° grid is provided as a spatial-convolutional baseline under identical training and evaluation conditions.

---

## Repository Structure

```
TWSA_HiGNN-LSTM/
├── configs/
│   ├── config2D.yaml                    # ConvLSTM2D configuration
│   └── config_heterograph.yaml          # HiGNN-LSTM configuration
├── data/                                # Data directory — see data/README.md
├── scripts/
│   ├── preprocess_data/
│   │   ├── read_GRACE-REC.py            # Convert GRACE-REC .mat files to NetCDF
│   │   ├── join_graceTWSA_era5.py       # Align GRACE TWSA with ERA5, coarsen to 1°
│   │   ├── join_GRACE-REC_era5.py       # Same pipeline for reconstructed GRACE data
│   │   └── prepareFCast.py             # Prepare external forecast data for evaluation
│   ├── train_HiGNN.py                   # Train HiGNN-LSTM
│   ├── train_ConvLSTM2D.py              # Train ConvLSTM2D baseline
│   ├── evaluation_HiGNN.py              # Evaluate HiGNN-LSTM on the test set
│   ├── evaluation_ConvLSTM.py           # Evaluate ConvLSTM2D on the test set
│   ├── compare_models.py                # Side-by-side model comparison
│   └── GNNfeatures_analysis.py          # SHAP-based feature importance analysis
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
│       └── utils.py                     # Configuration I/O and plotting utilities
├── checkpoints/                         # Model checkpoints
├── requirements.txt
└── setup.py
```

---

## Installation

**Requirements:** Python ≥ 3.8, CUDA ≥ 11.0

```bash
git clone <repository-url>
cd TWSA_HiGNN-LSTM
```

Using conda (recommended):
```bash
conda env create -f environment.yml
conda activate twsa-hignn
```

Using pip:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

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

See `data/README.md` for the ERA5 CDS API request and the expected directory layout.

### Preprocessing

All preprocessing scripts are in `scripts/preprocess_data/`. Update the input/output file paths at the top of each script before running. All outputs are NetCDF files on a common 1° × 1° global grid. The final preprocessed data used for training and evaluating the model is also available at [here](https://doi.org/10.5281/zenodo.19664592).

**Step 1 — Convert GRACE-REC from `.mat` to NetCDF:**
```bash
python scripts/preprocess_data/read_GRACE-REC.py
```
Applies area-weighted spatial coarsening and optional linear detrending.

**Step 2 — Merge GRACE-REC with ERA5 (main training dataset):**
```bash
python scripts/preprocess_data/join_GRACE-REC_era5.py
```
Performs latitude-weighted coarsening to 1° and temporal alignment with a ±20-day tolerance.

**Step 3 — Merge observed GRACE TWSA with ERA5 (evaluation dataset):**
```bash
python scripts/preprocess_data/join_graceTWSA_era5.py
```

**Step 4 — Prepare the Li et al. forecast for comparison (optional):**
```bash
python scripts/preprocess_data/prepareFCast.py
```

The output file paths are referenced in the configuration files via `datamodule.ds_path`.

---

## Training

All scripts must be run from the repository root with `src/` on the Python path:

```bash
export PYTHONPATH=src/
```

**HiGNN-LSTM:**
```bash
python scripts/train_HiGNN.py --config configs/config_heterograph.yaml
```

**ConvLSTM2D baseline:**
```bash
python scripts/train_ConvLSTM2D.py --config configs/config2D.yaml
```

Training progress is logged via [Weights & Biases](https://wandb.ai). To run without W&B logging, set `wandb.mode: offline` in the configuration file.

Key configuration parameters:

| Parameter | Description |
|-----------|-------------|
| `datamodule.ds_path` | Path to the preprocessed NetCDF training file |
| `datamodule.input_seq_len` | Input sequence length in months |
| `datamodule.output_seq_len` | Forecast horizon in months |
| `model.hidden_dim` | Hidden representation size |
| `pl.max_epochs` | Maximum number of training epochs |
| `pl.devices` | GPU device IDs (list) |

---

## Evaluation

The evaluation scripts expect a directory containing a `config.yaml` and at least one `.ckpt` checkpoint file (as saved by PyTorch Lightning).

```bash
# Evaluate HiGNN-LSTM
python scripts/evaluation_HiGNN.py --exp_path <experiment_dir> 

# Evaluate ConvLSTM2D
python scripts/evaluation_ConvLSTM.py --exp_path <experiment_dir> 

# Compare HiGNN-LSTM to FCast from Li et al.
python scripts/compare_models.py
```

Outputs are written to `eval_on_test/<experiment_dir>/` and include per-timestep RMSE, MAPE, SMAPE, and explained variance scores as well as JSON summaries.

---

## Feature Importance Analysis

Input feature attribution is computed using the Saliency and NoiseTunnel methods from [Captum](https://captum.ai/):

```bash
python scripts/GNNfeatures_analysis.py --exp_path <experiment_dir> 
```

---

## Model Architecture

### HiGNN-LSTM

The model processes each time step through a four-stage hierarchical encoder before passing the resulting sequence to an LSTM decoder:

1. **Embedder** — independent MLPs project grid, mesh, and ocean node features into a shared latent space.
2. **Grid → Mesh** — message passing aggregates 1° land cell features into basin-level mesh nodes.
3. **Mesh → Mesh** — message passing propagates information across basin nodes and ocean clusters. Timestep embeddings (FiLM conditioning) allow the network to differentiate forecast steps during autoregressive decoding.
4. **Mesh → Grid** — message passing distributes basin-scale representations back to the 1° grid.
5. **LSTM decoder** — receives the sequence of per-timestep GNN outputs and predicts additive residuals relative to the most recent observed TWSA map.

### ConvLSTM2D

A stack of convolutional LSTM cells operating on the full 1° global grid. Each cell applies 3×3 convolutions within its gating mechanism, allowing the model to capture spatially local dependencies. 

---





## Acknowledgements

- GRACE and GRACE-FO mission teams (NASA/DLR)
- ERA5 reanalysis provided by the Copernicus Climate Change Service
- PyTorch Lightning and PyTorch Geometric open-source communities
- The implementation of the hierarchical GNN is deduced from https://github.com/corentin-dfg/graph4sen2dwater: "Forecasting water resources from satellite image time series using a graph-based learning strategy" by Corentin Dufourg, Charlotte Pelletier, Stéphane May and Sébastien Lefèvre, at ISPRS Technical Commission II Symposium 2024

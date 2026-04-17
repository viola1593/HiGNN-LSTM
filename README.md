# TWSA HiGNN-LSTM: Global Water Storage Prediction with Heterogeneous Graph Neural Networks and LSTMs

This repository contains the code and instructions for training and evaluating the HiGNN-LSTM model for global Terrestrial Water Storage Anomaly (TWSA) prediction, along with baseline comparisons.

## Overview

The HiGNN-LSTM model combines heterogeneous graph neural networks with LSTM layers to predict global water storage variations from satellite data, climate reanalysis, and geographic features.

**Key Features:**
- Heterogeneous graph neural network architecture for capturing multi-scale water cycle processes
- LSTM decoder for temporal prediction
- Comparison with ConvLSTM2D baseline
- Model interpretation through SHAP analysis
- Evaluation on GRACE and synthetic data

## Installation

### Prerequisites
- Python >= 3.8
- CUDA >= 11.0 (for GPU support)
- conda (recommended) or pip

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd TWSA_HiGNN-LSTM
```

2. **Create a Python environment:**

Using conda:
```bash
conda env create -f environment.yml
conda activate twsa-hignn
```

Or using pip with venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up data directory:**

See [data/README.md](data/README.md) for instructions on data download and preparation.

## Project Structure

```
TWSA_HiGNN-LSTM/
├── configs/              # Configuration files for models
├── data/                 # Data directory (see data/README.md)
├── src/                  # Source code
│   ├── datamodules/      # Data loading and preprocessing
│   ├── models/           # Model definitions
│   ├── losses/           # Loss functions
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utility functions
├── scripts/              # Executable scripts
├── checkpoints/          # Pre-trained model checkpoints
└── notebooks/            # Jupyter notebooks (tutorials)
```

## Quick Start

### 1. Training Models

**Train HiGNN-LSTM model:**
```bash
python scripts/train_HiGNN.py --config configs/config_heterograph.yaml
```

**Train ConvLSTM2D baseline:**
```bash
python scripts/train_ConvLSTM2D.py --config configs/config2D.yaml
```

### 2. Evaluating Models

**Evaluate HiGNN-LSTM:**
```bash
python scripts/evaluation_HiGNN.py --exp_path <path-to-checkpoint> --dataset test
```

**Evaluate ConvLSTM2D:**
```bash
python scripts/evaluation_ConvLSTM.py --exp_path <path-to-checkpoint> --dataset test
```

### 3. Model Comparison

**Compare HiGNN-LSTM vs ConvLSTM2D:**
```bash
python scripts/compare_models.py
```

### 4. SHAP Analysis

**Generate SHAP-based model interpretability analysis:**
```bash
python scripts/shap_analysis.py
```

## Configuration

Models are configured using YAML files in the `configs/` directory:

- `config_heterograph.yaml` - HiGNN-LSTM configuration (recommended)
- `config2D.yaml` - ConvLSTM2D configuration
- `config_graphGRU.yaml` - Alternative GRU-based graph model

Key parameters:
- `datamodule.input_seq_len` - Input sequence length (months)
- `datamodule.output_seq_len` - Forecast horizon (months)
- `model.hidden_dim` - Hidden layer dimensions
- `model.num_layers` - Number of stacked layers
- `pl.trainer.max_epochs` - Training epochs

## Data

The model requires:
- **GRACE/GRACE-FO satellite data**: Terrestrial Water Storage Anomalies
- **Climate reanalysis (ERA5)**: Precipitation and other climate variables
- **Geographic data**: Land-sea mask, elevation, climate zones

See [data/README.md](data/README.md) for detailed setup instructions.

## Model Checkpoints

Pre-trained checkpoints can be placed in `checkpoints/pretrained/`:
- `pretrained/hignn_model.ckpt` - HiGNN-LSTM checkpoint
- `pretrained/convlstm2d_model.ckpt` - ConvLSTM2D checkpoint

## Experiment Details

For detailed information about experiments, configurations, and hyperparameters used in the manuscript, see [EXPERIMENTS.md](EXPERIMENTS.md).

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your2024manuscript,
  title={Your Paper Title},
  author={Your Names},
  journal={Journal Name},
  year={2024}
}
```

## License

[Add license information]

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or issues, please contact: [your-email@domain.com]

## Acknowledgments

- GRACE/GRACE-FO mission (NASA/DLR)
- ERA5 reanalysis from Copernicus Climate Data Store
- PyTorch Lightning and PyTorch Geometric communities

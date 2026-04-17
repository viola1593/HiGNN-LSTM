# Experiments and Results

This document details the experiments, configurations, and hyperparameters used in the manuscript.

## Base Configuration Parameters

### Common Settings (All Models)
- **Input sequence length**: 12 months
- **Output sequence length**: 6 months (forecast horizon)
- **Training period**: [To be specified]
- **Validation period**: [To be specified]
- **Test period**: [To be specified]
- **Batch size**: 32
- **Loss function**: KGE (Kling-Gupta Efficiency)

### Data Preprocessing
- **Spatial resolution**: 1° x 1° global grid
- **Target variable**: TWSA (Terrestrial Water Storage Anomalies) from GRACE/GRACE-FO
- **Input features**: 
  - Climate variables from ERA5 (precipitation, temperature, etc.)
  - Static geographic features (elevation, climate zones)
  - Temporal encodings (month of year)
- **Normalization**: Standardization (zero-mean, unit-variance) per variable

---

## HiGNN-LSTM Model

### Configuration File
`configs/config_heterograph.yaml`

### Architecture Parameters
```yaml
model:
  hidden_dim: 128
  num_layers: [specify]
  num_heads: [specify]  # Multi-head attention (if applicable)
  dropout: 0.2
  
datamodule:
  input_seq_len: 12
  output_seq_len: 6
  batch_size: 32
```

### Training Configuration
```yaml
optimizer:
  type: Adam
  lr: 0.001
  weight_decay: 1e-5

scheduler:
  type: ReduceLROnPlateau
  patience: 10
  factor: 0.5

pl:
  trainer:
    max_epochs: 100
    devices: [0]  # GPU device ID
    precision: 32
    log_every_n_steps: 10
```

### Training Results
- **Best validation loss**: [To be filled]
- **Training time**: [To be filled]
- **Hardware**: [GPU model, memory]

### Command to Train
```bash
python scripts/train_HiGNN.py --config configs/config_heterograph.yaml
```

### Evaluation Results
- **Test RMSE**: [To be filled]
- **Test NSE**: [To be filled]
- **Test Correlation**: [To be filled]

### Command to Evaluate
```bash
python scripts/evaluation_HiGNN.py --exp_path checkpoints/pretrained/hignn_model.ckpt --dataset test
```

---

## ConvLSTM2D Baseline

### Configuration File
`configs/config2D.yaml`

### Architecture Parameters
```yaml
model:
  hidden_dim: 64
  num_layers: [specify]
  kernel_size: 3
  dropout: 0.2
  
datamodule:
  input_seq_len: 12
  output_seq_len: 6
  batch_size: 32
```

### Training Configuration
```yaml
optimizer:
  type: Adam
  lr: 0.001
  weight_decay: 1e-5

pl:
  trainer:
    max_epochs: 100
    devices: [0]
    precision: 32
```

### Command to Train
```bash
python scripts/train_ConvLSTM2D.py --config configs/config2D.yaml
```

### Evaluation Results
- **Test RMSE**: [To be filled]
- **Test NSE**: [To be filled]
- **Test Correlation**: [To be filled]

### Command to Evaluate
```bash
python scripts/evaluation_ConvLSTM.py --exp_path checkpoints/pretrained/convlstm2d_model.ckpt --dataset test
```

---

## Model Comparison

### Command
```bash
python scripts/compare_models.py
```

### Key Metrics
| Metric | HiGNN-LSTM | ConvLSTM2D | Improvement |
|--------|-----------|-----------|-------------|
| RMSE (cm) | [value] | [value] | [%] |
| NSE | [value] | [value] | [%] |
| Correlation | [value] | [value] | [%] |
| Training Time | [hours] | [hours] | - |

---

## SHAP Analysis

### Purpose
Model interpretation and feature importance analysis using SHAP (SHapley Additive exPlanations) values.

### Command
```bash
python scripts/shap_analysis.py
```

### Key Findings
[To be filled with SHAP analysis results]

---

## Data & Environment

### Datasets Used
- **GRACE/GRACE-FO**: [Version, DOI]
- **ERA5**: [Variables, period]
- **Static data**: [Sources]

### Computing Environment
- **OS**: Linux/macOS/Windows
- **Python**: 3.9+
- **Framework versions**:
  - PyTorch Lightning: 1.7+
  - PyTorch Geometric: 2.0+
  - CUDA: 11.8 (for GPU)

### Hardware
- **Training GPU**: [Model, VRAM]
- **RAM**: [Amount]
- **Storage**: [Requirements]

---

## Reproducibility

To reproduce the exact results from the manuscript:

1. Download/prepare data as per [data/README.md](data/README.md)
2. Download pre-trained checkpoints from [URL to be added]
3. Run evaluation scripts:
   ```bash
   python scripts/evaluation_HiGNN.py --exp_path checkpoints/pretrained/hignn_model.ckpt --dataset test
   python scripts/evaluation_ConvLSTM.py --exp_path checkpoints/pretrained/convlstm2d_model.ckpt --dataset test
   ```

### Note on Random Seeds
All experiments use `seed_everything(42)` in PyTorch Lightning for reproducibility.

---

## Additional Analyses

### Cross-Validation
[Describe CV strategy if used]

### Uncertainty Quantification
[Describe any uncertainty estimates]

### Ablation Studies
[Describe any ablation study results]

---

## Paper Figures

The following figures are generated during evaluation:

| Figure | Script | Output |
|--------|--------|--------|
| Time series predictions | `evaluation_HiGNN.py` | `BasinPlots/` |
| NSE/KGE maps | `evaluation_HiGNN.py` | `NSE/KGE/` |
| Model comparison | `compare_models.py` | - |
| SHAP analysis | `shap_analysis.py` | - |

---

## Troubleshooting

### Common Issues

1. **Out of memory during training**
   - Reduce batch size
   - Use gradient accumulation
   - See model config for memory-efficient options

2. **Data loading issues**
   - Verify data directory structure
   - Check file formats and paths
   - See data/README.md

3. **Checkpoint loading errors**
   - Ensure PyTorch versions match
   - Check checkpoint path validity

---

## Questions & Support

For questions about experimental details, please refer to the main manuscript or contact the authors.

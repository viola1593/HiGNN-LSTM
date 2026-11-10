# Model Checkpoints

This directory holds the trained PyTorch Lightning checkpoints of every run behind the revised results:
HiGNN-LSTM, its three ablations and the ConvLSTM2D baseline with all 8 seeds (42–49) each, and the six variants of
the graph-parameter ablation with 4 seeds (42–45) each, 64 runs in total. Each `seed<N>/` directory
contains the best checkpoint of that run (lowest `val_rmse`) and the `config.yaml` it was trained with. The
evaluation scripts read both, so each seed directory is self-contained. The `config.yaml` of seed 42 of the five
main studies is identical to the file in `configs/` (see the table below); the other seeds differ only in `seed` and
`experiment.save_dir`.

```
checkpoints/
├── HiGNN_LSTM/
│   ├── seed42/
│   │   ├── config.yaml
│   │   └── epoch=26-step=567.ckpt
│   ├── seed43/
│   └── ...                      (seed42 – seed49)
├── Ablations/
│   ├── no_mp/                   (seed42 – seed49)
│   ├── no_ocean/                (seed42 – seed49)
│   ├── no_era5/                 (seed42 – seed49)
│   └── graph_ablation/
│       ├── oc10/                (seed42 – seed45)
│       ├── oc50/
│       ├── m2m5/
│       ├── m2m20/
│       ├── m2g1/
│       └── m2g5/
├── ConvLSTM/                    (seed42 – seed49)
└── README.md (this file)
```

| Directory | Study | Config of seed 42 |
|-----------|-------|-------------------|
| `HiGNN_LSTM/` | HiGNN-LSTM | `configs/hignn_lstm.yaml` |
| `Ablations/no_mp/` | No MP | `configs/no_mp.yaml` |
| `Ablations/no_ocean/` | No ocean nodes | `configs/no_ocean.yaml` |
| `Ablations/no_era5/` | No ERA5 | `configs/no_era5.yaml` |
| `ConvLSTM/` | ConvLSTM2D | `configs/convlstm.yaml` |
| `Ablations/graph_ablation/<variant>/` | Graph-parameter ablation | `configs/hignn_lstm.yaml` with one datamodule change (see below) |


`scripts/run_seed_study.py`, `scripts/seed_study_convlstm.py` and `scripts/run_graph_ablation.py` retrain the checkpoints.

---

## HiGNN-LSTM

**Architecture** (`configs/hignn_lstm.yaml` → `model`)

| Parameter | Value |
|-----------|-------|
| `hidden_dim` | 32 |
| `num_processing_gnns` | 1 |
| `gnn_embedding_dim` | 4 |
| `rnn_layers` / `rnn_hidden_size` | 1 / 32 |
| `rnn_dropout` | 0.4 |
| `timestep_embed_dim` | 2 |
| `residual_from_last` | `false` (the anomaly is predicted directly) |

**Data** — input sequence 12 months, forecast horizon 6 months, `delay` 0, chronological
70/20/10 train/val/test split of `data/processed/02_RecTWSA_era5.nc`. Nine input
variables: `twsa`, `sst`, `sp`, `t2m`, `si10`, `tp`, `e`, `ro`, `pev`. With `twsa_anomaly` and `era5_anomaly`
the target and all inputs are anomalies with respect to the monthly climatology of the training period.

**Graph** — `graphs/full_unitsphere_v2` (`graph_variant: full`, `graph_embedding_version: unitsphere_v2`),
20 ocean clusters, `k_mesh2mesh` 10, `k_mesh2grid` 3. The checkpoint is tied to this graph: regenerating it changes
the node and edge ordering the weights were trained on.

**Training** — AdamW (`lr` 2e-3, `weight_decay` 1e-2) with a cosine schedule over `max_epochs` (down to 1 % of
`lr`), latitude-weighted MAE loss, batch size 8 with `accumulate_grad_batches` 2 (effective 16), `bf16-mixed`
precision, `deterministic: True`, seeds 42–49, 50 epochs. The released checkpoint of each seed is the one selected
by `ModelCheckpoint` on the lowest `val_rmse` (epoch 26 for seed 42).

## Ablations

Each ablation changes one thing in `configs/hignn_lstm.yaml` and is otherwise trained the same way:

| Model | Config | Change | Checkpoint (seed 42) |
|-------|--------|--------|------------|
| No MP | `configs/no_mp.yaml` | `model.skip_gnn: true` (no mesh → mesh and mesh → grid message passing) | epoch 28 |
| No ocean nodes | `configs/no_ocean.yaml` | `datamodule.graph_variant: no_ocean` (graph `graphs/no_ocean_unitsphere_v2`) | epoch 17 |
| No ERA5 | `configs/no_era5.yaml` | `input_vars: [twsa]`, `era5_anomaly: false` | epoch 37 |

### Graph-parameter ablation

Each variant changes one datamodule entry of `configs/hignn_lstm.yaml` (set by `scripts/run_graph_ablation.py`)
and was trained with seeds 42–45:

| Directory | Change | Graph |
|-----------|--------|-------|
| `oc10/` | `num_ocean_clusters: 10` | `graphs/full_unitsphere_v2_oc10` |
| `oc50/` | `num_ocean_clusters: 50` | `graphs/full_unitsphere_v2_oc50` |
| `m2m5/` | `k_mesh2mesh: 5` | `graphs/full_unitsphere_v2` |
| `m2m20/` | `k_mesh2mesh: 20` | `graphs/full_unitsphere_v2` |
| `m2g1/` | `k_mesh2grid: 1` | `graphs/full_unitsphere_v2` |
| `m2g5/` | `k_mesh2grid: 5` | `graphs/full_unitsphere_v2` |

## ConvLSTM2D (baseline)

**Architecture** (`configs/convlstm.yaml` → `model`)

| Parameter | Value |
|-----------|-------|
| `input_dim` | 12 |
| `hidden_dim` | 32 |
| `num_layers` | 1 |
| `residual_from_last` | `false` |
| kernel size | 3 × 3 (set in `scripts/train_ConvLSTM2D.py` and `scripts/evaluation_ConvLSTM.py`) |

**Data** — same file, split, sequence length, horizon and anomalies as above, plus the land-sea mask `lsm`
as a tenth input variable.

**Training** — AdamW (`lr` 2e-3, `weight_decay` 1e-2) with a cosine schedule, batch size 8 with
`accumulate_grad_batches` 2, `bf16-mixed` precision, `deterministic: True`, seeds 42–49, 50 epochs. The released
checkpoint of seed 42 is epoch 19.

---

## Using a checkpoint

Run the evaluation scripts from the repository root with `src/` on the Python path — they locate
the `.ckpt` and its `config.yaml` themselves:

```bash
export PYTHONPATH=src/

python scripts/evaluation_HiGNN.py --exp_path checkpoints/HiGNN_LSTM/seed42 --dataset test --gpu 0
python scripts/evaluation_ConvLSTM.py --exp_path checkpoints/ConvLSTM/seed42 --dataset test --gpu 0
```

The outputs are written to `eval_on_<dataset>/` under `experiment.save_dir` of the `config.yaml`, i.e. into the
`experiments/` path of the original run, not into `checkpoints/`. Edit `save_dir` to write them elsewhere.
The table and ablation scripts (`results_tables.py`, `run_graph_ablation.py`, `basin_ablation.py`, ...) find the
runs through the seed-study manifests under `experiments/`, not through this directory.

A directory must contain exactly one checkpoint besides an optional `last.ckpt`; the scripts load that one.

To load a model manually:

```python
# with PYTHONPATH=src/
from models.pl_hmmodel import HMModule
from datamodules.TWSA_heterograph_datamodule import TWSA_HeteroGraphDatamodule
from utils.utils import read_config

config = read_config("checkpoints/HiGNN_LSTM/seed42/config.yaml")
datamodule = TWSA_HeteroGraphDatamodule(**config["datamodule"])
datamodule.setup(stage="fit")

model = HMModule.load_from_checkpoint(
    "checkpoints/HiGNN_LSTM/seed42/epoch=26-step=567.ckpt",
    **config["model"],
    num_input_features=datamodule.num_features,
    num_ocean_features=datamodule.num_ocean_features,
    use_ocean_nodes=datamodule.use_ocean_nodes,
    sequence_length=datamodule.input_seq_len,
    output_seq_len=datamodule.target_seq_len,
    land_sea_mask=None,
)
model.eval()
```

The datamodule has to be constructed first: the number of input and ocean features and the
sequence lengths are derived from the dataset, not stored in the checkpoint.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'models'`** — the packages live under `src/`, so run from
the repository root with `export PYTHONPATH=src/`.

**`AssertionError: expected one best-val_rmse ckpt in ...`** — `--exp_path` must point at the directory
containing the checkpoint (e.g. `checkpoints/HiGNN_LSTM/seed42`), not at the `.ckpt` file itself, and the directory may hold
only one checkpoint besides `last.ckpt`.

**Size mismatch when loading `state_dict`** — the `config.yaml` next to the checkpoint does not
match it, or the graph in `graphs/<graph_variant>_<graph_embedding_version>` differs from the one the model was
trained on.

**CUDA out of memory** — reduce `datamodule.batch_size`, or pass a different `--gpu`.

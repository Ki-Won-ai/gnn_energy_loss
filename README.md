# onboard-gnn

Simplified ML training pipeline for new engineers - GNN baselines on foundation-research patterns.

## Overview

This repository provides a minimal but complete ML training pipeline based on foundation-research patterns. It supports two GNN models (SimpleGCN and SimpleGATv2) across multiple benchmark tasks.

**Goal**: Learn the core ML pipeline patterns without cognitive overload.

## Available Tasks

| Task | Description | Status |
|------|-------------|--------|
| **DeepJEB** | Von Mises stress prediction | ✅ Complete |
| **HDWIA** | Heat diffusion / temperature prediction | 🔄 Placeholder (data prep TBD) |

**Models**: SimpleGCN and SimpleGATv2 (both tasks)

## Quick Start

```bash
# Install dependencies
make install

# DeepJEB Task
make deepjeb-train-mini    # Quick sanity check (5 epochs)
make deepjeb-train         # Full training with GATv2 (default)
make deepjeb-train-gcn     # Train with SimpleGCN

# HDWIA Task (placeholder - data prep not yet implemented)
make hdwia-train-mini      # Quick sanity check
make hdwia-train           # Full training
```

## Make Commands

### DeepJEB Task

| Command | Description |
|---------|-------------|
| `make deepjeb-prepare` | Prepare dataset (H5 -> NPZ) |
| `make deepjeb-train` | Train with GATv2 (default) |
| `make deepjeb-train-mini` | Quick sanity check (5 epochs) |
| `make deepjeb-train-gcn` | Train with SimpleGCN |
| `make deepjeb-predict CKPT=...` | Run predictions |
| `make deepjeb-visualize RESULTS=...` | Generate plots |

### HDWIA Task

| Command | Description |
|---------|-------------|
| `make hdwia-prepare` | Prepare dataset (placeholder) |
| `make hdwia-train` | Train with GATv2 (default) |
| `make hdwia-train-mini` | Quick sanity check (5 epochs) |
| `make hdwia-train-gcn` | Train with SimpleGCN |
| `make hdwia-predict CKPT=...` | Run predictions |
| `make hdwia-visualize RESULTS=...` | Generate plots |

### Development

| Command | Description |
|---------|-------------|
| `make test` | Run all tests with pytest |
| `make test-v` | Run tests with verbose output (shows each test name) |
| `make test-fast` | Run tests excluding `@pytest.mark.slow` tests |
| `make lint` | Check code style with ruff |
| `make format` | Auto-format code with ruff |

### Utilities

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies (`uv sync`) |
| `make clean` | Remove `outputs/`, `__pycache__/`, `.pytest_cache/` |
| `make help` | Show all available commands |

## Project Structure

```
onboard-gnn/
├── shared/                      # Reusable utilities
│   ├── npz_dataset.py          # NPZ data loader
│   ├── npz_datamodule.py       # Lightning DataModule
│   ├── collators.py            # Data collation (SimpleCollator)
│   ├── io_manager.py           # Normalization handling
│   ├── metrics.py              # Loss functions
│   └── visualization.py        # 3D comparison plots
│
├── models/                      # GNN architectures
│   ├── simple_gcn.py           # SimpleGCN baseline
│   ├── simple_gatv2.py         # SimpleGATv2 baseline
│   └── builder.py              # Model factory
│
├── tasks/deepjeb/              # DeepJEB benchmark task ✅
│   ├── __main__.py             # CLI entry (Click + Hydra)
│   ├── datamodule.py           # Task DataModule
│   ├── model.py                # LightningModule wrapper
│   ├── training.py             # Training orchestration
│   ├── preparation.py          # Data preparation
│   ├── predict.py              # Prediction pipeline
│   ├── visualize.py            # Visualization from results
│   └── configs/                # Hydra configs
│       ├── train.yaml          # Main training config
│       ├── train-mini.yaml     # Quick sanity check
│       ├── prepare.yaml        # Preparation config
│       ├── predict.yaml        # Prediction config
│       ├── visualize.yaml      # Visualization config
│       ├── model/
│       │   ├── gcn.yaml        # SimpleGCN config
│       │   └── gatv2.yaml      # SimpleGATv2 config (default)
│       └── dataset/
│           ├── base.yaml       # Full dataset
│           └── small.yaml      # Subset for quick tests
│
└── tasks/hdwia/                # HDWIA benchmark task 🔄
    ├── __main__.py             # CLI entry (Click + Hydra)
    ├── datamodule.py           # Task DataModule
    ├── model.py                # LightningModule wrapper
    ├── training.py             # Training orchestration
    ├── preparation.py          # Data preparation (placeholder)
    ├── predict.py              # Prediction pipeline
    ├── visualize.py            # Visualization from results
    └── configs/                # Hydra configs
        ├── train.yaml          # Main training config
        ├── train-mini.yaml     # Quick sanity check
        ├── prepare.yaml        # Preparation config
        ├── predict.yaml        # Prediction config
        ├── visualize.yaml      # Visualization config
        ├── model/
        │   ├── gcn.yaml        # SimpleGCN config
        │   └── gatv2.yaml      # SimpleGATv2 config (default)
        └── dataset/
            ├── base.yaml       # Full dataset (TBD)
            └── small.yaml      # Subset (TBD)
```

**Status**: ✅ Complete = Ready to use | 🔄 Placeholder = Structure ready, data prep TBD

## CLI Commands

### DeepJEB

```bash
# Data preparation (H5 -> NPZ)
uv run python -m tasks.deepjeb prepare

# Training
uv run python -m tasks.deepjeb train                    # Default (GATv2)
uv run python -m tasks.deepjeb train --config train-mini # Quick test (5 epochs)
uv run python -m tasks.deepjeb train -o model=gcn       # Use SimpleGCN

# Override config values
uv run python -m tasks.deepjeb train -o training.max_epochs=200
uv run python -m tasks.deepjeb train -o optimizer.lr=1e-4

# Prediction
uv run python -m tasks.deepjeb predict -m outputs/best.ckpt

# Visualization
uv run python -m tasks.deepjeb visualize -r outputs/predictions/results.pt
```

### HDWIA

```bash
# Data preparation (NOT YET IMPLEMENTED)
uv run python -m tasks.hdwia prepare
# Error: NotImplementedError - data prep structure ready, waiting for implementation

# Training (structure ready, waiting for prepared data)
uv run python -m tasks.hdwia train                    # GATv2 (default)
uv run python -m tasks.hdwia train --config train-mini # Quick test (5 epochs)
uv run python -m tasks.hdwia train -o model=gcn       # SimpleGCN

# Prediction and visualization (once data is available)
uv run python -m tasks.hdwia predict -m outputs/best.ckpt
uv run python -m tasks.hdwia visualize -r outputs/predictions/results.pt
```

## Models

### SimpleGCN
- Simplest GNN baseline
- 4 GCNConv layers with residual connections
- Input: 9D node features (pos, curvatures×norms, norms)
- Output: 1D regression (stress/temperature)
- Expected R²: DeepJEB ~0.65-0.66 | HDWIA TBD

### SimpleGATv2  
- GNN with multi-head attention (4 heads)
- 4 GATv2Conv layers with residual connections
- Input: 9D node features (pos, curvatures×norms, norms)
- Output: 1D regression (stress/temperature)
- Expected R²: DeepJEB ~0.68-0.69 | HDWIA TBD

Both models use **SimpleCollator** (generic for all tasks):
- Concatenates node features: `[pos, curvatures×norms, norms]` → 9D
- Renames targets to `y` for Lightning compatibility

## Logging

Experiment tracking auto-detects based on environment variables:

| Env Variable | Logger |
|--------------|--------|
| `WANDB_API_KEY` | WandB enabled |
| `MLFLOW_TRACKING_URI` | MLflow enabled |
| Both set | Both loggers active |

Configure in `.env`:
```bash
WANDB_API_KEY=your_key
MLFLOW_TRACKING_URI=https://mlflow.labs.solverx.ai
MLFLOW_TRACKING_TOKEN=your_token
```

## Development

```bash
# Run tests
make test

# Lint code
make lint

# Format code
make format
```

## Key Patterns

### 4-Stage Pipeline

```
prepare → train → predict → visualize
```

### Click + Hydra CLI

```python
@cli.command()
@click.option('--config', '-c', default='train')
@click.option('--overrides', '-o', multiple=True)
def train(config: str, overrides: tuple):
    cfg = load_hydra_config(config, overrides)
    run_training(cfg)
```

### Lightning Module

```python
class TaskLitModule(L.LightningModule):
    def training_step(self, batch, batch_idx):
        results = self.model(batch, return_loss=True)
        self.log('train/loss', results['loss'])
        return results['loss']
```

### Collator Pattern

```python
class SimpleCollator:
    def __call__(self, samples):
        for sample in samples:
            sample.x = torch.cat([sample.pos, ...], dim=-1)
            sample.y = sample.targets
        return Batch.from_data_list(samples)
```

## SkyPilot (Cloud Training)

Run training on AWS H100 GPUs using SkyPilot. No Docker required - uses `uv` + `pyproject.toml` directly.

### Data Flow

```
S3 Bucket                         SkyPilot VM
─────────                         ───────────
s3://deepjeb/
├── processed/*.h5   ──COPY──►   /data/DeepJEB/processed/
│                                       │
│                                       ▼ prepare
│                                /data/DeepJEB/prepared/*.npz
│                                       │
└── prepared/*.npz   ◄─upload─         ─┘
        │
        └────────────COPY──►    /data/DeepJEB/prepared/
                                       │
                                       ▼ train
                                  Training
```

### DATA_DIR Environment Variable

| Environment | DATA_DIR | Set by |
|-------------|----------|--------|
| 214 Server | `/home/work/data/dataset/DEEPJEB` | Default in yaml |
| SkyPilot | `/data/DeepJEB` | `envs:` in SkyPilot config |

The Hydra configs use `${oc.env:DATA_DIR,/home/work/data/dataset/DEEPJEB}` which:
- Uses `DATA_DIR` env var if set
- Falls back to 214 server path if not set

### SkyPilot Files

| File | Purpose | Command |
|------|---------|---------|
| `skypilot/sky-launch.yaml` | Launch H100 cluster, install deps | `sky launch` |
| `skypilot/sky-prepare.yaml` | Convert H5 → NPZ | `sky exec` |
| `skypilot/sky-train.yaml` | Train model | `sky exec` |
| `skypilot/sky-predict.yaml` | Run predictions | `sky exec` |
| `skypilot/sky-visualize.yaml` | Generate plots from results | `sky exec` |

### Workflow

```bash
# 1. Launch cluster (once)
sky launch -c onboard skypilot/sky-launch.yaml

# 2. Prepare data if needed (H5 → NPZ)
sky exec onboard skypilot/sky-prepare.yaml

# Optional: Upload NPZ to S3 for future runs
ssh onboard
aws s3 sync /data/DeepJEB/prepared s3://deepjeb/prepared
exit

# 3. Train model
sky exec onboard skypilot/sky-train.yaml

# Quick sanity check
sky exec onboard skypilot/sky-train.yaml --env TRAIN_CONFIG=train-mini

# Train with custom run name
sky exec onboard skypilot/sky-train.yaml --env "TRAIN_OVERRIDES=logging.name=gatv2-skypilot_check"

# Train GCN baseline
sky exec onboard skypilot/sky-train.yaml --env "TRAIN_OVERRIDES=model=gcn"

# 4. Run predictions
sky exec onboard skypilot/sky-predict.yaml --env CHECKPOINT_PATH=outputs/mini-gatv2-20251226/checkpoints/best.ckpt

# Disable visualization during predict
sky exec onboard skypilot/sky-predict.yaml --env CHECKPOINT_PATH=outputs/best.ckpt --env "PREDICT_OVERRIDES=visualization.enabled=false"

# 5. Re-generate visualizations with different settings
sky exec onboard skypilot/sky-visualize.yaml --env RESULTS_PATH=outputs/predictions/results.pt
sky exec onboard skypilot/sky-visualize.yaml --env RESULTS_PATH=outputs/predictions/results.pt --env "VIZ_OVERRIDES=visualization.vmin=-2"

# Monitor
sky logs onboard
ssh onboard

# Terminate (auto-stops after 60min idle)
sky down onboard
```

## Learning Exercises

1. **Implement HDWIA data preparation**: Convert H5 files to NPZ format (see `tasks/deepjeb/preparation.py` for reference)
2. **Modify hyperparameters**: Change learning rate, batch size, epochs
3. **Add a new model**: Implement GraphSAGE or GAT (v1) baseline
4. **Add a new metric**: Implement relative L2 error or MAPE
5. **Experiment tracking**: Enable WandB/MLflow logging via `.env`
6. **Multi-GPU training**: Modify trainer for distributed data parallel (DDP)

## References

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Lightning](https://lightning.ai/docs/pytorch/stable/)
- [Hydra](https://hydra.cc/docs/intro/)

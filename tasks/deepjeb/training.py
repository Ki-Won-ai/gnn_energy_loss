"""
Training module for DeepJEB benchmark task.

Usage:
    uv run python -m tasks.deepjeb train
    uv run python -m tasks.deepjeb train --config train-mini
"""

from __future__ import annotations

import os
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger, MLFlowLogger
from omegaconf import DictConfig

from .datamodule import DeepJEBDataModule
from .model import DeepJEBLitModel


def run_training(cfg: DictConfig):
    """
    Main training function for DeepJEB benchmark.

    Handles:
    - Setup and seeding
    - DataModule creation
    - Model creation
    - Callbacks and logger setup
    - Training execution

    Args:
        cfg: Hydra config with all training settings
    """
    print("=" * 80)
    print("DeepJEB Benchmark Training")
    print("=" * 80)

    # 1. Setup
    seed = cfg.training.get('seed', 42)
    print(f"\n Setting random seed: {seed}")
    L.seed_everything(seed, workers=True)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f" Output directory: {output_dir.absolute()}")

    # 2. Create DataModule
    print("\n Creating DataModule...")
    datamodule = DeepJEBDataModule(cfg)
    datamodule.setup('fit')

    print(f"  Dataset loaded successfully")
    print(f"    Train samples: {len(datamodule.train_dataset)}")
    print(f"    Val samples: {len(datamodule.val_dataset)}")

    # 3. Create Lightning Module
    print("\n Creating Lightning module...")
    lit_module = DeepJEBLitModel(cfg=cfg)

    # 4. Setup Callbacks
    print("\n Setting up callbacks...")
    callbacks = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / 'checkpoints',
        monitor=cfg.callbacks.checkpoint.monitor,
        mode=cfg.callbacks.checkpoint.mode,
        save_top_k=cfg.callbacks.checkpoint.save_top_k,
        save_last=cfg.callbacks.checkpoint.save_last,
        filename='epoch={epoch:02d}-val_loss={val/loss:.4f}',
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)
    print(f"   ModelCheckpoint: monitor={cfg.callbacks.checkpoint.monitor}")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    print(f"   LearningRateMonitor")

    if cfg.callbacks.early_stopping.enabled:
        early_stop = EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
        )
        callbacks.append(early_stop)
        print(f"   EarlyStopping: patience={cfg.callbacks.early_stopping.patience}")

    # 5. Setup Logger
    loggers = []
    if cfg.logging.enabled:
        wandb_api_key = os.getenv('WANDB_API_KEY')
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')

        # Auto-detect: WandB if API key present
        if wandb_api_key:
            import wandb
            print("\n Setting up WandB logging (WANDB_API_KEY detected)...")
            wandb_logger = WandbLogger(
                project=cfg.logging.project,
                name=cfg.logging.name,
                tags=cfg.logging.tags,
                notes=cfg.logging.notes,
                save_dir=output_dir / 'wandb',
                save_code=True,
                settings=wandb.Settings(api_key=wandb_api_key),
            )
            loggers.append(wandb_logger)
            print(f"   WandB project: {cfg.logging.project}")

        # Auto-detect: MLflow if tracking URI present
        if mlflow_uri:
            print("\n Setting up MLflow logging (MLFLOW_TRACKING_URI detected)...")
            tags = {}
            if cfg.logging.tags:
                for i, tag in enumerate(cfg.logging.tags):
                    tags[f"tag_{i}"] = str(tag)
            if cfg.logging.notes:
                tags["notes"] = cfg.logging.notes

            mlflow_logger = MLFlowLogger(
                experiment_name=cfg.logging.get('experiment_name', cfg.logging.project),
                run_name=cfg.logging.get('run_name', cfg.logging.name),
                tracking_uri=mlflow_uri,
                tags=tags,
                save_dir=str(output_dir / 'mlruns'),
            )
            loggers.append(mlflow_logger)
            print(f"   MLflow experiment: {cfg.logging.get('experiment_name', cfg.logging.project)}")
            print(f"   Tracking URI: {mlflow_uri}")

        if not loggers:
            print("\n Logging enabled but no API keys found (WANDB_API_KEY or MLFLOW_TRACKING_URI)")
    else:
        print("\n Logging disabled")

    # 6. Create Trainer
    print("\n Creating Trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.grad_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=cfg.training.deterministic,
        callbacks=callbacks,
        logger=loggers if loggers else None,
        default_root_dir=output_dir,
    )

    print(f"   Max epochs: {cfg.training.max_epochs}")
    print(f"   Accelerator: {cfg.training.accelerator}")
    print(f"   Devices: {cfg.training.devices}")
    print(f"   Precision: {cfg.training.precision}")

    # 7. Train!
    print("\n" + "=" * 80)
    print(" Starting training...")
    print("=" * 80 + "\n")

    resume_ckpt = cfg.training.get('resume_from_ckpt', None)
    trainer.fit(
        model=lit_module,
        datamodule=datamodule,
        ckpt_path=resume_ckpt,
    )

    # 8. Done!
    print("\n" + "=" * 80)
    print(" Training complete!")
    print("=" * 80)
    print(f"\n Checkpoints saved to: {checkpoint_callback.dirpath}")
    print(f" Best model: {checkpoint_callback.best_model_path}")
    print(f" Best {checkpoint_callback.monitor}: {checkpoint_callback.best_model_score:.6f}")

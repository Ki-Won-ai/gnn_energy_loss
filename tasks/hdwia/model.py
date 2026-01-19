"""
Lightning Module for HDWIA task.

This module wraps SimpleGCN and SimpleGATv2 models with PyTorch Lightning
infrastructure for training, validation, and prediction.
"""

from __future__ import annotations

import logging

import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.data import Data

from models import build_model
from shared.metrics import compute_metrics

logger = logging.getLogger(__name__)


class HDWIALitModel(L.LightningModule):
    """
    PyTorch Lightning module for HDWIA temperature prediction task.

    Handles:
    - Model building from config
    - Training step with loss computation
    - Validation step with metrics logging
    - Optimizer configuration with warmup + cosine schedule

    Args:
        cfg: Hydra config with model, optimizer, and metrics settings

    Example:
        >>> lit_model = HDWIALitModel(cfg)
        >>> trainer.fit(lit_model, datamodule)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        # Save model config to checkpoint for automatic model detection
        # Convert to container to ensure serialization works
        from omegaconf import OmegaConf
        self.save_hyperparameters({'model': OmegaConf.to_container(cfg.model)})

        self.model = build_model(cfg)

        # Metrics for epoch-level R² computation
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()

        # Log model info
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Total parameters: {num_params:,}")

    def forward(self, batch: Data, return_loss: bool = False) -> dict:
        """Forward pass through the model."""
        return self.model(batch, return_loss=return_loss)

    def training_step(self, batch, batch_idx):
        """Training step with loss computation and logging."""
        results = self(batch, return_loss=True)
        target = batch.y
        pred = results['y']
        loss = results['loss']

        # Log loss
        self.log('train/loss', loss, prog_bar=True, batch_size=batch.num_graphs)

        # Compute and log metrics
        for name in self.cfg.metrics:
            if name == 'r2':
                # Use torchmetrics for epoch-level R²
                self.train_r2.update(pred.flatten(), target.flatten())
                continue
            metric_value = compute_metrics(pred, target, name)
            self.log(
                f'train/{name}', metric_value,
                prog_bar=False, on_epoch=True, batch_size=batch.num_graphs
            )

        return loss

    def on_train_epoch_end(self):
        """Compute and log epoch-level R²."""
        if 'r2' in self.cfg.metrics:
            r2 = self.train_r2.compute().item()
            self.log('train/r2', r2, prog_bar=True, on_epoch=True)
            self.train_r2.reset()

    def validation_step(self, batch, batch_idx):
        """Validation step with metrics logging."""
        results = self(batch, return_loss=True)
        target = batch.y
        pred = results['y']
        loss = results['loss']

        # Log loss
        self.log('val/loss', loss, prog_bar=True, batch_size=batch.num_graphs)

        # Compute and log metrics
        for name in self.cfg.metrics:
            if name == 'r2':
                self.val_r2.update(pred.flatten(), target.flatten())
                continue
            metric_value = compute_metrics(pred, target, name)
            self.log(
                f'val/{name}', metric_value,
                prog_bar=False, batch_size=batch.num_graphs
            )

    def on_validation_epoch_end(self):
        """Compute and log epoch-level R²."""
        if 'r2' in self.cfg.metrics:
            r2 = self.val_r2.compute().item()
            self.log('val/r2', r2, prog_bar=True, on_epoch=True)
            self.val_r2.reset()

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference.

        Returns dict with predictions, targets, and metadata for visualization.
        """
        results = self(batch, return_loss=False)
        pred = results['y']
        target = batch.y

        # Get identifier from batch (filename or index)
        identifier = getattr(batch, 'identifier', None)
        if identifier is None:
            identifier = batch_idx

        return {
            'y_pred': pred.detach().cpu(),
            'y_true': target.detach().cpu(),
            'pos': batch.pos.detach().cpu(),
            'identifier': identifier,
        }

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with warmup."""
        opt_cfg = self.cfg.optimizer

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
        )

        warmup_epochs = opt_cfg.get('warmup_epochs', 0)

        if warmup_epochs > 0:
            # Warmup + cosine annealing
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs - warmup_epochs,
                eta_min=opt_cfg.get('eta_min', 1e-6)
            )
            scheduler = SequentialLR(
                optimizer,
                [warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
            )
            logger.info(
                f"LR schedule: {warmup_epochs} warmup epochs + "
                f"cosine decay (total: {self.trainer.max_epochs} epochs)"
            )
        else:
            # Pure cosine annealing
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=opt_cfg.get('eta_min', 1e-6)
            )
            logger.info(
                f"LR schedule: cosine decay over {self.trainer.max_epochs} epochs"
            )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

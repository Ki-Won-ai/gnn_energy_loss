"""
Lightning DataModule for NPZ-based PyG datasets.

This module provides a DataModule class for loading prepared NPZ data
that can be shared across different tasks.
"""

import pickle
from pathlib import Path
from typing import Callable, Optional

from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .io_manager import IOManager
from .npz_dataset import NPZDataset


class NPZDataModule(LightningDataModule):
    """
    Generic Lightning DataModule for NPZ-based PyG datasets.

    This DataModule handles loading prepared NPZ data with customizable collation.
    It supports train, validation, and optional predict stages.

    Features:
        - Configurable collator via __init__ for task-specific batching logic
        - Automatic IO manager loading for denormalization
        - Support for train/val/predict splits
        - Lazy dataset loading for memory efficiency

    Args:
        cfg: Hydra config containing:
            - prepared_data: Path to prepared data directory
            - dataloader.train/val/predict: DataLoader configurations
        collate_fn: Callable collator for batching PyG Data objects.
                   If None, uses default PyG batching.

    Example:
        >>> from shared.collators import SimpleCollator
        >>> datamodule = NPZDataModule(cfg, collate_fn=SimpleCollator())
        >>> datamodule.setup('fit')
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(
        self,
        cfg: DictConfig,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.collate_fn = collate_fn

        # Datasets (created in setup())
        self.train_dataset: Optional[NPZDataset] = None
        self.val_dataset: Optional[NPZDataset] = None
        self.predict_dataset: Optional[NPZDataset] = None

        # IO Manager (for denormalization during inference)
        self.io_manager: Optional[IOManager] = None

    def prepare_data(self):
        """
        Check if prepared data exists (called only on rank 0).

        Note: Data preparation is done by the task-specific 'prepare' command.
        This method just validates that data exists.
        """
        prepared_data = self.cfg.get('prepared_data')

        if prepared_data:
            data_dir = Path(prepared_data)
            if not data_dir.exists():
                raise FileNotFoundError(
                    f"Prepared data not found: {data_dir}\n"
                    f"Please run the 'prepare' command first."
                )

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        data_root = Path(self.cfg.prepared_data)

        # Load IO manager for denormalization
        io_info_path = data_root / 'io_info.pkl'
        if io_info_path.exists():
            with open(io_info_path, 'rb') as f:
                io_info = pickle.load(f)
            self.io_manager = IOManager()
            self.io_manager.load_io_info(io_info)

        # Setup datasets based on stage
        if stage == 'fit' or stage is None:
            self.train_dataset = NPZDataset(
                root=str(data_root),
                split='train',
            )
            self.val_dataset = NPZDataset(
                root=str(data_root),
                split='val',
            )
        elif stage == 'predict':
            self.predict_dataset = NPZDataset(
                root=str(data_root),
                split='val',
            )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.dataloader.train.batch_size,
            shuffle=self.cfg.dataloader.train.shuffle,
            num_workers=self.cfg.dataloader.train.num_workers,
            persistent_workers=self.cfg.dataloader.train.get('persistent_workers', True),
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.dataloader.val.batch_size,
            shuffle=self.cfg.dataloader.val.shuffle,
            num_workers=self.cfg.dataloader.val.num_workers,
            persistent_workers=self.cfg.dataloader.val.get('persistent_workers', True),
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        """Create prediction DataLoader."""
        return DataLoader(
            self.predict_dataset,
            batch_size=self.cfg.dataloader.predict.batch_size,
            shuffle=self.cfg.dataloader.predict.shuffle,
            num_workers=self.cfg.dataloader.predict.num_workers,
            persistent_workers=self.cfg.dataloader.predict.get('persistent_workers', True),
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

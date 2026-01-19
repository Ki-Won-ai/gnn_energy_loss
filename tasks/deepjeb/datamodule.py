"""
Lightning DataModule for DeepJEB task.

This module provides the DataModule for loading prepared DeepJEB data
with SimpleCollator for node feature preparation.
"""

from omegaconf import DictConfig

from shared.collators import SimpleCollator
from shared.npz_datamodule import NPZDataModule

__all__ = ['DeepJEBDataModule']


class DeepJEBDataModule(NPZDataModule):
    """
    Lightning DataModule for DeepJEB task.

    Uses SimpleCollator by default for node feature preparation:
    - Concatenates pos, curvature_weighted_norms, norms -> (N, 9)
    - Renames targets to y

    Example:
        >>> datamodule = DeepJEBDataModule(cfg)
        >>> datamodule.setup('fit')
        >>> train_loader = datamodule.train_dataloader()
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize DataModule with SimpleCollator.

        Args:
            cfg: Hydra config with prepared_data and dataloader settings
        """
        collate_fn = SimpleCollator()
        super().__init__(cfg, collate_fn=collate_fn)

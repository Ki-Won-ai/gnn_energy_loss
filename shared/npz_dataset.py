"""
NPZ Dataset for loading prepared PyG data.

This module provides a dataset class for loading preprocessed NPZ data
that was created by the prepare command.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class NPZDataset(Dataset[Data]):
    """
    Generic dataset for loading prepared NPZ data.

    This dataset loads preprocessed PyG data that was created by the prepare command.
    Each sample is stored as a single NPZ file with tensors and metadata.

    Structure:
        root/
        ├── metadata.parquet     # Dataset metadata with 'identifier' and 'is_train' columns
        ├── io_info.pkl          # IO normalization info (optional)
        ├── *.npz                # Data files named by identifier

    Features:
        - Lazy loading: NPZ files are loaded on-demand for memory efficiency
        - Automatic tensor conversion: numpy arrays are converted to torch tensors
        - Identifier tracking: Each sample includes its 'identifier' field

    Args:
        root: Root directory containing prepared data
        split: One of 'train', 'val'. Filters data based on 'is_train' column.

    Example:
        >>> dataset = NPZDataset('outputs/prepared', split='train')
        >>> data = dataset[0]
        >>> print(data.identifier)
        'sample_001'
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
    ):
        """
        Initialize dataset.

        Args:
            root: Root directory with prepared data
            split: 'train' or 'val'
        """
        self.root = Path(root)
        self.split = split

        # Check if directory exists
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root}\n"
                f"Please run 'prepare' command first."
            )

        # Load metadata to get sample information
        metadata_path = self.root / 'metadata.parquet'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found: {metadata_path}\n"
                f"Please run 'prepare' command first."
            )

        metadata = pd.read_parquet(metadata_path)
        metadata = metadata[metadata['is_train'] == (split == 'train')]

        # Store file paths and identifiers for lazy loading
        self.file_paths = [
            self.root / f"{row['identifier']}.npz"
            for _, row in metadata.iterrows()
        ]
        self.identifiers = [
            row['identifier']
            for _, row in metadata.iterrows()
        ]

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Data:
        """
        Load a single sample.

        Args:
            idx: Sample index

        Returns:
            PyG Data object with all tensors from NPZ file
        """
        # Lazy load NPZ file on demand
        path = self.file_paths[idx]
        data_dict = np.load(path, allow_pickle=True)

        # Convert numpy arrays to torch tensors
        data_dict = {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in data_dict.items()
        }

        # Add identifier for traceability
        data_dict['identifier'] = self.identifiers[idx]

        return Data.from_dict(data_dict)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'root={self.root}, '
            f'split={self.split}, '
            f'num_samples={len(self)})'
        )

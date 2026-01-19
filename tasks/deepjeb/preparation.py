"""
Data preparation module for DeepJEB benchmark task.

Converts HDF5 files to NPZ format for efficient training.

Usage:
    uv run python -m tasks.deepjeb prepare
"""

import os
import pickle
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from shared.io_manager import IOManager


def load_single_file(file_path: str) -> dict[str, torch.Tensor]:
    """Load an HDF5 file and convert to tensors."""
    with h5py.File(file_path, 'r') as f:
        data = {k: torch.tensor(f[k][:]) for k in f.keys()}
    return data


def extract_target(data: dict[str, torch.Tensor], cfg: DictConfig) -> torch.Tensor:
    """
    Extract target values based on config.

    Supports:
        - Direct key: "von_mises" -> data['von_mises']
        - Slice notation: "outputs[:,0,4]" -> data['outputs'][:,0,4]
    """
    target_key = cfg.feature.target_key

    if '[:,' in target_key:
        # Parse slice notation like "outputs[:,0,4]"
        key = target_key.split('[')[0]
        slice_str = target_key.split('[')[1].rstrip(']')
        indices = [int(i) for i in slice_str.replace(':', '').split(',') if i]
        tensor = data[key].float()
        return tensor[:, indices[0], indices[1]]
    else:
        return data[target_key].float()


def process_sample(
    data: dict[str, torch.Tensor],
    io_manager: IOManager,
    cfg: DictConfig,
) -> dict[str, np.ndarray]:
    """
    Process a single sample: normalize features and prepare for saving.

    Args:
        data: Raw data dictionary from HDF5
        io_manager: IOManager with normalization info
        cfg: Config with feature settings

    Returns:
        Dictionary of numpy arrays ready for NPZ saving
    """
    # Extract features based on config keys
    pos = data[cfg.feature.pos_key].float()
    curvatures = data.get(cfg.feature.curvatures_key, torch.zeros(pos.shape[0], 1)).float()
    norms = data.get(cfg.feature.norms_key, torch.zeros_like(pos)).float()
    edge_index = data[cfg.feature.edge_key].long()
    targets = extract_target(data, cfg)

    # Ensure proper shapes
    if curvatures.ndim == 1:
        curvatures = curvatures.unsqueeze(-1)
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)

    # Normalize targets if configured
    if cfg.feature.get('normalize_targets', False):
        targets = io_manager.normalize(targets, cfg.feature.target_key)

    # Center positions (optional)
    if cfg.feature.get('center_positions', True):
        pos = pos - pos.mean(dim=0, keepdim=True)

    # Normalize scale (optional)
    if cfg.feature.get('normalize_scale', True):
        scale = pos.abs().max()
        if scale > 1e-8:
            pos = pos / scale

    # Create output dictionary
    output = {
        'pos': pos.numpy(),
        'curvatures': curvatures.numpy(),
        'norms': norms.numpy(),
        'edge_index': edge_index.numpy(),
        'targets': targets.numpy(),
    }

    return output


@torch.no_grad()
def run_preparation(cfg: DictConfig):
    """
    Main data preparation function.

    Converts HDF5 files to NPZ format with optional normalization.

    Args:
        cfg: Hydra config with dataset and feature settings
    """
    print("=" * 80)
    print("DeepJEB Data Preparation")
    print("=" * 80)

    io_manager = IOManager()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1. Load training data and collect statistics
    print("\n Loading training data...")
    train_data = []
    for path in tqdm(cfg.dataset.train, desc='Loading train'):
        data = load_single_file(Path(cfg.source_dir) / path)
        # Extract targets for statistics collection
        data[cfg.feature.target_key] = extract_target(data, cfg)
        train_data.append(data)

    # 2. Collect statistics for normalization
    if cfg.feature.get('normalize_targets', False):
        print("\n Collecting feature statistics...")
        io_manager.collect_feature_statistics(
            train_data,
            [cfg.feature.target_key]
        )
        io_manager.set_zscore_normalization(cfg.feature.target_key)
        print(f"   Target key: {cfg.feature.target_key}")
        print(f"   Stats: {io_manager.feature_stats[cfg.feature.target_key]}")

    # 3. Load validation data
    print("\n Loading validation data...")
    val_data = []
    for path in tqdm(cfg.dataset.valid, desc='Loading valid'):
        data = load_single_file(Path(cfg.source_dir) / path)
        val_data.append(data)

    # 4. Process and save all data
    print("\n Processing and saving data...")
    all_data = train_data + val_data
    all_paths = cfg.dataset.train + cfg.dataset.valid

    metadata_records = []
    for idx, (data, file_path) in enumerate(
        tqdm(zip(all_data, all_paths), total=len(all_data), desc='Converting')
    ):
        try:
            # Process sample
            processed = process_sample(data, io_manager, cfg)

            # Save as NPZ
            path = Path(file_path).with_suffix('.npz')
            dst_path = Path(cfg.output_dir) / path.name
            np.savez(dst_path, **processed)

            # Track metadata
            is_train = idx < len(cfg.dataset.train)
            metadata_records.append({
                'identifier': path.stem,
                'is_train': is_train,
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            print(traceback.format_exc())
            continue

    # 5. Save IO info
    io_info = io_manager.get_io_info()
    with open(f"{cfg.output_dir}/io_info.pkl", 'wb') as f:
        pickle.dump(io_info, f)

    # 6. Save metadata
    pd.DataFrame.from_records(metadata_records).to_parquet(
        f"{cfg.output_dir}/metadata.parquet"
    )

    print("\n" + "=" * 80)
    print(" Preparation complete!")
    print("=" * 80)
    print(f"\n Output directory: {cfg.output_dir}")
    print(f"   Train samples: {len(cfg.dataset.train)}")
    print(f"   Val samples: {len(cfg.dataset.valid)}")
    print(f"   Total samples: {len(metadata_records)}")

"""
Model factory for building GNN models from config.
"""

import pickle
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .simple_gcn import SimpleGCN
from .simple_gatv2 import SimpleGATv2


def _load_y_norm_params(cfg: DictConfig) -> dict:
    """
    Load y normalization parameters (mean, std) from io_info.pkl.
    
    Args:
        cfg: Hydra config containing data.io_info_path
        
    Returns:
        Dictionary with y_mean and y_std keys
    """
    # Check if io_info_path is specified in config
    if not hasattr(cfg, 'data') or not hasattr(cfg.data, 'io_info_path'):
        return {}
    
    io_info_path = Path(cfg.data.io_info_path)
    if not io_info_path.exists():
        print(f"Warning: io_info.pkl not found at {io_info_path}, using default y_mean=0, y_std=1")
        return {}
    
    try:
        with open(io_info_path, 'rb') as f:
            io_info = pickle.load(f)
        
        # Look for output normalization info (key pattern: 'outputs[:,0,4]' or similar)
        for key, value in io_info.items():
            if 'output' in key.lower() or key.startswith('outputs'):
                if isinstance(value, dict) and 'center' in value and 'scale' in value:
                    return {
                        'y_mean': float(value['center']),
                        'y_std': float(value['scale'])
                    }
        
        print(f"Warning: Could not find output normalization params in {io_info_path}")
        return {}
    except Exception as e:
        print(f"Warning: Failed to load io_info.pkl: {e}")
        return {}


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Factory function for model instantiation.

    Args:
        cfg: Hydra config containing model configuration under cfg.model
             For GATv2, also loads y normalization params from cfg.data.io_info_path

    Returns:
        Instantiated model

    Example:
        >>> cfg = OmegaConf.create({
        ...     'model': {
        ...         'type': 'gatv2',
        ...         'arch': {'in_channels': 9, 'hidden_channels': 128}
        ...     },
        ...     'data': {
        ...         'io_info_path': '/path/to/io_info.pkl'
        ...     }
        ... })
        >>> model = build_model(cfg)
    """
    model_type = cfg.model.type

    if model_type == "gcn":
        return SimpleGCN(**cfg.model.arch)
    elif model_type == "gatv2":
        # Load y normalization parameters for energy-weighted loss
        arch_params = OmegaConf.to_container(cfg.model.arch, resolve=True)
        y_norm_params = _load_y_norm_params(cfg)
        arch_params.update(y_norm_params)
        return SimpleGATv2(**arch_params)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Supported types: gcn, gatv2"
        )

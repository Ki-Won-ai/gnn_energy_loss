"""
SimpleGATv2: GNN with attention mechanism.

This model uses GATv2Conv layers with multi-head attention and residual
connections for node-level regression.

Expected DeepJEB R²: ~0.85-0.90
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from .losses import LogEnergyWeightedMSELoss, EnergyWeightedMSELoss


class SimpleGATv2(nn.Module):
    """
    GATv2 baseline with attention mechanism.

    Architecture:
        Input projection (Linear) -> N x GATv2Conv layers (with residual)
        -> LayerNorm -> MLP head

    Features:
        - Multi-head attention for learning edge importance
        - Residual connections for stable training
        - Support for attention weight visualization
        - Energy-weighted loss using denormalized stress values

    Args:
        in_channels: Number of input features (default: 9 for DeepJEB)
        hidden_channels: Hidden dimension (default: 128)
        num_layers: Number of GATv2 layers (default: 4)
        heads: Number of attention heads (default: 4)
        out_channels: Number of output features (default: 1)
        dropout: Dropout rate (default: 0.1)
        y_mean: Mean (μ) of original y values for z-score denormalization (default: 0.0)
        y_std: Std (σ) of original y values for z-score denormalization (default: 1.0)
        loss_type: Type of energy-weighted loss function (default: "log")
            - "log": w = 1 + log(1 + |stress|) - smoother weighting
            - "squared": w = (stress + ε)² - aggressive weighting

    Example:
        >>> model = SimpleGATv2(in_channels=9, hidden_channels=128, y_mean=22.86, y_std=85.74)
        >>> batch = Batch.from_data_list([data1, data2])
        >>> pred = model(batch)  # (total_nodes, 1)
        >>> pred, attns = model(batch, return_attention=True)  # With attention weights
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 128,
        num_layers: int = 4,
        heads: int = 4,
        out_channels: int = 1,
        dropout: float = 0.1,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        loss_type: str = "log",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.loss_type = loss_type

        # Loss function for energy-weighted MSE
        if loss_type == "log":
            self.loss_fn = LogEnergyWeightedMSELoss(y_mean=y_mean, y_std=y_std)
        elif loss_type == "squared":
            self.loss_fn = EnergyWeightedMSELoss(y_mean=y_mean, y_std=y_std)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'log' or 'squared'.")

        # Ensure hidden_channels is divisible by heads
        assert hidden_channels % heads == 0, (
            f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
        )

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GATv2 layers
        # Each layer has `heads` attention heads outputting hidden_channels // heads each
        self.convs = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                concat=True,  # Concatenate head outputs
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(hidden_channels)

        # MLP head for regression
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(
        self,
        data: Data,
        return_loss: bool = False,
        return_attention: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index
            return_loss: If True, compute and return custom loss
            return_attention: If True, return attention weights

        Returns:
            Dictionary with:
            - 'y': Predicted values (N, out_channels)
            - 'loss': Custom loss (y_pred^2 - y^2)^2 (if return_loss=True)
            - 'attentions': List of attention weights (if return_attention=True)
        """
        x = data.x
        edge_index = data.edge_index

        # Input projection
        x = self.input_proj(x)

        # Track attention weights if requested
        attentions = [] if return_attention else None

        # GATv2 layers with residual connections
        for conv in self.convs:
            x_res = x

            if return_attention:
                x_new, attn = conv(x, edge_index, return_attention_weights=True)
                attentions.append(attn)
            else:
                x_new = conv(x, edge_index)

            x = F.gelu(x_new)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection

        # Normalize and predict
        x = self.norm(x)
        y_pred = self.head(x)

        results = {'y': y_pred}

        if return_loss and hasattr(data, 'y'):
            # Use energy-weighted MSE loss function
            loss = self.loss_fn(y_pred, data.y)
            results['loss'] = loss

        if return_attention:
            results['attentions'] = attentions

        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'in_channels={self.in_channels}, '
            f'hidden_channels={self.hidden_channels}, '
            f'num_layers={self.num_layers}, '
            f'heads={self.heads}, '
            f'out_channels={self.out_channels}, '
            f'loss_type={self.loss_type!r})'
        )

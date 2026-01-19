"""
SimpleGCN: Simplest GNN baseline.

This is the simplest possible GNN baseline using GCNConv layers with residual
connections and an MLP head for node-level regression.

Expected DeepJEB R²: ~0.75-0.80
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class SimpleGCN(nn.Module):
    """
    Simplest GNN baseline using GCN layers.

    Architecture:
        Input projection (Linear) -> N x GCNConv layers (with residual) -> MLP head

    Args:
        in_channels: Number of input features (default: 9 for DeepJEB)
        hidden_channels: Hidden dimension (default: 64)
        num_layers: Number of GCN layers (default: 4)
        out_channels: Number of output features (default: 1)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> model = SimpleGCN(in_channels=9, hidden_channels=64)
        >>> batch = Batch.from_data_list([data1, data2])
        >>> pred = model(batch)  # (total_nodes, 1)
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 64,
        num_layers: int = 4,
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GCN layers
        self.convs = nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels)
            for _ in range(num_layers)
        ])

        # Layer normalization after GNN
        self.norm = nn.LayerNorm(hidden_channels)

        # MLP head for regression
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, data: Data, return_loss: bool = False) -> dict:
        """
        Forward pass.

        Args:
            data: PyG Data object with x, edge_index
            return_loss: If True, compute and return custom loss

        Returns:
            Dictionary with:
            - 'y': Predicted values (N, out_channels)
            - 'loss': Custom loss (y_pred^2 - y^2)^2 (if return_loss=True)
        """
        x = data.x
        edge_index = data.edge_index

        # Input projection
        x = self.input_proj(x)

        # GCN layers with residual connections
        for conv in self.convs:
            x_res = x
            x = conv(x, edge_index)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection

        # Normalize and predict
        x = self.norm(x)
        y_pred = self.head(x)

        results = {'y': y_pred}

        if return_loss and hasattr(data, 'y'):
            # Custom loss: ((y+0.001)^2)(y_pred - y)^2
            loss = torch.mean((data.y+0.001)**2 * (y_pred - data.y) ** 2)
            results['loss'] = loss

        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'in_channels={self.in_channels}, '
            f'hidden_channels={self.hidden_channels}, '
            f'num_layers={self.num_layers}, '
            f'out_channels={self.out_channels})'
        )

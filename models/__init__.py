"""GNN model implementations."""

from .simple_gcn import SimpleGCN
from .simple_gatv2 import SimpleGATv2
from .builder import build_model
from .losses import (
    LogEnergyWeightedMSELoss,
    EnergyWeightedMSELoss,
    log_energy_weighted_mse_loss,
    energy_weighted_mse_loss,
)

__all__ = [
    "SimpleGCN",
    "SimpleGATv2",
    "build_model",
    # Loss classes
    "LogEnergyWeightedMSELoss",
    "EnergyWeightedMSELoss",
    # Loss functions
    "log_energy_weighted_mse_loss",
    "energy_weighted_mse_loss",
]

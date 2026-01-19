"""
Custom loss functions for GNN models.

This module provides various loss functions for training GNN models,
including energy-weighted losses for stress prediction.

Available loss functions:
    - LogEnergyWeightedMSELoss: w = 1 + log(1 + |stress|) - smoother weighting
    - EnergyWeightedMSELoss: w = (stress + ε)² - aggressive weighting
"""

import torch
import torch.nn as nn
from typing import Optional


class LogEnergyWeightedMSELoss(nn.Module):
    """
    Log-scaled energy-weighted MSE loss for stress prediction.
    
    This loss gives more weight to larger stress values using log scaling,
    providing a smoother gradient for high-stress regions.
    
    Loss formula:
        L = mean(w * (y_pred - y_true)^2)
        where w = 1 + log(1 + |original_stress|)
    
    Args:
        y_mean: Mean (μ) of original y values for z-score denormalization
        y_std: Std (σ) of original y values for z-score denormalization
        
    Example:
        >>> loss_fn = LogEnergyWeightedMSELoss(y_mean=22.86, y_std=85.74)
        >>> loss = loss_fn(y_pred, y_true)
    """
    
    def __init__(self, y_mean: float = 0.0, y_std: float = 1.0):
        super().__init__()
        self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer('y_std', torch.tensor(y_std, dtype=torch.float32))
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute log-scaled energy-weighted MSE loss.
        
        Args:
            y_pred: Predicted values (normalized)
            y_true: Ground truth values (normalized)
            
        Returns:
            Scalar loss value
        """
        # Denormalize y to get original stress values: original = z * σ + μ
        original_stress = y_true * self.y_std + self.y_mean
        
        # Log-scaled energy weight: smoother weighting for large stress values
        energy_weight = 1 + torch.log(1 + torch.abs(original_stress))
        
        # Weighted MSE
        loss = torch.mean(energy_weight * (y_pred - y_true) ** 2)
        
        return loss


class EnergyWeightedMSELoss(nn.Module):
    """
    Energy-weighted MSE loss for stress prediction.
    
    This loss gives aggressive weighting to larger stress values using
    the square of stress (proportional to energy: E ~ σ²).
    
    Loss formula:
        L = mean(w * (y_pred - y_true)^2)
        where w = (original_stress + ε)²
    
    Args:
        y_mean: Mean (μ) of original y values for z-score denormalization
        y_std: Std (σ) of original y values for z-score denormalization
        eps: Small epsilon for numerical stability (default: 0.001)
        
    Example:
        >>> loss_fn = EnergyWeightedMSELoss(y_mean=22.86, y_std=85.74)
        >>> loss = loss_fn(y_pred, y_true)
    """
    
    def __init__(self, y_mean: float = 0.0, y_std: float = 1.0, eps: float = 0.001):
        super().__init__()
        self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer('y_std', torch.tensor(y_std, dtype=torch.float32))
        self.eps = eps
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute squared energy-weighted MSE loss.
        
        Args:
            y_pred: Predicted values (normalized)
            y_true: Ground truth values (normalized)
            
        Returns:
            Scalar loss value
        """
        # Denormalize y to get original stress values: original = z * σ + μ
        original_stress = y_true * self.y_std + self.y_mean
        
        # Energy weight: aggressive weighting (energy ~ stress²)
        energy_weight = (original_stress + self.eps) ** 2
        
        # Weighted MSE
        loss = torch.mean(energy_weight * (y_pred - y_true) ** 2)
        
        return loss


def log_energy_weighted_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mean: float = 0.0,
    y_std: float = 1.0,
) -> torch.Tensor:
    """
    Functional version of log-scaled energy-weighted MSE loss.
    
    Args:
        y_pred: Predicted values (normalized)
        y_true: Ground truth values (normalized)
        y_mean: Mean (μ) of original y values for z-score denormalization
        y_std: Std (σ) of original y values for z-score denormalization
        
    Returns:
        Scalar loss value
        
    Example:
        >>> loss = log_energy_weighted_mse_loss(y_pred, y_true, y_mean=22.86, y_std=85.74)
    """
    # Denormalize y to get original stress values
    original_stress = y_true * y_std + y_mean
    
    # Log-scaled energy weight
    energy_weight = 1 + torch.log(1 + torch.abs(original_stress))
    
    # Weighted MSE
    loss = torch.mean(energy_weight * (y_pred - y_true) ** 2)
    
    return loss


def energy_weighted_mse_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    eps: float = 0.001,
) -> torch.Tensor:
    """
    Functional version of energy-weighted MSE loss.
    
    Args:
        y_pred: Predicted values (normalized)
        y_true: Ground truth values (normalized)
        y_mean: Mean (μ) of original y values for z-score denormalization
        y_std: Std (σ) of original y values for z-score denormalization
        eps: Small epsilon for numerical stability (default: 0.001)
        
    Returns:
        Scalar loss value
        
    Example:
        >>> loss = energy_weighted_mse_loss(y_pred, y_true, y_mean=22.86, y_std=85.74)
    """
    # Denormalize y to get original stress values
    original_stress = y_true * y_std + y_mean
    
    # Energy weight
    energy_weight = (original_stress + eps) ** 2
    
    # Weighted MSE
    loss = torch.mean(energy_weight * (y_pred - y_true) ** 2)
    
    return loss

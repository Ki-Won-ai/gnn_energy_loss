"""
Core metrics for training and evaluation.

Provides essential loss functions and metrics:
- MSE, MAE, RMSE (standard losses)
- R² score (coefficient of determination)
- Relative L2 error
"""

import torch
import torch.nn.functional as F

EPS_SMALL = 1e-8


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    name: str,
    **kwargs,
) -> torch.Tensor:
    """
    Compute a metric by name.

    Args:
        pred: Predicted values
        target: Ground truth values
        name: Metric name (mse, mae, rmse, r2, rel_l2)
        **kwargs: Additional arguments for specific metrics

    Returns:
        Scalar metric value
    """
    # Flatten tensors
    pred = pred.flatten()
    target = target.flatten()

    name = name.lower()

    if name == "mse":
        return mse_loss(pred, target)
    elif name == "mae":
        return mae_loss(pred, target)
    elif name == "rmse":
        return rmse_loss(pred, target)
    elif name == "r2":
        return r2_score(pred, target)
    elif name == "rel_l2":
        return rel_l2_loss(pred, target)
    else:
        raise ValueError(f"Unknown metric name: {name}")


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error loss.

    MSE = mean((pred - target)^2)
    """
    return torch.mean((pred - target) ** 2)


def squared_diff_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Squared difference of squares loss.

    Loss = mean((pred^2 - target^2)^2)
    """
    return torch.mean((pred ** 2 - target ** 2) ** 2)


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error loss.

    MAE = mean(|pred - target|)
    """
    return torch.mean(torch.abs(pred - target))


def rmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Root Mean Squared Error loss.

    RMSE = sqrt(mean((pred - target)^2))
    """
    return torch.sqrt(torch.mean((pred - target) ** 2))


def rel_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Relative L2 error.

    rel_L2 = ||pred - target||_2 / ||target||_2
    """
    l2_diff = torch.sqrt(torch.sum((pred - target) ** 2))
    l2_target = torch.sqrt(torch.sum(target ** 2))
    return l2_diff / (l2_target + EPS_SMALL)


def r2_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute R² (coefficient of determination) score.

    R² = 1 - SS_res / SS_tot
    where:
        SS_res = sum((target - pred)^2)  # Residual sum of squares
        SS_tot = sum((target - mean(target))^2)  # Total sum of squares

    Args:
        pred: Predicted values (flattened)
        target: Ground truth values (flattened)

    Returns:
        R² score (1.0 = perfect, 0.0 = baseline, negative = worse than baseline)
    """
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)

    # Handle edge case: if target is constant (ss_tot ≈ 0), R² is undefined
    if ss_tot < EPS_SMALL:
        return torch.tensor(0.0, device=pred.device)

    return 1.0 - (ss_res / ss_tot)

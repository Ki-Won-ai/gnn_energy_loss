"""
Visualization utilities for GNN predictions.

Provides 3D comparison plots for ground truth vs predictions.
Uses matplotlib for static plot generation.
"""

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch


def plot_comparison(
    coords: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    title: str,
    save_path: str,
    component_mask: Optional[torch.Tensor] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    error_vmin: float = 0.0,
    error_vmax: float = 0.5,
):
    """
    Create 3-panel comparison plot: Ground Truth | Prediction | Error.

    Args:
        coords: Node coordinates [N, 3]
        pred: Predicted values [N] or [N, 1]
        gt: Ground truth values [N] or [N, 1]
        title: Plot title (usually sample identifier)
        save_path: Path to save the figure
        component_mask: Optional boolean mask for filtering nodes
        vmin: Minimum value for colormap (default: -1.0 for normalized data)
        vmax: Maximum value for colormap (default: 1.0 for normalized data)
        error_vmin: Minimum value for error colormap
        error_vmax: Maximum value for error colormap

    Example:
        >>> plot_comparison(
        ...     coords=data.pos,
        ...     pred=model_output,
        ...     gt=data.y,
        ...     title="sample_001",
        ...     save_path="outputs/comparison_001.png"
        ... )
    """
    # Convert to numpy (float() handles bfloat16 which numpy doesn't support)
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().float().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().float().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().float().numpy()

    # Flatten if needed
    pred = pred.flatten()
    gt = gt.flatten()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Apply component mask if provided
    if component_mask is not None:
        if isinstance(component_mask, torch.Tensor):
            component_mask = component_mask.detach().cpu().numpy().astype(bool)
        if component_mask.sum() > 0:
            coords = coords[component_mask]
            pred = pred[component_mask]
            gt = gt[component_mask]

    # Ground Truth
    sc0 = axes[0].scatter3D(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=gt, cmap='seismic', norm=norm, s=1
    )
    axes[0].set_title(f'Ground Truth\n{title}')
    axes[0].set_aspect('equal')
    plt.colorbar(sc0, ax=axes[0], shrink=0.6)

    # Prediction
    sc1 = axes[1].scatter3D(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=pred, cmap='seismic', norm=norm, s=1
    )
    axes[1].set_title(f'Prediction\n{title}')
    axes[1].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[1], shrink=0.6)

    # Absolute Error
    error = np.abs(pred - gt)
    sc2 = axes[2].scatter3D(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=error, cmap='hot', s=1, vmin=error_vmin, vmax=error_vmax
    )
    axes[2].set_title(f'Absolute Error\n{title}')
    axes[2].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[2], shrink=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter(
    pred: torch.Tensor,
    gt: torch.Tensor,
    title: str,
    save_path: str,
    error_range: float = 0.3,
    plot_range: Optional[tuple] = None,
):
    """
    Create scatter plot of predictions vs ground truth.

    Args:
        pred: Predicted values [N]
        gt: Ground truth values [N]
        title: Plot title
        save_path: Path to save the figure
        error_range: Acceptable error range as fraction (default: 0.3 = +/-30%)
        plot_range: Optional (min, max) for axis limits

    Example:
        >>> plot_scatter(pred, gt, "sample_001", "outputs/scatter_001.png")
    """
    # Convert to numpy (float() handles bfloat16 which numpy doesn't support)
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().float().numpy().flatten()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().float().numpy().flatten()

    # Determine plot range
    if plot_range is None:
        vmin = float(min(gt.min(), pred.min()))
        vmax = float(max(gt.max(), pred.max()))
        pad = 0.05 * (vmax - vmin + 1e-8)
        plot_range = (vmin - pad, vmax + pad)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Error band
    line_range = np.linspace(plot_range[0], plot_range[1], 100)
    upper_bound = (1 + error_range) * line_range
    lower_bound = (1 - error_range) * line_range

    ax.fill_between(
        line_range, lower_bound, upper_bound,
        color='lightgray', alpha=0.3, label=f'+/-{error_range*100:.0f}% range'
    )

    # Scatter plot
    ax.scatter(gt, pred, alpha=0.5, s=8, c='steelblue', label=f'{len(gt):,} points')

    # Reference lines
    ax.plot(line_range, line_range, 'k--', alpha=0.7, linewidth=1, label='y = x')
    ax.plot(line_range, upper_bound, c='cornflowerblue', ls='--', alpha=0.5, linewidth=1)
    ax.plot(line_range, lower_bound, c='crimson', ls='--', alpha=0.5, linewidth=1)

    # Axis settings
    ax.set_xlim(plot_range[0], plot_range[1])
    ax.set_ylim(plot_range[0], plot_range[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Statistics
    ratio_diff = np.abs(pred - gt) / (np.abs(gt) + 1e-8)
    within_range = np.sum(ratio_diff <= error_range) / len(ratio_diff) * 100
    mae = np.abs(pred - gt).mean()

    ax.set_title(
        f'{title}\nWithin +/-{error_range*100:.0f}%: {within_range:.1f}% | MAE: {mae:.4f}',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlabel('Ground Truth', fontsize=11)
    ax.set_ylabel('Prediction', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

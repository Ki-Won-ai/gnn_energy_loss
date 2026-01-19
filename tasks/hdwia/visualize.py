"""
Visualization pipeline for HDWIA task.

Generates plots from saved prediction results without re-running inference.
"""

from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from shared.visualization import plot_comparison, plot_scatter


def run_visualize(cfg: DictConfig):
    """
    Generate visualizations from saved prediction results.

    Args:
        cfg: Hydra config with:
            - results_path: Path to results.pt from predict
            - output_dir: Directory for visualization outputs
            - visualization: Plot settings (vmin, vmax, error_vmax)

    Example:
        >>> cfg = load_hydra_config("visualize", ())
        >>> run_visualize(cfg)
    """
    # Validate results path
    results_path = Path(cfg.results_path)
    if not results_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_path}\n"
            f"Run 'predict' command first to generate results.pt"
        )

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f" Output directory: {output_dir.absolute()}")

    # Load results
    print(f"\n Loading results from: {results_path}")
    results = torch.load(results_path, weights_only=False)
    print(f"   Loaded {len(results)} samples")

    # Get visualization settings
    viz_cfg = cfg.visualization
    vmin = viz_cfg.get('vmin', -1.0)
    vmax = viz_cfg.get('vmax', 1.0)
    error_vmax = viz_cfg.get('error_vmax', 0.5)

    print(f"\n Visualization settings:")
    print(f"   vmin: {vmin}, vmax: {vmax}, error_vmax: {error_vmax}")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate comparison plots
    if viz_cfg.get('plot_comparison', {}).get('enabled', True):
        print("\n Creating 3D comparison plots...")
        for result in tqdm(results, desc="   Plotting"):
            identifier = result.get('identifier', 'unknown')
            if isinstance(identifier, list):
                identifier = identifier[0]

            save_path = plots_dir / f"comparison_{identifier}.png"
            plot_comparison(
                coords=result['pos'].to(dtype=torch.float32),
                pred=result['y_pred'].to(dtype=torch.float32),
                gt=result['y_true'].to(dtype=torch.float32),
                title=str(identifier),
                save_path=str(save_path),
                vmin=vmin,
                vmax=vmax,
                error_vmax=error_vmax,
            )
        print(f"   Saved {len(results)} comparison plots to: {plots_dir}")

    # Generate aggregate scatter plot
    if viz_cfg.get('plot_scatter', {}).get('enabled', True):
        print("\n Creating scatter plot...")

        # Aggregate all predictions
        y_preds = [r['y_pred'].flatten() for r in results]
        y_trues = [r['y_true'].flatten() for r in results]
        y_pred_all = torch.cat(y_preds)
        y_true_all = torch.cat(y_trues)

        scatter_path = output_dir / "scatter_all.png"
        plot_scatter(
            pred=y_pred_all,
            gt=y_true_all,
            title="All Samples",
            save_path=str(scatter_path),
        )
        print(f"   Saved scatter plot to: {scatter_path}")

    print("\n" + "=" * 60)
    print(" Visualization complete!")
    print("=" * 60)

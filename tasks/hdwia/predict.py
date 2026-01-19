"""
Prediction pipeline for HDWIA task.

Runs inference on test data, computes metrics, and generates visualizations.
"""

from pathlib import Path

import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from shared.metrics import mse_loss, rmse_loss
from shared.visualization import plot_comparison, plot_scatter

from .datamodule import HDWIADataModule
from .model import HDWIALitModel


def load_model_config_from_checkpoint(ckpt_path: str) -> dict | None:
    """
    Load model configuration from checkpoint hyperparameters.

    Args:
        ckpt_path: Path to the checkpoint file

    Returns:
        Model config dict if found, None otherwise
    """
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Check for hyperparameters (saved by save_hyperparameters)
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        if 'model' in hparams:
            return hparams['model']

    return None


def run_predict(cfg: DictConfig):
    """
    Run prediction pipeline.

    Steps:
        1. Load checkpoint
        2. Run predictions on test/val data
        3. Compute aggregate metrics (MSE, RMSE, R2)
        4. Save results and metrics
        5. Generate comparison visualizations (optional)

    Args:
        cfg: Hydra config with:
            - model_checkpoint_path: Path to checkpoint
            - output_dir: Directory for outputs
            - training: Trainer settings (accelerator, devices, precision)
            - visualization: Plot settings (enabled, vmin, vmax)

    Example:
        >>> cfg = load_hydra_config("predict", ())
        >>> run_predict(cfg)
    """
    # Setup
    seed = cfg.training.get('seed', 42)
    print(f"\n Setting random seed: {seed}")
    L.seed_everything(seed, workers=True)

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f" Output directory: {output_dir.absolute()}")

    # Validate checkpoint
    ckpt_path = cfg.model_checkpoint_path
    if ckpt_path is None:
        raise ValueError("model_checkpoint_path is required for prediction")
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f" Checkpoint: {ckpt_path}")

    # Load model config from checkpoint (auto-detect model type)
    print("\n Loading model configuration from checkpoint...")
    ckpt_model_config = load_model_config_from_checkpoint(ckpt_path)

    if ckpt_model_config is not None:
        # Override config with model settings from checkpoint
        ckpt_model_type = ckpt_model_config.get('type', 'unknown')
        config_model_type = cfg.model.type
        print(f"   Checkpoint model type: {ckpt_model_type}")
        print(f"   Config model type: {config_model_type}")

        if ckpt_model_type != config_model_type:
            print(f"   Overriding config model '{config_model_type}' "
                  f"with checkpoint model '{ckpt_model_type}'")

        # Replace model config entirely (not merge, to avoid mixing arch params)
        OmegaConf.set_struct(cfg, False)
        cfg.model = OmegaConf.create(ckpt_model_config)
        OmegaConf.set_struct(cfg, True)
        print(f"   Using model: {cfg.model.type}")
    else:
        print("   No model config found in checkpoint (legacy checkpoint)")
        print(f"   Using config model: {cfg.model.type}")

    # Create DataModule
    print("\n Creating DataModule...")
    datamodule = HDWIADataModule(cfg)
    datamodule.setup('predict')
    print(f"   Dataset loaded: {len(datamodule.predict_dataset)} samples")

    # Create model (uses config from checkpoint if available)
    print("\n Creating Lightning module...")
    lit_module = HDWIALitModel(cfg=cfg)

    # Create trainer (inference mode)
    print("\n Creating Trainer...")
    trainer = L.Trainer(
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        callbacks=[],
        logger=False,
    )

    # Run predictions
    print("\n Running predictions...")
    results = trainer.predict(
        model=lit_module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        return_predictions=True,
    )

    print(f"   Predictions completed: {len(results)} samples")

    # Save raw results
    torch.save(results, output_dir / "results.pt")
    print(f"   Saved results to: {output_dir / 'results.pt'}")

    # Aggregate predictions and targets
    y_preds = []
    y_trues = []
    for result in results:
        y_pred = result['y_pred'].detach().cpu()
        y_true = result['y_true'].detach().cpu()
        y_preds.append(y_pred)
        y_trues.append(y_true)

    y_pred_all = torch.cat([t.flatten() for t in y_preds])
    y_true_all = torch.cat([t.flatten() for t in y_trues])

    # Compute metrics
    print("\n" + "=" * 60)
    print(" Metrics")
    print("=" * 60)

    mse = mse_loss(y_pred_all, y_true_all)
    rmse = rmse_loss(y_pred_all, y_true_all)
    r2 = torchmetrics.functional.r2_score(y_pred_all, y_true_all)

    print(f"   MSE:  {mse:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   R2:   {r2:.6f}")

    # Save metrics
    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"MSE: {mse}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write(f"R2: {r2}\n")
        f.write(f"\nCheckpoint: {ckpt_path}\n")
        f.write(f"Samples: {len(results)}\n")
    print(f"   Saved metrics to: {metrics_path}")

    # Generate visualizations
    if cfg.visualization.get('enabled', False):
        print("\n Generating visualizations...")
        viz_cfg = cfg.visualization

        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Generate comparison plots for each sample
        if viz_cfg.get('plot_comparison', {}).get('enabled', True):
            print("   Creating 3D comparison plots...")
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
                    vmin=viz_cfg.get('vmin', -1.0),
                    vmax=viz_cfg.get('vmax', 1.0),
                    error_vmax=viz_cfg.get('error_vmax', 0.5),
                )
            print(f"   Saved {len(results)} comparison plots to: {plots_dir}")

        # Generate aggregate scatter plot
        if viz_cfg.get('plot_scatter', {}).get('enabled', True):
            print("   Creating scatter plot...")
            scatter_path = output_dir / "scatter_all.png"
            plot_scatter(
                pred=y_pred_all,
                gt=y_true_all,
                title="All Samples",
                save_path=str(scatter_path),
            )
            print(f"   Saved scatter plot to: {scatter_path}")

    print("\n" + "=" * 60)
    print(" Prediction complete!")
    print("=" * 60)

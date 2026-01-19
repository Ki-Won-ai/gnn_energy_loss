"""
CLI entry point for HDWIA task.

HDWIA: Heat Diffusion with Weak Input Analyzer - Temperature prediction task.

Usage:
    uv run python -m tasks.hdwia prepare
    uv run python -m tasks.hdwia train
    uv run python -m tasks.hdwia train --config train-mini
    uv run python -m tasks.hdwia predict --model-checkpoint-path outputs/best.ckpt
"""

from pathlib import Path

import click
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

load_dotenv()


def load_hydra_config(config_name: str, overrides: tuple):
    """Load Hydra config with composition and overrides."""
    config_dir = Path(__file__).parent / "configs"
    with initialize_config_dir(
        config_dir=str(config_dir.absolute()),
        version_base=None
    ):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


@click.group()
def cli():
    """HDWIA benchmark task CLI."""
    pass


@cli.command()
@click.option('--config', '-c', default='prepare', help='Config name')
@click.option('--overrides', '-o', multiple=True, help='Config overrides')
def prepare(config: str, overrides: tuple):
    """
    Prepare dataset for training (H5 -> NPZ conversion).

    Examples:
        uv run python -m tasks.hdwia prepare
        uv run python -m tasks.hdwia prepare -o num_workers=16
    """
    cfg = load_hydra_config(config, overrides)

    from .preparation import run_preparation
    run_preparation(cfg)


@cli.command()
@click.option('--config', '-c', default='train', help='Config name')
@click.option('--overrides', '-o', multiple=True, help='Config overrides')
def train(config: str, overrides: tuple):
    """
    Train model on HDWIA benchmark.

    Examples:
        uv run python -m tasks.hdwia train
        uv run python -m tasks.hdwia train --config train-mini
        uv run python -m tasks.hdwia train -o model=gcn
        uv run python -m tasks.hdwia train -o training.max_epochs=200
    """
    cfg = load_hydra_config(config, overrides)

    from .training import run_training
    run_training(cfg)


@cli.command()
@click.option('--config', '-c', default='predict', help='Config name')
@click.option('--overrides', '-o', multiple=True, help='Config overrides')
@click.option('--model-checkpoint-path', '-m', type=str, default=None,
              help='Path to model checkpoint (overrides config)')
def predict(config: str, overrides: tuple, model_checkpoint_path: str):
    """
    Run predictions and generate visualizations.

    Examples:
        uv run python -m tasks.hdwia predict -m outputs/best.ckpt
        uv run python -m tasks.hdwia predict -m outputs/best.ckpt -o visualization.enabled=false
        uv run python -m tasks.hdwia predict -m outputs/best.ckpt -o output_dir=outputs/my-predictions
    """
    # Add checkpoint path to overrides if provided
    if model_checkpoint_path:
        overrides = overrides + (f'model_checkpoint_path={model_checkpoint_path}',)

    cfg = load_hydra_config(config, overrides)

    from .predict import run_predict
    run_predict(cfg)


@cli.command()
@click.option('--config', '-c', default='visualize', help='Config name')
@click.option('--overrides', '-o', multiple=True, help='Config overrides')
@click.option('--results-path', '-r', type=str, default=None,
              help='Path to results.pt from predict (overrides config)')
def visualize(config: str, overrides: tuple, results_path: str):
    """
    Generate visualizations from saved prediction results.

    Examples:
        uv run python -m tasks.hdwia visualize -r outputs/predictions/results.pt
        uv run python -m tasks.hdwia visualize -r outputs/predictions/results.pt -o visualization.vmin=-2
        uv run python -m tasks.hdwia visualize -r outputs/predictions/results.pt -o visualization.error_vmax=1.0
    """
    # Add results path to overrides if provided
    if results_path:
        overrides = overrides + (f'results_path={results_path}',)

    cfg = load_hydra_config(config, overrides)

    from .visualize import run_visualize
    run_visualize(cfg)


if __name__ == '__main__':
    cli()

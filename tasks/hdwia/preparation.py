"""
Data preparation module for HDWIA task.

PLACEHOLDER: Not yet implemented.

When ready to prepare HDWIA data, convert H5 files to NPZ format following
the foundation-research pattern (shared.io_manager, metadata.parquet, NPZ cache).

See tasks/deepjeb/preparation.py for reference implementation.
"""

from omegaconf import DictConfig


def run_preparation(cfg: DictConfig) -> None:
    """
    Placeholder for HDWIA data preparation.

    Not yet implemented. When ready, convert H5 files to NPZ format
    following foundation-research pattern.

    Args:
        cfg: Hydra config with preparation settings

    Raises:
        NotImplementedError: Always raised - placeholder only
    """
    raise NotImplementedError(
        "HDWIA preparation not yet implemented.\n"
        "\n"
        "To implement:\n"
        "1. Update this module to load H5 files\n"
        "2. Extract node features (geometry, physics)\n"
        "3. Create PyG Data objects\n"
        "4. Save as NPZ cache with metadata.parquet\n"
        "5. Generate io_info.pkl for normalization\n"
        "\n"
        "Reference: tasks/deepjeb/preparation.py\n"
    )

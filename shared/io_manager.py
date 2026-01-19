"""
Simplified IO Manager for feature normalization.

Handles basic z-score normalization for training data. Stores statistics
for denormalization during inference.
"""

from typing import Dict, List

import torch

EPSILON = 1e-8


class IOManager:
    """
    Manages feature normalization for ML pipelines.

    This simplified version supports:
    - Z-score normalization (mean=0, std=1)
    - Range normalization [min, max]
    - No normalization (passthrough)

    IO info is a dictionary with:
    - 'norm_type': normalization type
    - 'center': center for normalization
    - 'scale': scale for normalization
    - 'original_stats': original statistics

    Normalization: (feature_data - center) / scale
    Denormalization: normalized_data * scale + center

    Example:
        >>> io_manager = IOManager()
        >>> io_manager.collect_feature_statistics(data_list, ['pos', 'targets'])
        >>> io_manager.set_zscore_normalization('targets')
        >>> normalized = io_manager.normalize(data, 'targets')
    """

    def __init__(self):
        self.io_info = {}
        self.feature_stats = {}

    def collect_feature_statistics(
        self,
        data_list: List[Dict],
        feature_keys: List[str]
    ) -> Dict:
        """
        Collect statistics (mean, std, min, max) from all training samples.

        Args:
            data_list: List of data dictionaries from training samples
            feature_keys: List of feature keys to collect statistics for

        Returns:
            Dictionary containing statistics for each feature
        """
        stats = {}

        for feature_key in feature_keys:
            tensor_list = []
            for data in data_list:
                if feature_key in data:
                    feature_data = data[feature_key]
                    tensor_list.append(feature_data)

            if tensor_list:
                # Efficient concatenation and flattening
                all_values = torch.cat(
                    [t.flatten().float() for t in tensor_list], dim=0
                )

                stats[feature_key] = {
                    "mean": torch.mean(all_values).item(),
                    "std": torch.std(all_values).item(),
                    "min": torch.min(all_values).item(),
                    "max": torch.max(all_values).item(),
                }

        self.feature_stats = stats
        return stats

    def set_zscore_normalization(self, feature_key: str) -> None:
        """
        Set z-score normalization for a feature (mean=0, std=1).

        Args:
            feature_key: Feature key to normalize
        """
        if feature_key not in self.feature_stats:
            raise ValueError(f"Statistics not collected for {feature_key}")

        stats = self.feature_stats[feature_key]
        original_std = stats["std"]

        if original_std <= EPSILON:
            # Feature is constant, skip normalization
            self.set_no_normalization(feature_key)
            return

        self.io_info[feature_key] = {
            "norm_type": "zscore",
            "center": stats["mean"],
            "scale": original_std,
            "original_stats": stats,
        }

    def set_range_normalization(
        self, feature_key: str, target_min: float = 0.0, target_max: float = 1.0
    ) -> None:
        """
        Set range normalization to [target_min, target_max].

        Args:
            feature_key: Feature key to normalize
            target_min: Target minimum value
            target_max: Target maximum value
        """
        if feature_key not in self.feature_stats:
            raise ValueError(f"Statistics not collected for {feature_key}")

        stats = self.feature_stats[feature_key]
        original_range = stats["max"] - stats["min"]
        target_range = target_max - target_min

        if original_range <= EPSILON:
            # Feature is constant, skip normalization
            self.set_no_normalization(feature_key)
            return

        scale = original_range / target_range
        target_mid = (target_min + target_max) / 2
        center = (stats["min"] + stats["max"]) / 2 - target_mid * scale

        self.io_info[feature_key] = {
            "norm_type": f"range[{target_min},{target_max}]",
            "center": center,
            "scale": scale,
            "original_stats": stats,
        }

    def set_no_normalization(self, feature_key: str) -> None:
        """
        Set no normalization for a feature (passthrough).

        Args:
            feature_key: Feature key
        """
        self.io_info[feature_key] = {
            "norm_type": "none",
            "center": 0.0,
            "scale": 1.0,
            "original_stats": self.feature_stats.get(feature_key, {}),
        }

    def normalize(
        self, feature_data: torch.Tensor, feature_key: str
    ) -> torch.Tensor:
        """
        Apply normalization to feature data.

        Args:
            feature_data: Feature tensor to normalize
            feature_key: Feature key

        Returns:
            Normalized feature tensor
        """
        if feature_key not in self.io_info:
            return feature_data

        info = self.io_info[feature_key]
        center = info["center"]
        scale = info["scale"]
        return (feature_data - center) / scale

    def denormalize(
        self, normalized_data: torch.Tensor, feature_key: str
    ) -> torch.Tensor:
        """
        Convert normalized data back to original scale.

        Args:
            normalized_data: Normalized tensor
            feature_key: Feature key

        Returns:
            Denormalized tensor
        """
        if feature_key not in self.io_info:
            return normalized_data

        info = self.io_info[feature_key]
        scale = info["scale"]
        center = info["center"]

        return normalized_data * scale + center

    def get_io_info(self) -> Dict:
        """Get complete IO information for saving with model."""
        return self.io_info

    def load_io_info(self, info: Dict) -> None:
        """Load IO information from saved model."""
        self.io_info = info
        self.feature_stats = {}
        for feature_key in info.keys():
            self.feature_stats[feature_key] = info[feature_key].get("original_stats", {})

"""
Simplified collators for PyG data batching.

This module provides collator classes for converting lists of PyG Data objects
into batched Data objects with preprocessing.
"""

import torch
import torch_geometric.data as pyg_data
from torch_geometric.data import Batch


class SimpleCollator:
    """
    Simple collator for GNN batching.

    Prepares node features by concatenating pos, curvatures, and norms,
    and creates batched PyG Data objects.

    Processing per sample:
        1. Compute curvature-weighted normals: curvatures * norms (N, 3)
        2. Concatenate node features: [pos, curvature_weighted_norms, norms] -> (N, 9)
        3. Rename 'targets' to 'y' for Lightning compatibility

    Example:
        >>> collator = SimpleCollator()
        >>> batch = collator(samples)
        >>> print(batch.x.shape)  # (total_nodes, 9)
    """

    @torch.no_grad()
    def collate(self, samples: list[pyg_data.Data]) -> pyg_data.Data:
        """
        Collate a list of Data objects into a single batched Data object.

        Args:
            samples: List of PyG Data objects with pos, curvatures, norms, targets

        Returns:
            Batched PyG Data object with:
            - x: Node features (N, 9) = [pos, curvature_weighted_norms, norms]
            - y: Target values
            - edge_index: Edge indices
            - batch: Batch assignment for nodes
        """
        for sample in samples:
            # Curvature-weighted normals: (N, 1) * (N, 3) -> (N, 3)
            curvatures = sample.curvatures
            if sample.curvatures.ndim == 1 or sample.curvatures.shape[1] == 1:
                curvatures = curvatures.view(-1, 1) * sample.norms

            # Concatenate node features: [pos, curvature_weighted_norms, norms]
            sample.x = torch.cat(
                [sample.pos, curvatures, sample.norms], dim=-1
            )
            assert sample.x.shape[1] == 9, (
                f"Expected 9 features, got {sample.x.shape[1]}. "
                f"pos={sample.pos.shape}, curvatures={sample.curvatures.shape}, "
                f"norms={sample.norms.shape}"
            )

            # Rename targets to y for consistency
            sample.y = sample.targets
            assert sample.y.shape[0] == sample.x.shape[0], (
                f"Expected {sample.x.shape[0]} targets, got {sample.y.shape[0]}"
            )

        # Batch all samples into a single graph
        batch = Batch.from_data_list(samples)

        return batch

    def __call__(self, samples: list[pyg_data.Data]) -> pyg_data.Data:
        """Callable interface for DataLoader collate_fn."""
        return self.collate(samples)

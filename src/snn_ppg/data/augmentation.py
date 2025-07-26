"""
augmentation.py

Defines PPGDatasetExtremeAug, which oversamples extreme systolic and diastolic
values to address imbalance in regression tasks by repeating samples beyond
quantile thresholds.
"""
from typing import Tuple

import torch
from torch import Tensor

from .dataset import PPGDataset

class PPGDatasetExtremeAug(PPGDataset):
    """
    Oversamples extreme SP/DP cases for PPG regression tasks.

    Parameters
    ----------
    signals : Tensor
        PPG signal data of shape (N, T).
    targets : Tensor
        SP and DP targets of shape (N, 2).
    extreme_quantile : float
        Lower quantile (e.g., 0.05) defining extremes; samples below this or
        above 1 - quantile are deemed extreme.
    n_aug_per_extreme : int
        Number of extra repetitions for each extreme sample (beyond the base copy).
    """
    def __init__(
        self,
        signals: Tensor,
        targets: Tensor,
        extreme_quantile: float,
        n_aug_per_extreme: int
    ):
        """
        Initialize PPGDatasetExtremeAug and compute oversampled indices.

        The dataset will include all normal samples once, and each extreme sample
        will be repeated (n_aug_per_extreme + 1) times total.
        """
        super().__init__(signals, targets)
        sp_vals, dp_vals = self.targets.unbind(dim=1)

        # compute quantile thresholds for extremes
        q = torch.tensor([extreme_quantile, 1 - extreme_quantile], device=sp_vals.device)
        (sp_low, sp_high), (dp_low, dp_high) = (
            torch.quantile(vals, q) for vals in (sp_vals, dp_vals)
        )

        # boolean mask for extreme values
        extreme_mask = (
            (sp_vals <= sp_low) | (sp_vals >= sp_high) |
            (dp_vals <= dp_low) | (dp_vals >= dp_high)
        )

        # gather indices
        extreme_indices = extreme_mask.nonzero(as_tuple=False).view(-1).tolist()
        normal_indices = (~extreme_mask).nonzero(as_tuple=False).view(-1).tolist()

        # build oversampled index list
        self.indices = (
            normal_indices +
            [i for i in extreme_indices for _ in range(n_aug_per_extreme + 1)]
        )

    def __len__(self) -> int:
        """Return the total number of samples including oversampled extremes."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Retrieve the signal and target for the given (possibly repeated) index."""
        orig_idx = self.indices[idx]
        return self.signals[orig_idx], self.targets[orig_idx]

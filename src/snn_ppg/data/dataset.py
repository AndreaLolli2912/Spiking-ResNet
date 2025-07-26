"""
dataset.py

Module for PPGDataset: simple pairing of PPG signals and SP/DP targets.
"""
from typing import Tuple

from torch import Tensor
from torch.utils.data import Dataset

class PPGDataset(Dataset):
    """Simple feature/label pairing for PPG regression."""
    def __init__(self, signals: Tensor, targets: Tensor):
        """
        Parameters
        ----------
        signals : Tensor
            Shape (N, T), PPG signal data.
        targets : Tensor
            Shape (N, 2), systolic and diastolic values.
        """
        self.signals = signals   # (N, T)
        self.targets = targets   # (N, 2)

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Retrieve the signal and target at the specified index."""
        return self.signals[idx], self.targets[idx]

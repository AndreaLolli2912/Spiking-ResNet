"""
Prediction utilities: inference over a DataLoader.
"""
from typing import Optional, Tuple, Any
import numpy as np
import torch
from torch import nn
from snntorch import spikegen
from omegaconf import DictConfig

def predict_on_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
    target_scaler: Optional[Any],
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on data from a DataLoader, returning true and predicted arrays.

    Args:
        model: trained spiking model
        loader: DataLoader yielding (signals, targets)
        cfg: configuration with 'training.device' and 'model.encoding' / 'model.n_steps'
        target_scaler: Optional scaler to inverse-transform outputs

    Returns:
        (y_true, y_pred): two NumPy arrays of shape (N, 2) for SBP/DBP
    """
    model.eval()

    trues = []
    preds = []

    with torch.no_grad():
        for batch, labels in loader:
            x = batch.to(device).float()
            y = labels.to(device).float()
            # apply encoding if needed
            if cfg.model.encoding:
                x = spikegen.rate(data=x.unsqueeze(1), num_steps=cfg.model.n_steps)
            else:
                x = x.unsqueeze(1)

            out = model(x)
            # take final spiking head output
            if isinstance(out, (list, tuple)):
                out = out[-1]

            y_np = y.cpu().numpy()
            out_np = out.cpu().numpy()

            if target_scaler:
                y_np = target_scaler.inverse_transform(y_np)
                out_np = target_scaler.inverse_transform(out_np)

            trues.append(y_np)
            preds.append(out_np)

    return np.vstack(trues), np.vstack(preds)

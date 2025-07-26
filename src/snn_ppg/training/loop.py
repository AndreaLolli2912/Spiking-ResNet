"""
Training loops: per-epoch training and validation using repository config.
"""
import argparse
from typing import Optional, Any, Tuple
from dataclasses import dataclass
import torch
from torch import nn, optim
from snntorch import spikegen
from omegaconf import DictConfig

@dataclass
class LoopConfig:
    """
    Bundles arguments for train/validate loops.
    """
    loader: torch.utils.data.DataLoader
    loss_fn: nn.Module
    optimizer: optim.Optimizer
    scheduler: Optional[optim.lr_scheduler._LRScheduler]
    cfg: DictConfig
    args: argparse.Namespace
    target_scaler: Optional[Any]
    device: torch.device


def train_one_epoch(
    model: nn.Module,
    state: LoopConfig
) -> Tuple[float, float, float]:
    """
    Run one training epoch and compute average loss, SBP MAE, and DBP MAE.
    """
    model.train()
    total_loss = sbp_mae = dbp_mae = 0.0

    for batch, labels in state.loader:
        x = batch.to(state.device).float()
        y = labels.to(state.device).float()
        if state.cfg.model.encoding:
            x = spikegen.rate(data=x.unsqueeze(1), num_steps=state.cfg.model.n_steps)
        else:
            x = x.unsqueeze(1)

        state.optimizer.zero_grad()
        yp = model(x)[-1]
        loss = state.loss_fn(yp, y)
        loss.backward()
        state.optimizer.step()
        if state.scheduler:
            state.scheduler.step()

        bsz = x.size(0)
        total_loss += loss.item() * bsz

        if state.target_scaler:
            yt = torch.from_numpy(
                state.target_scaler.inverse_transform(y.cpu().numpy())
                ).to(state.device)
            yh = torch.from_numpy(
                state.target_scaler.inverse_transform(yp.detach().cpu().numpy())
                ).to(state.device)
        else:
            yt, yh = y, yp

        sbp_mae += torch.abs(yh[:, 0] - yt[:, 0]).sum().item()
        dbp_mae += torch.abs(yh[:, 1] - yt[:, 1]).sum().item()

    return total_loss / len(state.loader.dataset), \
            sbp_mae / len(state.loader.dataset), \
            dbp_mae / len(state.loader.dataset)


def validate_one_epoch(
    model: nn.Module,
    state: LoopConfig
) -> Tuple[float, float, float]:
    """
    Run one validation epoch and compute average loss, SBP MAE, and DBP MAE.
    """
    model.eval()
    total_loss = sbp_mae = dbp_mae = 0.0

    with torch.no_grad():
        for batch, labels in state.loader:
            x = batch.to(state.device).float()
            y = labels.to(state.device).float()
            if state.cfg.model.encoding:
                x = spikegen.rate(data=x.unsqueeze(1), num_steps=state.cfg.model.n_steps)
            else:
                x = x.unsqueeze(1)

            yp = model(x)[-1]
            loss = state.loss_fn(yp, y)

            bsz = x.size(0)
            total_loss += loss.item() * bsz

            if state.target_scaler:
                yt = torch.from_numpy(
                    state.target_scaler.inverse_transform(y.cpu().numpy())
                    ).to(state.device)
                yh = torch.from_numpy(
                    state.target_scaler.inverse_transform(yp.cpu().numpy())
                    ).to(state.device)
            else:
                yt, yh = y, yp

            sbp_mae += torch.abs(yh[:, 0] - yt[:, 0]).sum().item()
            dbp_mae += torch.abs(yh[:, 1] - yt[:, 1]).sum().item()

    return total_loss / len(state.loader.dataset), \
            sbp_mae / len(state.loader.dataset), \
            dbp_mae / len(state.loader.dataset)

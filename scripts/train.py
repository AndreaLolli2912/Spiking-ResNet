"""
train.py

Entry point for training the SNN-PPG model as a module invocation (python -m scripts.train).
Loads and merges YAML configs via OmegaConf, sets up logging, then runs single fold or k-fold.
"""
import argparse
import logging
from omegaconf import DictConfig
from snn_ppg.training.runner import run_kfold

logger = logging.getLogger(__name__)

def train_entry(
        args: argparse.Namespace,
        cfg: DictConfig,
) -> None:
    """
    Train entry
    """
    results, y_true_all, y_pred_all = run_kfold(
        args,
        cfg
    )

    logger.info(
        "K-Fold results: SP MAE %d | DP MAE %d",
        abs(y_true_all[:, 0] - y_pred_all[:, 0]).mean(),
        abs(y_true_all[:, 1] - y_pred_all[:, 1]).mean()
    )

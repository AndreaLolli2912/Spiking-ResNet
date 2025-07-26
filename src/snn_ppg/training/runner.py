"""
Fold and K-Fold runner utilities for training and evaluation.
"""
import argparse
import json
from pathlib import Path
import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from omegaconf import DictConfig

from snn_ppg.data.data import make_data_loaders
from snn_ppg.models.registry import get_model
from snn_ppg.training.loop import LoopConfig, train_one_epoch, validate_one_epoch
from snn_ppg.training.predict import predict_on_loader

logger = logging.getLogger(__name__)


def _prepare_fold(
    args: argparse.Namespace,
    cfg: DictConfig,
    fold_idx: int,
    device: str
):
    """
    Prepare data loaders, model, loss, optimizer, scheduler, and target scaler.
    """
    # configure fold index
    train_loader, val_loader, _, target_scaler = make_data_loaders(args, cfg, fold_idx)

    # instantiate model
    model = get_model(cfg.model.name, cfg).to(device)

    # loss and optimizer
    loss_fn = nn.SmoothL1Loss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay
    )

    # scheduler
    total_steps = cfg.training.n_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.training.scheduler.lr,
        total_steps=total_steps,
        div_factor=cfg.training.scheduler.div_factor,
        pct_start=cfg.training.scheduler.pct_start,
        final_div_factor=cfg.training.scheduler.final_div_factor
    )

    return train_loader, val_loader, model, loss_fn, optimizer, scheduler, target_scaler


def run_fold(
    args: argparse.Namespace,
    cfg: DictConfig,
    fold_idx: int,
) -> tuple[float, float, dict, np.ndarray, np.ndarray]:
    """
    Run training and validation for a single fold.

    Returns:
        sbp_mae, dbp_mae, history, y_true, y_pred
    """
    logger.info("Starting fold %d", fold_idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup fold
    train_loader, val_loader, model, loss_fn, optimizer, scheduler, target_scaler = \
        _prepare_fold(args, cfg, fold_idx, device)

    history = {
        key: [] for key in (
        'train_loss','val_loss','train_sbp','val_sbp','train_dbp','val_dbp'
        )
    }
    best_val = float('inf')
    no_improve = 0
    patience = cfg.training.early_stopping.patience
    delta = cfg.training.early_stopping.delta

    vl_sbp = vl_dbp = None

    # training epochs
    for epoch in range(1, cfg.training.n_epochs + 1):
        # train
        train_state = LoopConfig(
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            args=args,
            target_scaler=target_scaler,
            device=device
        )
        tr_loss, tr_sbp, tr_dbp = train_one_epoch(model, train_state)

        # validate
        val_state = LoopConfig(
            loader=val_loader,
            loss_fn=loss_fn,
            optimizer=None,
            scheduler=None,
            cfg=cfg,
            args=args,
            target_scaler=target_scaler,
            device=device
        )
        vl_loss, vl_sbp, vl_dbp = validate_one_epoch(model, val_state)

        # record history
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['train_sbp'].append(tr_sbp)
        history['val_sbp'].append(vl_sbp)
        history['train_dbp'].append(tr_dbp)
        history['val_dbp'].append(vl_dbp)

        # logging
        if epoch % cfg.training.log_interval == 0:
            logger.info(
                "Epoch %d/%d L=%.4f/%.4f SBP=%.3f/%.3f DBP=%.3f/%.3f",
                epoch,
                cfg.training.n_epochs,
                tr_loss,
                vl_loss,
                tr_sbp,
                vl_sbp,
                tr_dbp,
                vl_dbp,
            )

        # early stopping
        if vl_loss < best_val - delta:
            best_val = vl_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                model.save_checkpoint(
                    Path(args.output_dir) / "model_weights" / f"best_model_fold{fold_idx}.pt"
                    )
                logger.info("Early stopping fold %d at epoch %d", fold_idx, epoch)
                break

    model.save_checkpoint(Path(args.output_dir) / "model_weights" / f"best_model_fold{fold_idx}.pt")

    # inference
    y_true, y_pred = predict_on_loader(model, val_loader, cfg, target_scaler, device)
    return vl_sbp, vl_dbp, history, y_true, y_pred


def run_kfold(
    args: argparse.Namespace,
    cfg: DictConfig
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray]:
    """
    Run K-Fold cross-validation.

    Returns:
        results, all_y_true, all_y_pred
    """

    results = []
    histories = {}
    all_trues, all_preds = [], []

    for f in range(cfg.dataset[args.dataset].n_folds):
        sbp_mae, dbp_mae, hist, y_t, y_p = run_fold(args, cfg, fold_idx=f) 
        logger.info("Fold %d result: SBP MAE=%.3f, DBP MAE=%.3f", f+1, sbp_mae, dbp_mae)

        results.append((sbp_mae, dbp_mae)) #NOTE: questo results si può gestire meglio
        histories[f] = hist
        all_trues.append(y_t)
        all_preds.append(y_p)

    # summary
    sbps = [r[0] for r in results]
    dbps = [r[1] for r in results]
    logger.info(
        "K-Fold Summary (%d folds) SBP MAE = %.3f ± %.3f, DBP MAE = %.3f ± %.3f",
        cfg.dataset[args.dataset].n_folds,
        np.mean(sbps),
        np.std(sbps, ddof=1),
        np.mean(dbps),
        np.std(dbps, ddof=1),
    )

    # stack results
    y_true_all = np.vstack(all_trues)
    y_pred_all = np.vstack(all_preds)

    # save logs
    log_dir = Path(cfg.output_dir) / "kfold_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "histories.json", "w", encoding="utf-8") as fp:
        json.dump(histories, fp, indent=2)

    return results, y_true_all, y_pred_all

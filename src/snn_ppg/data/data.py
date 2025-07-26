"""
data.py

Create DataLoaders: raw signal loading, optional scaling, dataset instantiation, and augmentation.
"""
import argparse
from typing import Tuple, Optional, Any, Dict

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .augmentation import PPGDatasetExtremeAug
from .dataset import PPGDataset
from .signal_loader import load_raw_signals

_SCALER_REGISTRY = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler":   MinMaxScaler,
}

def _make_scaler(name: str):
    """
    Lookup and return a scaler instance for the given name.

    Parameters
    ----------
    name : str
        Key for the scaler in the _SCALER_REGISTRY.

    Returns
    -------
    scaler : object
        An instance of the requested scaler.
    """
    try:
        return _SCALER_REGISTRY[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown scaler '{name}'. Valid options: {list(_SCALER_REGISTRY)}"
        ) from exc

def _scale_data(
    train_sig: Tensor,
    eval_sig: Tensor,
    train_tgt: Tensor,
    eval_tgt: Tensor,
    cfg: Dict
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Any], Optional[Any]]:
    """
    Apply feature and/or target scaling based on `cfg.preprocessing`.

    Parameters
    ----------
    train_sig : Tensor
        Training signals of shape (N_train, T).
    eval_sig : Tensor
        Validation signals of shape (N_eval, T).
    train_tgt : Tensor
        Training targets of shape (N_train, 2).
    eval_tgt : Tensor
        Validation targets of shape (N_eval, 2).
    cfg : Dict
        Configuration dict containing a `preprocessing` section with keys:
        - feature_scaling : bool
        - feature_scaler : str
        - target_scaling : bool
        - target_scaler : str

    Returns
    -------
    train_sig : Tensor
        Scaled training signals.
    eval_sig : Tensor
        Scaled validation signals.
    train_tgt : Tensor
        Scaled training targets.
    eval_tgt : Tensor
        Scaled validation targets.
    fs : object or None
        Fitted feature scaler instance, or None if not applied.
    ts : object or None
        Fitted target scaler instance, or None if not applied.
    """
    fs: Optional[Any] = None
    ts: Optional[Any] = None

    # Feature scaling
    if cfg.preprocessing.get("feature_scaling", False):
        scaler = _make_scaler(cfg.preprocessing.feature_scaler)
        fs = scaler.fit(train_sig.numpy())
        train_sig = torch.from_numpy(fs.transform(train_sig.numpy())).float()
        eval_sig = torch.from_numpy(fs.transform(eval_sig.numpy())).float()

    # Target scaling
    if cfg.preprocessing.get("target_scaling", False):
        scaler = _make_scaler(cfg.preprocessing.target_scaler)
        ts = StandardScaler().fit(train_tgt.numpy())
        train_tgt = torch.from_numpy(ts.transform(train_tgt.numpy())).float()
        eval_tgt = torch.from_numpy(ts.transform(eval_tgt.numpy())).float()

    return train_sig, eval_sig, train_tgt, eval_tgt, fs, ts


def make_data_loaders(
    args: argparse.Namespace,
    cfg: Dict,
    fold_idx: int
) -> Tuple[DataLoader, DataLoader, Optional[Any], Optional[Any]]:
    """
    Orchestrate signal loading, scaling, dataset creation, and DataLoader setup.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments, must include `args.dataset`.
    cfg : Dict
        Configuration dict with `dataset` and `training` sections.
    fold_idx : int
        Index of the validation fold.

    Returns
    -------
    train_loader : DataLoader
        DataLoader for the training dataset.
    val_loader : DataLoader
        DataLoader for the validation dataset.
    fs : object or None
        Fitted feature scaler, or None if not applied.
    ts : object or None
        Fitted target scaler, or None if not applied.
    """
    ds_cfg = cfg.dataset.get(args.dataset, {})
    # 1) load raw data
    train_sig, train_tgt, eval_sig, eval_tgt = load_raw_signals(
        cfg,
        dataset_name=args.dataset,
        val_idx=fold_idx
    )

    # 2) apply scaling when requested
    train_sig, eval_sig, train_tgt, eval_tgt, fs, ts = _scale_data(
        train_sig, eval_sig, train_tgt, eval_tgt, cfg
    )

    # 3) build Dataset instances
    if ds_cfg.get('extreme_aug', False):
        train_ds: Dataset = PPGDatasetExtremeAug(
            train_sig,
            train_tgt,
            extreme_quantile=ds_cfg.extreme_quantile,
            n_aug_per_extreme=ds_cfg.n_aug_per_extreme
        )
    else:
        train_ds: Dataset = PPGDataset(train_sig, train_tgt)
    val_ds: Dataset = PPGDataset(eval_sig, eval_tgt)

    # 4) create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, fs, ts

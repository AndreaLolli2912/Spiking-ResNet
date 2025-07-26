"""
raw_loader.py

Load raw PPG signals from .mat files and split into train/eval folds.
"""
import glob
import os
from typing import Tuple, List, Dict

import scipy.io as sio
import torch
from torch import Tensor


def _load_mat_file(fp: str) -> Dict[str, Tensor]:
    """
    Load a single .mat file and extract SP, DP, and PPG signal arrays.

    Parameters
    ----------
    fp : str
        Path to a .mat file containing 'SP', 'DP', and 'signal' keys.

    Returns
    -------
    Dict[str, Tensor]
        Dictionary with:
        - 'sp': Tensor of systolic pressure values.
        - 'dp': Tensor of diastolic pressure values.
        - 'signal': Tensor of PPG signal data.
    """
    raw = sio.loadmat(fp)
    return {
        'sp': torch.from_numpy(raw['SP']).float().squeeze(),
        'dp': torch.from_numpy(raw['DP']).float().squeeze(),
        'signal': torch.from_numpy(raw['signal']).float().squeeze()
    }


def _collect_folds(
    files: List[str],
    val_idx: int
) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
    """
    Split .mat files into training and evaluation sets by fold index.

    Parameters
    ----------
    files : List[str]
        List of file paths to .mat signal files.
    val_idx : int
        Fold index to designate evaluation files (filename containing 'fold_{val_idx}').

    Returns
    -------
    Tuple of six lists:
        - train_sp: List[Tensor] of training systolic values.
        - train_dp: List[Tensor] of training diastolic values.
        - train_sig: List[Tensor] of training PPG signals.
        - eval_sp:  List[Tensor] of eval systolic values.
        - eval_dp:  List[Tensor] of eval diastolic values.
        - eval_sig: List[Tensor] of eval PPG signals.
    """
    train_sp, train_dp, train_sig = [], [], []
    eval_sp, eval_dp, eval_sig = [], [], []

    for fp in sorted(files):
        data = _load_mat_file(fp)
        if f"fold_{val_idx}" in os.path.basename(fp):
            eval_sp.append(data['sp'])
            eval_dp.append(data['dp'])
            eval_sig.append(data['signal'])
        else:
            train_sp.append(data['sp'])
            train_dp.append(data['dp'])
            train_sig.append(data['signal'])

    if not train_sp or not eval_sp:
        raise RuntimeError("Train or eval fold not found in provided files")

    return train_sp, train_dp, train_sig, eval_sp, eval_dp, eval_sig


def load_raw_signals(
    cfg: Dict,
    dataset_name: str,
    val_idx: int = 0
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Read .mat files, split into train/eval folds, and stack into Tensors.

    Parameters
    ----------
    cfg : DictConfig
        Configuration containing dataset paths under cfg.dataset[dataset_name].paths.signal_folds.
    dataset_name : str
        Key in cfg.dataset specifying which dataset to load.
    val_idx : int, optional
        Fold index for evaluation split (default is 0).

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor]
        - train_signals: Tensor of shape (N_train, signal_length).
        - train_targets: Tensor of shape (N_train, 2) for SP and DP.
        - eval_signals:  Tensor of shape (N_eval,  signal_length).
        - eval_targets:  Tensor of shape (N_eval, 2).
    """
    signal_dir = cfg.dataset[dataset_name].paths.signal_folds
    files = glob.glob(os.path.join(signal_dir, '*.mat'))
    # filter files which contain mabp in the name
    files = [fp for fp in files if 'mabp' not in os.path.basename(fp)]
    train_sp, train_dp, train_sig, eval_sp, eval_dp, eval_sig = _collect_folds(files, val_idx)

    # Stack lists into tensors
    train_signals = torch.vstack(train_sig)
    eval_signals = torch.vstack(eval_sig)
    train_targets = torch.stack([torch.hstack(train_sp), torch.hstack(train_dp)], dim=1)
    eval_targets = torch.stack([torch.hstack(eval_sp), torch.hstack(eval_dp)], dim=1)
    
    return train_signals, train_targets, eval_signals, eval_targets

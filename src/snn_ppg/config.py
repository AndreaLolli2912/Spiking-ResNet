"""
config.py

Loads & merges dataset/model/train YAMLs via OmegaConf,
injecting infra args and then applying any leftover Hydra‐style overrides.
"""
from pathlib import Path
from typing import List, Optional
import os

from omegaconf import OmegaConf, DictConfig

def load_cfg(
    infra_args: Optional[object] = None,
    leftover_cli: Optional[List[str]] = None,
) -> DictConfig:
    
    # decide where your YAML lives
    if infra_args and getattr(infra_args, "config_dir", None):
        config_dir = Path(infra_args.config_dir)
    else:
        config_dir = (Path(__file__).parents[2] / "configs").resolve()

    ds = OmegaConf.load(os.path.join(config_dir, "dataset.yaml"))
    md = OmegaConf.load(os.path.join(config_dir, "model.yaml"))
    tr = OmegaConf.load(os.path.join(config_dir, "training.yaml"))
    pr = OmegaConf.load(os.path.join(config_dir, "preprocessing.yaml"))
    merged = OmegaConf.merge(ds, md, tr, pr)

    # CLI overrides: `python train.py training.use_kfold=True`
    overrides = leftover_cli or []
    cli_cfg   = OmegaConf.from_cli(overrides)
    return OmegaConf.merge(merged, OmegaConf.from_cli())

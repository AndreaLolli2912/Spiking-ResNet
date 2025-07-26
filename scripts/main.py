"""
main entry point
"""
import argparse
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from snn_ppg.cli import get_cli
from snn_ppg.config import load_cfg
from snn_ppg.logger import setup_logging
from .train import train_entry
from .test import test_entry

def main(
        args: argparse.Namespace,
        cfg: DictConfig,
) -> None:
    """
    Main training entry.
    """
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    cfg.output_dir = (
        Path(args.output_dir)
        / args.command
        / args.dataset
        / date_str
    )

    logger = setup_logging(
        output_dir= cfg.output_dir,
        level= "DEBUG" if args.verbose else "INFO",
    )

    if args.command == "train":
        train_entry(
            args,
            cfg
        )

    else: # args.command == "test"
        test_entry(
            args,
            cfg
        )

if __name__ == "__main__":

    # 1) get both infra‐args and leftover CLI overrides
    arguments, leftover_cli = get_cli()

    # 2) pass both into your config loader
    configuration = load_cfg(arguments, leftover_cli)

    # 3) dispatch
    main(arguments, configuration)

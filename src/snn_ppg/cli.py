"""
cli.py

Command-line interface parser for SNN-PPG workflows.
Supports subcommands (`train` and `test`), infrastructure flags,
and dataset selection.

Usage examples:
  python -m scripts.train train --dataset=bcg --config-dir=configs \
      --output-dir=results --seed=42 --device=cuda --verbose \
      training.lr=0.001

  python -m scripts.train test --dataset=bcg --config-dir=configs \
      --checkpoint=results/train/bcg/checkpoint.pt --batch-size=128
"""
import argparse
from typing import List, Tuple


def get_cli() -> Tuple[argparse.Namespace, List[str]]:
    """
    Build and parse command-line arguments, separating infrastructure args
    and dataset/subcommand choices from experiment hyperparameter overrides.

    Returns:
        args (Namespace): Parsed arguments, including:
            - args.command: 'train' or 'test'
            - args.dataset: specific dataset name or 'all'
            - infra flags: config-dir, output-dir, seed, device, verbose, etc.
            - subcommand-specific flags (resume, max-epochs or checkpoint, batch-size)
        leftover_cli (List[str]): List of strings for OmegaConf-style overrides
                                 (e.g. 'training.lr=0.005').
    """
    parser = argparse.ArgumentParser(
        description="SNN-PPG: train and evaluate spiking neural network on PPG data"
    )

    # Infrastructure flags
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Folder containing dataset.yaml, model.yaml, train.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Base directory to save logs, model checkpoints, and outputs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bcg",
        choices=["bcg"],  # or ['all', 'bcg', 'ppg', ...]
        help="Dataset to use ('all' or specific name)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging"
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Subcommand to execute: 'train' or 'test'",
    )

    # Train subcommand
    train_p = subparsers.add_parser(
        "train",
        help="Train the SNN model",
    )


    # Test subcommand
    test_p = subparsers.add_parser(
        "test",
        help="Evaluate the SNN model",
    )

    # Parse known args; leftover_cli holds OmegaConf overrides
    infra_args, leftover_cli = parser.parse_known_args()
    return infra_args, leftover_cli

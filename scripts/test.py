"""Test entry."""
import argparse
import logging

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def test_entry(
        args: argparse.Namespace,
        cfg: DictConfig,
) -> None:
    """Test entry"""
    pass

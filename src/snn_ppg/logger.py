"""
Logging utilities for experiments.

Provides a `setup_logging` function to configure the root logger
to emit to both console and a per-run log file.
"""
import logging
from pathlib import Path

def setup_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    """
    Configure Python’s root logger to write both to console and to a file.
    
    Parameters
    ----------
    output_dir : Path
        Directory where log file will be created.
    level : str
        Logging level (DEBUG, INFO, WARNING, etc.).

    Returns
    -------
    logger : logging.Logger  
        The configured root logger.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "experiment.log"

    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level),
        format=fmt,
        handlers=[
            logging.StreamHandler(),                 # console
            logging.FileHandler(log_path, mode="w")  # file
        ]
    )
    return logging.getLogger()

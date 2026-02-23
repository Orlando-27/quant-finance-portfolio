"""
utils.py
--------
Logging, timing decorators, and shared helper functions.
"""

import os
import logging
import time
import functools
from datetime import datetime
from pathlib import Path


def get_logger(name: str, log_dir: str = "outputs/logs",
               level: str = "INFO") -> logging.Logger:
    """
    Return a named logger writing to both stdout and a daily rotating file.

    Parameters
    ----------
    name    : Logger name (typically the module __name__).
    log_dir : Directory for log files.
    level   : Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").

    Returns
    -------
    logging.Logger
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = os.path.join(
        log_dir, f"trading_bot_{datetime.now().strftime('%Y%m%d')}.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def timeit(func):
    """Decorator that logs the execution time of any function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.debug("%s completed in %.3f s", func.__qualname__, elapsed)
        return result
    return wrapper


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a float as a USD currency string."""
    return f"${value:,.{decimals}f}"


def pct_change(old: float, new: float) -> float:
    """Safe percentage change; returns 0 if old is zero."""
    return (new - old) / old if old != 0 else 0.0


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(value, hi))

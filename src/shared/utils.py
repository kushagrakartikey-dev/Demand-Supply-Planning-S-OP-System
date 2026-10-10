"""
Cross-cutting utilities: logging, deterministic seeding, IO helpers and
lightweight timing instrumentation.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from src.planning.config import LOG_DIR, PROCESSED_DIR, RAW_DIR, ensure_directories

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_CONFIGURED = False


def configure_logging(level: int = logging.INFO, to_file: bool = True) -> None:
    """Configure root logging once, with console and optional file handlers."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    ensure_directories()

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    if to_file:
        handler = logging.FileHandler(LOG_DIR / "pipeline.log", mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(handler)

    # Third-party libraries are extremely chatty during model fitting.
    for noisy in ("cmdstanpy", "prophet", "matplotlib", "numexpr", "py4j", "shap"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed every stochastic component and return a dedicated Generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    return np.random.default_rng(seed)


@contextmanager
def timed(label: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Log the wall-clock duration of a block."""
    log = logger or get_logger("timer")
    start = time.perf_counter()
    log.info("START  %s", label)
    try:
        yield
    finally:
        log.info("DONE   %s  (%.1fs)", label, time.perf_counter() - start)


def save_csv(df: pd.DataFrame, name: str, processed: bool = False) -> Path:
    """Persist a dataframe to data/raw or data/processed and return the path."""
    ensure_directories()
    target_dir = PROCESSED_DIR if processed else RAW_DIR
    path = target_dir / name
    df.to_csv(path, index=False)
    return path


def load_csv(name: str, processed: bool = False, parse_dates: Optional[Iterable[str]] = None
             ) -> pd.DataFrame:
    """Read a dataframe previously written by :func:`save_csv`."""
    target_dir = PROCESSED_DIR if processed else RAW_DIR
    path = target_dir / name
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset '{path}' not found. Run `python run_demo.py` first."
        )
    return pd.read_csv(path, parse_dates=list(parse_dates) if parse_dates else None)


def dataset_exists(name: str, processed: bool = False) -> bool:
    target_dir = PROCESSED_DIR if processed else RAW_DIR
    return (target_dir / name).exists()


def human_number(value: float, currency: str = "") -> str:
    """Format a number using the Indian crore / lakh convention."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return "n/a"
    sign = "-" if value < 0 else ""
    value = abs(float(value))
    if value >= 1e7:
        return f"{sign}{currency}{value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"{sign}{currency}{value / 1e5:,.2f} L"
    if value >= 1e3:
        return f"{sign}{currency}{value / 1e3:,.1f} K"
    return f"{sign}{currency}{value:,.0f}"


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises and never returns inf/nan."""
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        result = numerator / denominator
        return default if (np.isnan(result) or np.isinf(result)) else float(result)
    except (TypeError, ZeroDivisionError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Constrain a value to [low, high]; NaN maps to `low`."""
    if value is None or pd.isna(value):
        return low
    return float(min(max(value, low), high))


def summarise_frame(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Compact structural summary used in run manifests and tests."""
    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "null_cells": int(df.isna().sum().sum()),
    }


def chunked(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start:start + size]

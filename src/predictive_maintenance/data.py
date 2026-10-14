from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ID_COLUMNS, SENSOR_COLUMNS, SETTING_COLUMNS

RAW_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def load_cmapss(path: str | Path) -> pd.DataFrame:
    """Load a whitespace-separated C-MAPSS file and validate its schema."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"C-MAPSS file not found: {source}. Download CMAPSSData.zip and extract train_FD001.txt."
        )
    frame = pd.read_csv(source, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
    validate_cmapss(frame)
    return frame


def validate_cmapss(frame: pd.DataFrame) -> None:
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required C-MAPSS columns: {missing}")
    if frame.empty or frame[["unit_id", "cycle"]].isna().any().any():
        raise ValueError("C-MAPSS data must contain non-empty unit_id and cycle values.")
    if (frame["cycle"] < 1).any() or (frame["unit_id"] < 1).any():
        raise ValueError("unit_id and cycle must be positive.")


def add_rul_target(frame: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """Create piecewise-linear RUL from each engine's last observed cycle."""
    validate_cmapss(frame)
    result = frame.copy()
    last_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = last_cycle - result["cycle"]
    if cap is not None:
        if cap <= 0:
            raise ValueError("RUL cap must be positive or None.")
        result["rul"] = result["rul"].clip(upper=cap)
    return result


def split_by_engine(
    frame: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete engines, preventing rows from one engine crossing partitions."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    engines = np.array(sorted(frame["unit_id"].unique()))
    if len(engines) < 2:
        raise ValueError("At least two engine units are required for a split.")
    rng = np.random.default_rng(random_state)
    rng.shuffle(engines)
    n_test = max(1, int(round(len(engines) * test_size)))
    test_engines = set(engines[:n_test])
    test = frame[frame["unit_id"].isin(test_engines)].copy()
    train = frame[~frame["unit_id"].isin(test_engines)].copy()
    return train, test


def make_demo_data(n_engines: int = 12, min_cycles: int = 35, max_cycles: int = 70) -> pd.DataFrame:
    """Create deterministic demo data for tests and pipeline smoke tests."""
    rng = np.random.default_rng(42)
    rows: list[list[float]] = []
    for unit_id in range(1, n_engines + 1):
        end = int(rng.integers(min_cycles, max_cycles + 1))
        for cycle in range(1, end + 1):
            degradation = cycle / end
            settings = [0.1 * unit_id, np.sin(cycle / 10), np.cos(cycle / 11)]
            sensors = [
                100 * degradation + rng.normal(0, 0.5),
                50 * (1 - degradation) + rng.normal(0, 0.5),
            ]
            sensors += [rng.normal(0, 1) for _ in range(19)]
            rows.append([unit_id, cycle, *settings, *sensors])
    return pd.DataFrame(rows, columns=RAW_COLUMNS)

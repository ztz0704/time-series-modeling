"""Chronological train/test splitting."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.Series
    test: pd.Series


def temporal_train_test_split(series: pd.Series, test_size: int) -> TemporalSplit:
    """Split a series without shuffling while preserving its index."""
    if not isinstance(series, pd.Series):
        raise TypeError("series 必须是 pandas Series")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series 索引必须按递增顺序排列")
    if not isinstance(test_size, int) or isinstance(test_size, bool):
        raise TypeError("test_size 必须是整数")
    if test_size <= 0 or test_size >= len(series):
        raise ValueError("test_size 必须大于 0 且小于序列长度")

    split_at = len(series) - test_size
    return TemporalSplit(
        train=series.iloc[:split_at].copy(),
        test=series.iloc[split_at:].copy(),
    )

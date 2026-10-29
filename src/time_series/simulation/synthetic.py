"""Synthetic data used by the practice scripts."""

import numpy as np
import pandas as pd


def simulate_seasonal_random_walk(
    n_observations: int,
    season_length: int,
    level: float = 100.0,
    innovation_std: float = 1.0,
    seed: int | None = None,
    seasonal_amplitude: float = 4.0,
) -> pd.Series:
    """Simulate a seasonal random walk with a sinusoidal initial season."""
    if not isinstance(n_observations, int) or n_observations < 1:
        raise ValueError("n_observations 必须是正整数")
    if not isinstance(season_length, int) or season_length < 1:
        raise ValueError("season_length 必须是正整数")
    if n_observations < season_length:
        raise ValueError("n_observations 不能小于 season_length")
    if innovation_std < 0:
        raise ValueError("innovation_std 不能为负数")

    rng = np.random.default_rng(seed)
    innovations = rng.normal(0.0, innovation_std, n_observations)
    values = np.empty(n_observations, dtype=float)

    initial_size = min(season_length, n_observations)
    phase = np.arange(initial_size)
    values[:initial_size] = (
        level
        + seasonal_amplitude * np.sin(2.0 * np.pi * phase / season_length)
        + innovations[:initial_size]
    )
    for position in range(season_length, n_observations):
        values[position] = values[position - season_length] + innovations[position]

    index = pd.date_range("2010-01-31", periods=n_observations, freq="ME")
    return pd.Series(values, index=index, name="value")

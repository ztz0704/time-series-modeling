"""Simple forecast baselines."""

import numpy as np
import pandas as pd


def _validate_train(train: pd.Series) -> None:
    if not isinstance(train, pd.Series):
        raise TypeError("train 必须是 pandas Series")
    if train.empty:
        raise ValueError("train 不能为空")
    if train.isna().any():
        raise ValueError("train 不能包含缺失值")


def mean_forecast(train: pd.Series, forecast_index: pd.Index) -> pd.Series:
    """Forecast every future observation with the training mean."""
    _validate_train(train)
    return pd.Series(float(train.mean()), index=forecast_index, dtype=float)


def naive_forecast(train: pd.Series, forecast_index: pd.Index) -> pd.Series:
    """Forecast every future observation with the last training value."""
    _validate_train(train)
    return pd.Series(float(train.iloc[-1]), index=forecast_index, dtype=float)


def seasonal_naive_forecast(
    train: pd.Series,
    forecast_index: pd.Index,
    season_length: int,
) -> pd.Series:
    """Repeat the final observed season across the forecast horizon."""
    _validate_train(train)
    if not isinstance(season_length, (int, np.integer)) or season_length < 1:
        raise ValueError("season_length 必须是正整数")
    if len(train) < season_length:
        raise ValueError("训练集长度不能小于 season_length")

    last_season = train.iloc[-season_length:].to_numpy(dtype=float)
    values = np.resize(last_season, len(forecast_index))
    return pd.Series(values, index=forecast_index, dtype=float)

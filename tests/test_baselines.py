import numpy as np
import pandas as pd
import pytest

from time_series.models.baselines import mean_forecast,naive_forecast,seasonal_naive_forecast

def test_mean_forecast() -> None:
    train = pd.Series(
        [1.0, 2.0, 3.0],
    )
    forecast_index = pd.Index(
        ['a', 'b'],
    )

    prediction = mean_forecast(
        train=train,
        forecast_index=forecast_index,
    )

    np.testing.assert_allclose(
        prediction.to_numpy(),
        [2.0, 2.0],
    )


def test_naive_forecast() -> None:
    train = pd.Series(
        [1.0, 2.0, 3.0],
    )
    forecast_index = pd.Index(
        ["a", "b", "c"],
    )

    prediction = naive_forecast(
        train=train,
        forecast_index=forecast_index,
    )

    np.testing.assert_allclose(
        prediction.to_numpy(),
        [3.0, 3.0, 3.0],
    )


def test_seasonal_naive_repeats_last_season() -> None:
    train = pd.Series(
        [10.0, 20.0, 30.0, 40.0],
    )
    forecast_index = pd.RangeIndex(5)

    prediction = seasonal_naive_forecast(
        train=train,
        forecast_index=forecast_index,
        season_length=2,
    )

    np.testing.assert_allclose(
        prediction.to_numpy(),
        [30.0, 40.0, 30.0, 40.0, 30.0],
    )


def test_seasonal_naive_rejects_short_train() -> None:
    train = pd.Series(
        [1.0, 2.0],
    )

    with pytest.raises(
        ValueError,
        match="不能小于",
    ):
        seasonal_naive_forecast(
            train=train,
            forecast_index=pd.RangeIndex(2),
            season_length=3,
        )


def test_baseline_rejects_missing_values() -> None:
    train = pd.Series(
        [1.0, np.nan, 3.0],
    )

    with pytest.raises(ValueError):
        naive_forecast(
            train=train,
            forecast_index=pd.RangeIndex(2),
        )
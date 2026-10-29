import numpy as np
import pytest

from time_series.evaluation.metrics import (
    mae,
    mase,
    rmse,
)


def test_mae() -> None:
    result = mae(
        y_true=[1.0, 2.0, 3.0],
        y_pred=[1.0, 4.0, 2.0],
    )

    assert result == pytest.approx(1.0)


def test_rmse() -> None:
    result = rmse(
        y_true=[1.0, 2.0],
        y_pred=[2.0, 4.0],
    )

    expected = np.sqrt(
        (1.0**2 + 2.0**2) / 2.0
    )

    assert result == pytest.approx(expected)


def test_nonseasonal_mase() -> None:
    result = mase(
        y_true=[5.0, 6.0],
        y_pred=[4.0, 7.0],
        y_train=[1.0, 2.0, 3.0, 4.0],
        seasonality=1,
    )

    # 分子 MAE = 1，分母 mean(|diff(train)|) = 1。
    assert result == pytest.approx(1.0)


def test_seasonal_mase() -> None:
    result = mase(
        y_true=[3.0, 12.0],
        y_pred=[2.0, 11.0],
        y_train=[1.0, 10.0, 2.0, 11.0],
        seasonality=2,
    )

    # 分母为 mean(|2-1|, |11-10|) = 1。
    assert result == pytest.approx(1.0)


def test_metric_rejects_different_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="长度必须一致",
    ):
        mae(
            y_true=[1.0, 2.0],
            y_pred=[1.0],
        )


def test_mase_rejects_zero_scale() -> None:
    with pytest.raises(
        ValueError,
        match="分母为零",
    ):
        mase(
            y_true=[1.0],
            y_pred=[2.0],
            y_train=[5.0, 5.0, 5.0],
            seasonality=1,
        )
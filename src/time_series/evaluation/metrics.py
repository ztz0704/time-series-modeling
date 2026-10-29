"""Common point-forecast metrics."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _as_finite_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} 必须是一维非空序列")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 不能包含缺失值或无穷值")
    return array


def _paired_arrays(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    actual = _as_finite_array(y_true, name="y_true")
    predicted = _as_finite_array(y_pred, name="y_pred")
    if actual.size != predicted.size:
        raise ValueError("y_true 与 y_pred 的长度必须一致")
    return actual, predicted


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return mean absolute error."""
    actual, predicted = _paired_arrays(y_true, y_pred)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Return root mean squared error."""
    actual, predicted = _paired_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mase(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_train: Sequence[float],
    seasonality: int = 1,
) -> float:
    """Return mean absolute scaled error using a seasonal-naive scale."""
    actual, predicted = _paired_arrays(y_true, y_pred)
    train = _as_finite_array(y_train, name="y_train")
    if not isinstance(seasonality, (int, np.integer)) or seasonality < 1:
        raise ValueError("seasonality 必须是正整数")
    if train.size <= seasonality:
        raise ValueError("训练集长度必须大于 seasonality")

    scale = float(np.mean(np.abs(train[seasonality:] - train[:-seasonality])))
    if np.isclose(scale, 0.0):
        raise ValueError("MASE 分母为零，无法计算")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def evaluate_forecasts(
    y_true: Sequence[float],
    predictions: Mapping[str, Sequence[float]],
    y_train: Sequence[float],
    seasonality: int = 1,
) -> pd.DataFrame:
    """Evaluate named forecasts and return results ordered by MAE."""
    if not predictions:
        raise ValueError("predictions 不能为空")

    rows = []
    for model, prediction in predictions.items():
        rows.append(
            {
                "model": model,
                "mae": mae(y_true, prediction),
                "rmse": rmse(y_true, prediction),
                "mase": mase(y_true, prediction, y_train, seasonality),
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ignore_index=True)

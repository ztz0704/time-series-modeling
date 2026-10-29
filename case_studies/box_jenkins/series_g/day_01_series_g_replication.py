from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from time_series.evaluation.metrics import evaluate_forecasts
from time_series.evaluation.split import temporal_train_test_split
from time_series.models.baselines import mean_forecast, naive_forecast, seasonal_naive_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / 'data'
    / 'raw'
    / 'box_jenkins'
    / 'series_g_airline.csv'
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / 'data'
    / 'processed'
    / 'box_jenkins'
)

FIGURE_DIR = PROJECT_ROOT / 'results' / 'figures'
TABLE_DIR = PROJECT_ROOT / 'results' / 'tables'

def load_series_g(file_path: Path) -> pd.Series:

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到 Series G 数据：{file_path}"            
        )

    data = pd.read_csv(
        file_path,
        parse_dates=['date'],
    )

    required_columns = {
        'date',
        'passengers'
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"数据缺少列：{sorted(missing_columns)}"
        )

    if data["date"].isna().any():
        raise ValueError("date 列存在缺失值。")

    if data["passengers"].isna().any():
        raise ValueError("passengers 列存在缺失值。")

    if data["date"].duplicated().any():
        raise ValueError("日期存在重复值。")

    series = (
        data
        .set_index('date')['passengers']
        .sort_index()
        .astype(float)
        .rename('passengers')
    )

    if not series.index.is_monotonic_increasing:
        raise ValueError("时间索引必须递增排列。")

    if len(series) != 144:
        raise ValueError(
            f"Series G 应有 144 个观测，当前为 {len(series)}。"
        )

    expected_start = pd.Timestamp('1949-01-01')
    expected_end = pd.Timestamp('1960-12-01')

    expected_start = pd.Timestamp("1949-01-01")
    expected_end = pd.Timestamp("1960-12-01")

    if series.index[0] != expected_start:
        raise ValueError("Series G 起始日期不正确。")

    if series.index[-1] != expected_end:
        raise ValueError("Series G 结束日期不正确。")

    return series

def save_series_plot(
    series: pd.Series,
    title: str,
    ylabel: str,
    output_file: Path,
) -> None:

    figure, axis = plt.subplots(
        figsize=(12, 6),
    )

    series.plot(
        ax=axis,
        linewidth=1.5,
    )

    axis.set_title(title)
    axis.set_xlabel('date')
    axis.set_ylabel(ylabel)

    figure.tight_layout()

    figure.savefig(
        output_file,
        dpi=150,
    )

    plt.close(figure)

def run_replication() -> pd.DataFrame:

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    series = load_series_g(DATA_FILE)

    log_series = (
        np.log(series)
        .rename("log_passengers")
    )

    processed_file = (
        PROCESSED_DIR
        / 'series_g_log.csv'
    )

    log_series.to_csv(
        processed_file,
        index=True,
        index_label='date',
    )

    raw_figure_file = (
        FIGURE_DIR
        / 'series_g_raw.png'
    )

    save_series_plot(
        series=series,
        title='Box-Jenkins Series G: Airline Passengers',
        ylabel='Passengers(thousands)',
        output_file=raw_figure_file,
    )

    log_figure_file = (
        FIGURE_DIR
        / 'series_g_log.png'
    )

    save_series_plot(
        series=log_series,
        title='Box-Jenkins Series G: Log Airline Passengers',
        ylabel='Log passengers',
        output_file=log_figure_file,
    )

    split = temporal_train_test_split(
        series=series,
        test_size=24,
    )

    forecast_index = split.test.index

    mean_prediction = mean_forecast(
        train=split.train,
        forecast_index=forecast_index,
    )

    naive_prediction = naive_forecast(
        train=split.train,
        forecast_index=forecast_index,
    )

    seasonal_naive_prediction = seasonal_naive_forecast(
        train=split.train,
        forecast_index=forecast_index,
        season_length=12,
    )

    predictions = {
        "mean": mean_prediction,
        "naive": naive_prediction,
        "seasonal_naive": seasonal_naive_prediction,
    }

    results = evaluate_forecasts(
        y_true=split.test,
        predictions=predictions,
        y_train=split.train,
        seasonality=12,
    )

    metrics_file = (
        TABLE_DIR
        / "series_g_baseline_metrics.csv"
    )

    results.to_csv(
        metrics_file,
        index=False,
    )

    forecast_table = pd.DataFrame(
        {
            "actual": split.test,
            "mean": mean_prediction,
            "naive": naive_prediction,
            "seasonal_naive": seasonal_naive_prediction,
        }
    )

    forecasts_file = (
        TABLE_DIR
        / "series_g_baseline_forecasts.csv"
    )

    forecast_table.to_csv(
        forecasts_file,
        index=True,
        index_label="date",
    )

    forecast_figure_file = (
        FIGURE_DIR
        / "series_g_baseline_forecasts.png"
    )

    figure, axis = plt.subplots(
        figsize=(12, 6),
    )

    split.train.iloc[-48:].plot(
        ax=axis,
        label="train",
    )

    split.test.plot(
        ax=axis,
        label="actual",
        linewidth=2,
    )

    mean_prediction.plot(
        ax=axis,
        label="mean",
        linestyle="--",
    )

    naive_prediction.plot(
        ax=axis,
        label="naive",
        linestyle="--",
    )

    seasonal_naive_prediction.plot(
        ax=axis,
        label="seasonal_naive",
        linestyle="--",
    )

    axis.set_title(
        "Series G: Baseline Forecast Comparison"
    )

    axis.set_xlabel("Date")
    axis.set_ylabel("Passengers (thousands)")
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        forecast_figure_file,
        dpi=150,
    )

    plt.close(figure)

    print("\nSeries G data summary\n")

    print(f"Observations: {len(series)}")
    print(f"Start: {series.index.min().date()}")
    print(f"End: {series.index.max().date()}")

    print("\nBaseline metrics\n")

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nOutput files\n")
    print(processed_file)
    print(raw_figure_file)
    print(log_figure_file)
    print(metrics_file)
    print(forecasts_file)
    print(forecast_figure_file)

    return results


def main() -> None:
    run_replication()


if __name__ == "__main__":
    main()

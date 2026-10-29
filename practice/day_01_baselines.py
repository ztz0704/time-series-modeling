from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from time_series.simulation.synthetic import simulate_seasonal_random_walk
from time_series.evaluation.metrics import evaluate_forecasts
from time_series.evaluation.split import temporal_train_test_split
from time_series.models.baselines import mean_forecast,naive_forecast,seasonal_naive_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FIGURE_DIR = PROJECT_ROOT / 'results' / 'figures'
OUTPUT_TABLE_DIR = PROJECT_ROOT / 'results' / 'tables'

def run_day01_experiment() -> pd.DataFrame:

    OUTPUT_FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )
    OUTPUT_TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    series = simulate_seasonal_random_walk(
        n_observations=144,
        season_length=12,
        level=100.0,
        innovation_std=0.8,
        seed=42,
    )

    split = temporal_train_test_split(
        series=series,
        test_size=24,
    )

    forecast_index = split.test.index

    mean_prediction = mean_forecast(
        train = split.train,
        forecast_index=forecast_index
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
        'mean': mean_prediction,
        'naive': naive_prediction,
        'seasonal_naive': seasonal_naive_prediction,
    }

    results = evaluate_forecasts(
        y_true=split.test,
        predictions=predictions,
        y_train=split.train,
        seasonality=12,
    )

    metrics_file = (
        OUTPUT_TABLE_DIR
        / 'day_01_baseline_metrics.csv'
    )

    results.to_csv(
        metrics_file,
        index=False
    )

    forecast_table = pd.DataFrame(
        {
            'actual': split.test,
            'mean': mean_prediction,
            'naive': naive_prediction,
            'seasonal_naive':seasonal_naive_prediction
        }
    )

    forecasts_file = (
        OUTPUT_TABLE_DIR
        / 'day_01_baseline_forecasts.csv'
    )

    forecast_table.to_csv(
        forecasts_file,
        index=True,
        index_label='date'
    )

    figure, axis = plt.subplots(
        figsize=(12, 6),
    )

    split.train.iloc[-48:].plot(
        ax=axis,
        label='train',
    )

    split.test.plot(
        ax=axis,
        label='actual',
        linewidth=2,
    )

    mean_prediction.plot(
        ax=axis,
        label='mean',
        linestyle='--'
    )

    naive_prediction.plot(
        ax=axis,
        label='naive',
        linestyle='--',
    )

    seasonal_naive_prediction.plot(
        ax=axis,
        label='seasonal_naive',
        linestyle='--',
    )

    axis.set_title(
        'Day 1:Baseline Forecast Comparison'
    )
    axis.set_xlabel('Date')
    axis.set_ylabel('Value')
    axis.legend()

    figure.tight_layout()

    figure_file = (
        OUTPUT_FIGURE_DIR
        / 'day_01_baseline_comparison.png'
    )

    figure.savefig(
        figure_file,
        dpi=150,
    )

    plt.close(figure)

    print('\nDay 1 baseline comparison\n')
    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print('\n输出文件:')
    print(metrics_file)
    print(forecasts_file)
    print(figure_file)

    return results

def main() -> None:
    run_day01_experiment()

if __name__ == "__main__":
    main()

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.graphics.tsaplots import (
    plot_acf,
    plot_pacf,
)

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
)

from statsmodels.tsa.arima.model import (
    ARIMA,
)


# %%
# 1. Box-Jenkins Series F

x = np.array(
    [
        47, 64, 23, 71, 38,
        64, 55, 41, 59, 48,
        71, 35, 57, 40, 58,
        44, 80, 55, 37, 74,
        51, 57, 50, 60, 45,
        57, 50, 45, 25, 59,
        50, 71, 56, 74, 50,
        58, 45, 54, 36, 54,
        48, 55, 45, 57, 50,
        62, 44, 64, 43, 52,
        38, 59, 55, 41, 53,
        49, 34, 35, 54, 45,
        68, 38, 50, 60, 39,
        59, 40, 57, 54, 23,
    ],
    dtype=float,
)


# %%
# 2. 原始序列

plt.figure(figsize=(10, 4))

plt.plot(
    np.arange(1, len(x) + 1),
    x,
    marker="o",
    markersize=3,
)

plt.title(
    "Box-Jenkins Series F"
)
plt.xlabel("Batch")
plt.ylabel("Yield")
plt.tight_layout()
plt.show()


# %%
# 3. ACF 与 PACF

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4),
)

plot_acf(
    x,
    lags=25,
    fft=False,
    ax=axes[0],
)

plot_pacf(
    x,
    lags=20,
    method="ywm",
    ax=axes[1],
)

axes[0].set_title(
    "Series F: ACF"
)

axes[1].set_title(
    "Series F: PACF"
)

plt.tight_layout()
plt.show()


# %%
# 4. 拟合 AR(2)

model = ARIMA(
    x,
    order=(2, 0, 0),
    trend="c",
).fit()

print(model.summary())

print("\n估计参数：")

parameter_table = pd.DataFrame(
    {
        "parameter":
            model.param_names,
        "estimate":
            model.params,
    }
)

print(
    parameter_table.to_string(
        index=False
    )
)


# %%
# 5. 六步预测

forecast = model.get_forecast(
    steps=6
)

forecast_table = forecast.summary_frame(
    alpha=0.10
)

forecast_table.index = np.arange(
    71,
    77,
)

forecast_table.index.name = "period"

print("\n六步预测：")
print(forecast_table)


# %%
# 6. 预测图

future_time = np.arange(
    71,
    77,
)

mean_forecast = (
    forecast_table["mean"].to_numpy()
)

lower = (
    forecast_table[
        "mean_ci_lower"
    ].to_numpy()
)

upper = (
    forecast_table[
        "mean_ci_upper"
    ].to_numpy()
)

plt.figure(figsize=(10, 4))

plt.plot(
    np.arange(1, 71),
    x,
    label="Observed",
)

plt.plot(
    future_time,
    mean_forecast,
    marker="o",
    label="Forecast",
)

plt.fill_between(
    future_time,
    lower,
    upper,
    alpha=0.2,
    label="90% interval",
)

plt.title(
    "Series F: AR(2) forecast"
)
plt.xlabel("Period")
plt.ylabel("Yield")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# 7. 残差诊断

residuals = model.resid

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4),
)

axes[0].plot(residuals)
axes[0].axhline(0)
axes[0].set_title(
    "AR(2) residuals"
)

plot_acf(
    residuals,
    lags=20,
    fft=False,
    ax=axes[1],
)

axes[1].set_title(
    "Residual ACF"
)

plt.tight_layout()
plt.show()


ljung_box = acorr_ljungbox(
    residuals,
    lags=[6, 12, 18],
    model_df=2,
    return_df=True,
)

print("\nLjung-Box 检验：")
print(ljung_box)


# %%
# 8. 使用书中参数手工递推预测

mu = 51.1286
phi_1 = -0.3198
phi_2 = 0.1797

history = list(x)
manual_forecast = []

for _ in range(6):
    next_value = (
        mu
        + phi_1
        * (history[-1] - mu)
        + phi_2
        * (history[-2] - mu)
    )

    manual_forecast.append(
        next_value
    )

    history.append(next_value)


comparison = pd.DataFrame(
    {
        "period":
            future_time,
        "published_parameter_forecast":
            manual_forecast,
        "statsmodels_forecast":
            mean_forecast,
    }
)

print(
    "\n书中参数递推预测与现代估计比较："
)

print(
    comparison.to_string(
        index=False
    )
)

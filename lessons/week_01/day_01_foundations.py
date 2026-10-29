import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.tsa.stattools import adfuller

SEED = 42
N = 300
MAX_LAG = 40

def sample_acf(x, max_lag):

    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)

    acov = np.array(
        [
            np.sum(x[k:] * x[:n-k]) / n 
            for k in range(max_lag + 1)
        ]
    )

    return acov / acov[0]

rng = np.random.default_rng(SEED)

white_noise = rng.normal(size=N)

random_walk = np.cumsum(
    rng.normal(size=N)
)

near_unit_root = np.zeros(N)
innovations = rng.normal(size=N)

for t in range(1, N):
    near_unit_root[t] = (
        0.98 * near_unit_root[t - 1]
        + innovations[t]
    )

u = np.zeros(N)
innovations = rng.normal(size=N)

for t in range(1, N):
    u[t] = (
        0.6 * u[t - 1]
        + innovations[t]
    )

trend_stationary = (
    0.05 * np.arange(1, N + 1)
    + u
)

series = {
    "White noise": white_noise,
    "Random walk": random_walk,
    "Near-unit-root AR(1)": near_unit_root,
    "Trend stationary": trend_stationary,   
}

fig, ax = plt.subplots(
    4,
    2,
    figsize=(12, 11),
)

confidence = 1.96 / np.sqrt(N)

for row, (name, values) in enumerate(
    series.items()
):
    ax[row, 0].plot(values)
    ax[row, 0].set_title(
        f'{name}: path'
    )
    ax[row, 0].set_xlabel('Time')

    correlation = sample_acf(
        values,
        MAX_LAG,
    )
    ax[row, 1].stem(
        range(MAX_LAG + 1),
        correlation,
    )

    ax[row, 1].axhline(
        confidence,
        linestyle='--',
    )
    ax[row, 1].axhline(
        -confidence,
        linestyle='--',
    )
    ax[row, 1].set_title(
        f"{name}: sampl ACF"
    )
    ax[row, 1].set_xlabel('Lag')

plt.tight_layout()
plt.show()


custom = sample_acf(
    white_noise,
    MAX_LAG,
)

library = sm_acf(
    white_noise,
    nlags=MAX_LAG,
    fft=False,
)

max_error = np.max(
    np.abs(custom - library)
)

print(
    "手写 ACF 与 statsmodels 最大误差：",
    max_error,
)

rows = []

for name, values in {
    "Random walk": random_walk,
    "Near-unit-root": near_unit_root,
    "Trend stationary": trend_stationary,
}.items():
    difference = np.diff(values)

    rows.append(
        {
            'DSG': name,
            "level_acf_1": sample_acf(values, 1)[1],
            "difference_acf_1": sample_acf(difference, 1)[1],
            "level_variance": np.var(values),
            "difference_variance": np.var(difference),
        }
    )

difference_table = pd.DataFrame(rows)

print("\n差分前后比较: ")
print(
    difference_table.to_string(
        index=False
    )
)

def acf_monte_carlo(
    sample_size,
    replications=500,
    max_lag=20,
):
    rng = np.random.default_rng(
        1000 + sample_size
    )

    simulated_acf = np.array(
        [
            sample_acf(
                rng.normal(size=sample_size),
                max_lag,
            )[1:]
            for _ in range(replications)
        ]
    )

    boundary = (
        1.96 / np.sqrt(sample_size)
    )

    return {
        'n': sample_size,
        'Var(acf_1)': 
            simulated_acf[:, 0].var(
                ddof=1
            ),
        '1/n':
            1 / sample_size,
        'pointwise_exceedance':
            np.mean(
                np.abs(simulated_acf)
                > boundary
            ),
        'any_lag_exceedance':
            np.mean(
                np.any(
                    np.abs(simulated_acf)
                    > boundary,
                    axis=1
                )
            ),
    }

mc_table = pd.DataFrame(
    [
        acf_monte_carlo(n)
        for n in [100, 200, 300]
    ]
)

print("\n白噪声 ACF Monte Carlo：")
print(
    mc_table.to_string(
        index=False
    )
)

print(
    "\n20 个独立检验至少一次越界的参考概率：",
    1 - 0.95**20,
)

adf_rows = []

for name,values in series.items():
    for regression in [
        'n',
        'c',
        'ct',
    ]:
        result = adfuller(
            values,
            regression=regression,
            autolag='AIC'
        )

        adf_rows.append(
            {
                "DGP": name,
                "regression": regression,
                "statistic": result[0],
                "p_value": result[1],
                "used_lag": result[2],
                "decision_5%": (
                    "reject"
                    if result[1] < 0.05
                    else "not reject"
                ),
            }
        )

adf_table = pd.DataFrame(adf_rows)

print("\nADF 结果：")
print(
    adf_table.to_string(
        index=False
    )
)
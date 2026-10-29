import numpy as np

def simulate_arma11(
    phi,
    theta,
    sigma_a,
    sample_size,
    burn_in=500,
    seed=2026,
):
    rng = np.random.default_rng(seed)

    total_size = burn_in + sample_size

    innovations = rng.normal(
        loc=0.0,
        scale=sigma_a,
        size=total_size,
    )

    series = np.zeros(total_size)

    for t in range(1, total_size):
        series[t] = phi * series[t-1] + innovations[t] - theta * innovations[t-1]

    return series[burn_in:], innovations[burn_in:]

z, true_a = simulate_arma11(
    phi=0.7,
    theta=0.4,
    sigma_a=1,
    sample_size=10000,
    burn_in=500,
    seed=2026,
)

def recover_innovations(
    series,
    phi,
    theta,
):

    innovations = np.zeros(len(series))

    for t in range(1, len(series)):
        innovations[t] = series[t] - phi * series[t-1] + theta * innovations[t-1]

    return innovations

estimated_a = recover_innovations(
    z,
    phi=0.7,
    theta=0.4,
)

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

results = []
orders = [
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
]

for order in orders:
    result = ARIMA(
        z,
        order=order,
        trend='n',
    ).fit()

    results.append((order,result.aic))

result_df = pd.DataFrame(results, columns=['order', 'AIC']).sort_values(by='AIC', ascending=True).reset_index(drop=True)
best_order = result_df.loc[0, 'order']
best_result = ARIMA(z, order=best_order,trend='n').fit()

from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(
    best_result.resid,
    lags=[10,20],
    model_df=best_order[0]+best_order[2],
    return_df=True
)

print(lb)
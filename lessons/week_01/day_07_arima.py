import numpy as np

def simulate_arima011(
    theta,
    sigma_a,
    sample_size,
    z0=0.0,
    seed=2026,
):

    rng = np.random.default_rng(seed)

    innovations = rng.normal(
        loc=0.0,
        scale=sigma_a,
        size=sample_size+1,
    )

    w = innovations[1:] - theta * innovations[:-1]
    z = z0 + np.cumsum(w)

    return z, w

z, w = simulate_arima011(
    theta=0.4,
    sigma_a=1,
    sample_size=10000
)

z0 = 0
dz = np.diff(
    np.concatenate(([z0], z))
)

def forecast_arima011(
        z_T,
        a_T,
        theta,
        sigma_a,
        horizon,      
):

    h = np.arange(1, horizon + 1)

    forecast = np.full(
        horizon,
        z_T - theta * a_T,
    )

    forecast_var = (
        sigma_a ** 2
        * (1 + (h - 1) * (1 - theta) ** 2)
    )

    forecast_std = np.sqrt(forecast_var)

    return forecast, forecast_var, forecast_std
    
forecast, forecast_var, forecast_std = forecast_arima011(
    z_T=10.0,
    a_T=0.8,
    theta=0.4,
    sigma_a=1.0,
    horizon=5,
)

from statsmodels.tsa.arima.model import ARIMA

result = ARIMA(
    z,
    order=(0, 1, 1),
    trend='n',
).fit()

pred = result.get_forecast(steps=20)

forecast_mean = pred.predicted_mean
forecast_var = pred.var_pred_mean

print(forecast_mean[:5])
print(forecast_var[:5])
print(np.diff(forecast_var))
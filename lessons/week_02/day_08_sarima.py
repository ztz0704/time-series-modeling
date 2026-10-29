import numpy as np

def simulate_seasonal_rw(
    sigma_a,
    sample_size,
    season_length=12,
    seed=2026,
):
    rng = np.random.default_rng(seed)

    
    total_size = season_length + sample_size

    innovations = rng.normal(
        loc=0.0,
        scale=sigma_a,
        size=total_size,
    )

    z = np.zeros(total_size)

    for t in range(season_length, total_size):
        z[t] = z[t - season_length] + innovations[t]

    return z[season_length:], innovations[season_length:]

def simulate_multiplicative_sma(
        theta,
        seasonal_theta,
        sigma_a,
        sample_size,
        season_length=12,
        seed=2026,
):

    rng = np.random.default_rng(seed)

    innnovations = rng.normal(
        loc=0.0,
        scale=sigma_a,
        size=sample_size+season_length+1,
    )

    w = (
        innnovations[season_length+1:] 
        - theta * innnovations[season_length:-1]
        - seasonal_theta * innnovations[1:-season_length]
        + theta * seasonal_theta * innnovations[:-season_length-1]
    )

    return w

w = simulate_multiplicative_sma(
    theta=0.4,
    seasonal_theta=0.6,
    sigma_a=1,
    sample_size=100000,
)
def sample_acf_at(series, lag):
    values = np.asarray(series)
    centered = values - np.mean(values)

    c0 = np.sum(centered * centered) /len(values)
    ck = np.sum(centered[lag:] * centered[:-lag]) / len(values)

    return ck / c0

from statsmodels.tsa.statespace.sarimax import SARIMAX

result = SARIMAX(
    w,
    order=(0, 0, 1),
    seasonal_order=(0, 0, 1, 12),
    trend='n',
).fit(disp=False)

print(result.params)

print(result.maparams)
print(result.seasonalmaparams)

from statsmodels.stats.diagnostic import acorr_ljungbox

lb = acorr_ljungbox(
    result.resid,
    lags=[12, 24],
    model_df=2,
)

print(lb)
import numpy as np

def simulate_ma1(
    theta: float,
    sigma_a: float,
    sample_size: int,
    seed: int = 2026,
) -> np.ndarray:
    if sample_size <= 0:
        raise ValueError("sample_size 必须正整数")

    rng = np.random.default_rng(seed)

    innovations = rng.normal(
        loc=0.0,
        scale=sigma_a,
        size=sample_size+1,
    )

    series = (
        innovations[1:]
        - theta * innovations[:-1]
    )

    return series

def sample_acf_at_lag(series, lag):
    values = np.asarray(series)

    centered = values - np.mean(values)

    c0 = np.sum(
        centered * centered
    ) / len(values)

    ck = np.sum(
        centered[lag:] * centered[:-lag]
    ) / len(values)

    rk = ck / c0

    return rk

sample_size = 100000

z_noninvertible = simulate_ma1(
    theta=2,
    sigma_a=1,
    sample_size=sample_size,
)

z_invertible = simulate_ma1(
    theta=0.5,
    sigma_a=2,
    sample_size=sample_size,
)

for name, z in [
    ('noninvertible', z_noninvertible),
    ('invertible', z_invertible),
]:
    variance = np.var(z)
    r1 = sample_acf_at_lag(z, 1)
    r2 = sample_acf_at_lag(z, 2)

    print(name, variance, r1, r2)
import numpy as np
from day_04_ma1 import sample_acf_at_lag

def simulate_ar2(
    phi1,
    phi2,
    sigma_a,
    sample_size,
    burn_in=500,
    seed=2026,
):
    rng = np.random.default_rng(seed)

    total_size = sample_size + burn_in

    innovations = rng.normal(
        loc=0.0,
        scale=sigma_a,
        size=total_size,
    )

    series = np.zeros(total_size)

    for t in range(2, total_size):
        series[t] = (
            phi1 * series[t-1] + phi2 * series[t-2] + innovations[t]
            )

    return series[2+burn_in:]

z = simulate_ar2(
    phi1=1.2,
    phi2=-0.5,
    sigma_a=1,
    sample_size=10000,
    seed=2026,
)

print(z.shape)
print(z[:5])

r1 = sample_acf_at_lag(z, 1)
r2 = sample_acf_at_lag(z, 2)

R = np.array([[1, r1], [r1, 1]])
r = np.array([r1, r2])

phi_hat1, phi_hat2 = np.linalg.solve(R, r)
print(phi_hat1, phi_hat2)

def check_ar2_stationary(phi1, phi2):
    roots = np.roots([-phi2, -phi1, 1])
    is_stationary = np.all(np.abs(roots) > 1)
    return is_stationary

print(check_ar2_stationary(1.2,-0.5))

phi1 = 1.2
phi2 = -0.5

B_roots = np.roots([-phi2, -phi1, 1])
lambda_roots = np.roots([1, -phi1, -phi2])

print(B_roots)
print(lambda_roots)
print(1 / lambda_roots)

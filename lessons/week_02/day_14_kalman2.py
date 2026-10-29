import numpy as np
from itertools import product

rng = np.random.default_rng(42)

n = 300
q_true = 0.05
r_true = 0.5

x = np.zeros(n)

w = rng.normal(0, np.sqrt(q_true), n)
v = rng.normal(0, np.sqrt(r_true), n)

for t in range(1, n):
    x[t] = x[t - 1] + w[t]

y = x + v

def kalman_nll(y, q, r):
    sample_size = len(y)
    a0 = np.zeros(sample_size + 1)
    a1 = np.zeros(sample_size + 1)
    P0 = np.ones(sample_size + 1)
    P1 = np.ones(sample_size + 1)
    niu = np.zeros(sample_size + 1)
    S = np.ones(sample_size + 1)
    for t in range(1, sample_size + 1):
        a1[t] = a0[t - 1]
        P1[t] = P0[t - 1] + q
        niu[t] = y[t - 1] - a1[t]
        S[t] = P1[t] + r
        a0[t] = a1[t] + P1[t] / S[t] * niu[t]
        P0[t] = (1 - P1[t] / S[t]) * P1[t]

    innovations = niu[1:]
    innovations_var = S[1:]
    nll = np.sum(np.log(innovations_var) + innovations ** 2 / innovations_var)

    return nll, innovations, innovations_var

cadidates = [
    (0.05, 0.5),
    (0.001, 2),
    (1.0, 0.05),
]

results = []
for candidate in cadidates:
    q = candidate[0]
    r = candidate[1]
    nll, innovations, innovations_var = kalman_nll(y , q, r)
    results.append((q, r, nll))

print(results)

nll, innovations, innovations_var = kalman_nll(y , 0.05, 0.5)
print(np.mean(innovations / np.sqrt(innovations_var)))
print(np.var(innovations / np.sqrt(innovations_var), ddof=1))
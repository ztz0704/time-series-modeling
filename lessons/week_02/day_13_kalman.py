import numpy as np

rng = np.random.default_rng(42)

n = 200
q = 0.05
r = 0.5

x = np.zeros(n)
x[0] = 0.0

w = rng.normal(0, np.sqrt(q), n)
v = rng.normal(0, np.sqrt(r), n)

for t in range(1, n):
    x[t] = x[t - 1] + w[t]

y = x + v

a = np.zeros(n)
P = np.zeros(n)
K = np.zeros(n)

a[0] = 0.0
P[0] = 1.0

if np.isnan(y[0]):
    K[0] = 0.0
else:
    K[0] = P[0] / (P[0] + r)
    a[0] = a[0] + K[0] * (y[0] - a[0])
    P[0] = (1 - K[0]) * P[0]

for t in range(1, n):
    a[t] = a[t - 1]
    P[t] = P[t - 1] + q

    if np.isnan(y[t]):
        K[t] = 0.0
    else:
        K[t] = P[t] / (P[t] + r)
        a[t] = a[t] + K[t] * (y[t] - a[t])
        P[t] = (1 - K[t]) * P[t]

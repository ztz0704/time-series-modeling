import numpy as np

rng = np.random.default_rng(42)

n = 200
q_level = 0.05
q_slope = 0.002
r = 0.5

level = np.zeros(n)
slope = np.zeros(n)
y = np.zeros(n)

level[0] = 0.0
slope[0] = 0.03

eta = rng.normal(0, np.sqrt(q_level), n)
zeta = rng.normal(0, np.sqrt(q_slope), n)

for t in range(1, n):
   level[t] = level[t - 1] + slope[t - 1] + eta[t]
   slope[t] = slope[t - 1] + zeta[t]

a = rng.normal(0, np.sqrt(r), n)
y = level + a

F = np.array([[1, 1], [0, 1]])
H = np.array([1, 0])
Q = np.array([[q_level, 0], [0, q_slope]])

x = np.zeros((n,2))
state_noise = rng.multivariate_normal([0, 0], Q, n)

x[0] = [0.0, 0.03]
for t in range(1, n):
   x[t] = F @ x[t - 1] + state_noise[t]

y = x @ H + a
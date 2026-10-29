import numpy as np

rng = np.random.default_rng(42)

n = 400
tau = 200
delta = 2.0

mu = np.zeros(n)
mu[tau:] = delta

y = mu + rng.normal(0, 1, n)

def expanding_mean(y):
    em = np.zeros(len(y))
    for t in range(len(y)):
        em[t] = np.mean(y[:t+1])

    return em

def rolling_mean(y, window):
    rm = np.zeros(len(y))
    for t in range(len(y)):
        start = max(0, t - window +1)
        rm[t] = np.mean(y[start: t + 1])

    return rm

def ewma_mean(y, alpha):
    m = np.zeros(len(y))
    m[0] = y[0]
    for t in range(1, len(y)):
        m[t] = alpha * y[t] + (1 - alpha) * m[t - 1]

    return m

def MSE(y1, y2):
    return np.sqrt(np.mean((y1 - y2) ** 2))

windows = [10, 20, 50, 100]
alphas = [0.02, 0.05, 0.1, 0.2]

rolling_score = {}
ewma_score = {}

yv = y[200: 300]
for window in windows:
    rr = rolling_mean(y[:299], window)
    rolling_score[window] = MSE(rr[199: 299], yv)

for alpha in alphas:
    er = ewma_mean(y[:299], alpha)
    ewma_score[alpha] = MSE(er[199: 299], yv)

best_window = min(rolling_score, key=rolling_score.get)
best_alpha = min(ewma_score, key=ewma_score.get)

yt = y[:399]
mr = expanding_mean(yt)
MSE_mean = MSE(mr[299: 399], y[300:])
rr = rolling_mean(yt, best_window)
MSE_rolliing = MSE(rr[299: 399], y[300:])
er = ewma_mean(yt, best_alpha)
MSE_ewma = MSE(er[299: 399], y[300:])

def one_step_mse(y, rr, ini=300, end=400):
    return np.sqrt(np.mean((rr[ini-1: end-1] - y[ini: end]) ** 2))

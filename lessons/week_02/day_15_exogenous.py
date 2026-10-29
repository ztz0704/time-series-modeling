import numpy as np

rng = np.random.default_rng(42)

n = 600
beta = 1.5
rho = 0.8
phi = 0.6

x = np.zeros(n)
err = np.zeros(n)

u = rng.normal(0, 1.0, n)
eps = rng.normal(0, 0.7, n)

for t in range(1, n):
    x[t] = rho * x[t - 1] + u[t]
    err[t] = phi * err[t - 1] + eps[t]

y = beta * x + err

def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def dynamic_forecasts(x, y, window, start, end):
    pred_a = []
    pred_b = []
    pred_oracle = []
    beta_list = []
    rho_list = []
    phi_list = []

    for target in range(start, end):
        t = target - 1

        x_train = x[t - window + 1:t + 1]
        y_train = y[t - window + 1:t + 1]

        X = x_train.reshape(-1, 1)
        beta_hat = np.linalg.lstsq(X, y_train, rcond=None)[0][0]

        resid = y_train - beta_hat * x_train
        resid_lag = resid[:-1].reshape(-1, 1)
        phi_hat = np.linalg.lstsq(
            resid_lag, resid[1:], rcond=None
        )[0][0]

        x_lag = x_train[:-1].reshape(-1, 1)
        rho_hat = np.linalg.lstsq(
            x_lag, x_train[1:], rcond=None
        )[0][0]

        x_forecast = rho_hat * x[t]

        last_resid = resid[-1]
        forecast_a = beta_hat * x_forecast
        forecast_b = beta_hat * x_forecast + phi_hat * last_resid
        forecast_oracle = beta_hat * x[target] + phi_hat * last_resid

        pred_a.append(forecast_a)
        pred_b.append(forecast_b)
        pred_oracle.append(forecast_oracle)
        beta_list.append(beta_hat)
        rho_list.append(rho_hat)
        phi_list.append(phi_hat)

    return pred_a, pred_b, pred_oracle, beta_list, rho_list, phi_list


windows = [80, 150, 250]
validation_scores = {}

for window in windows:
    validation_results = dynamic_forecasts(
        x, y, window, start=350, end=500
    )
    validation_pred_b = validation_results[1]
    validation_scores[window] = mse(y[350:500], validation_pred_b)

best_window = min(validation_scores, key=validation_scores.get)


test_results = dynamic_forecasts(
    x, y, best_window, start=500, end=600
)

test_pred_a = test_results[0]
test_pred_b = test_results[1]
test_pred_oracle = test_results[2]
test_beta = test_results[3]
test_rho = test_results[4]
test_phi = test_results[5]

test_y = y[500:600]

mse_a = mse(test_y, test_pred_a)
mse_b = mse(test_y, test_pred_b)
mse_oracle = mse(test_y, test_pred_oracle)

mean_beta = np.mean(test_beta)
mean_rho = np.mean(test_rho)
mean_phi = np.mean(test_phi)


print("Validation scores:", validation_scores)
print("Best window:", best_window)
print("Test MSE A:", mse_a)
print("Test MSE B:", mse_b)
print("Test MSE oracle:", mse_oracle)
print("Mean beta estimate:", mean_beta, "True beta:", beta)
print("Mean rho estimate:", mean_rho, "True rho:", rho)
print("Mean phi estimate:", mean_phi, "True phi:", phi)

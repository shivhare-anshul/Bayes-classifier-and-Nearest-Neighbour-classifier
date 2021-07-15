import numpy as np


# Function to implement bayes
def bayes_mixture(theta_0, theta_1, weights_0, weights_1, X_test):
    mu_00, sigma_00 = theta_0[0]
    mu_01, sigma_01 = theta_0[1]
    mu_10, sigma_10 = theta_1[0]
    mu_11, sigma_11 = theta_1[1]
    w_00, w_01 = weights_0
    w_10, w_11 = weights_1

    Y_pred = np.zeros(len(X_test))

    for idx in range(len(X_test)):
        X = X_test[idx]
        q0 = w_00 * 1 / ((2 * np.pi * sigma_00) ** 0.5) * np.exp(-0.5 * (X - mu_00) ** 2 / sigma_00) + w_01 * 1 / (
                (2 * np.pi * sigma_01) ** 0.5) * np.exp(-0.5 * (X - mu_01) ** 2 / sigma_01)
        q1 = w_10 * 1 / ((2 * np.pi * sigma_10) ** 0.5) * np.exp(-0.5 * (X - mu_10) ** 2 / sigma_10) + w_11 * 1 / (
                (2 * np.pi * sigma_11) ** 0.5) * np.exp(-0.5 * (X - mu_11) ** 2 / sigma_11)

        if q0 >= q1:
            Y_pred[idx] = 0
        else:
            Y_pred[idx] = 1

    return Y_pred

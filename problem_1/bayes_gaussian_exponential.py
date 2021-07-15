import numpy as np

DIMENSIONS = 2


# Function to implement bayes classifier for Gaussian, Exponential class density
# Prior probability is assumed to be equal unless specified
def bayes_gaussian_exponential(theta_0, theta_1, X_test, p0=0.5, p1=0.5):
    # Density Class 0 parameters - mean and covariance matrix
    mu_0 = theta_0[0]
    sigma_0 = theta_0[1]
    det_sigma_0 = np.linalg.det(sigma_0)
    inv_sigma_0 = np.linalg.inv(sigma_0)

    # Density Class 1 parameters - lambda 1 and lambda 2
    # The features are assumed to be independent for exponential
    lambda_1 = theta_1[0]
    lambda_2 = theta_1[1]

    # Placeholder for output
    Y_pred = np.zeros(len(X_test))

    # Estimate the posterior density and label the output properly
    for idx in range(len(X_test)):
        X = X_test[idx]
        q0 = 1 / (((2 * np.pi) ** DIMENSIONS / 2) * (det_sigma_0 ** 0.5)) * np.exp(
            -0.5 * (X - mu_0) @ inv_sigma_0 @ np.transpose(X - mu_0)) * p0
        X = np.abs(X)
        q1 = lambda_1 * np.exp(-lambda_1 * X[0]) * lambda_2 * np.exp(-lambda_2 * X[1]) * p1

        if q0 >= q1:
            Y_pred[idx] = 0
        else:
            Y_pred[idx] = 1

    return Y_pred

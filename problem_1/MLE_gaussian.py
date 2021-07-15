import numpy as np

DIMENSIONS = 2


# Function to find the MLE estimate of mean and covariance for Gaussian density
def MLE_gaussian(X, N):
    # Estimate of mean
    mu_estimate = np.sum(X, axis=0) / N

    # Estimate of covariance matrix
    sigma_estimate = np.zeros([DIMENSIONS, DIMENSIONS])
    for idx in range(len(X)):
        outer_product = np.outer((X[idx] - mu_estimate), (X[idx] - mu_estimate))
        sigma_estimate += outer_product

    sigma_estimate = sigma_estimate / N

    return mu_estimate, sigma_estimate

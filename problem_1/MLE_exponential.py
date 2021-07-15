import numpy as np

DIMENSIONS = 2


# Function to find the MLE estimate of lambda for exponential density
def MLE_exponential(X, N):
    X = np.abs(X)
    lambda_estimate = N / np.sum(X, axis=0)

    return lambda_estimate

from sklearn.mixture import BayesianGaussianMixture


# Function to find the GMM mixture densities
def EM_gaussian(X_train):
    bgsm = BayesianGaussianMixture(n_components=2)

    # Fit the GMM on the samples and predict the labels
    labels = bgsm.fit_predict(X_train)

    # Since the label numbering is arbitrary we find whether the labels correspond to class 0 or class 1
    if labels[0] == 0:
        class_0 = 0
        class_1 = 1
    else:
        class_0 = 1
        class_1 = 0

    # Obtain the mean, covariance matrix and the density weights
    means = bgsm.means_
    weights = bgsm.weights_
    covariances = bgsm.covariances_

    # Wrapping parameters in a list
    theta_0 = [means[0], covariances[0]]
    theta_1 = [means[1], covariances[1]]

    return [theta_0, theta_1], weights

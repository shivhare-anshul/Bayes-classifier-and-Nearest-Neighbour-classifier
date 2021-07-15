import random
from sklearn.mixture import BayesianGaussianMixture
from problem_1.bayes import bayes
from problem_1.accuracy import accuracy


# Function for problem 1b
def problem_1b(X_train, Y_train, X_test, Y_test):
    bgsm = BayesianGaussianMixture(n_components=2)

    N_list = [10, 25, 50, 100, 150, 200]
    output_file = open('./results/problem_1b.txt', "w")
    output_file.write("N, GMM\n")

    for N in N_list:
        # Random sample N data points and sort the list
        rand_list = random.sample(range(0, 200), N)
        rand_list.sort()

        # Fit the GMM on the N samples and predict the labels
        X_train_rand = X_train[rand_list]
        labels = bgsm.fit_predict(X_train_rand)

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

        # Predict using Bayes classifier for Gaussian - Gaussian class conditional densities
        # Use the density weights as prior densities
        Y_pred = bayes(theta_0, theta_1, X_test, weights[0], weights[1])
        Y_pred[Y_pred == class_1] = -1
        Y_pred[Y_pred == class_0] = 1

        # Calculate the accuracy
        GMM_accuracy = accuracy(Y_pred, Y_test)

        # Save the output
        output_file.write("{}, {}\n".format(N, GMM_accuracy))

    output_file.close()

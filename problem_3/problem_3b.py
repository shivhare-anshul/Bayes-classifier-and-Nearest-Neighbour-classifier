from problem_3.bayes import bayes
from problem_3.EM_gaussian import EM_gaussian
from problem_3.MLE_gaussian import MLE_gaussian
from problem_3.bayes_mixture import bayes_mixture
from problem_3.nearest_neighbour import nearest_neighbour
from problem_3.accuracy import accuracy


# Function for problem 3b
def problem_3b(X_train, Y_train, X_test, Y_test):
    output_file = open('./results/problem_3b.txt', "w")

    # EM Algorithm implementation
    output_file.write("EM Algorithm\n")

    # Separate class 0 and class 1 data and estimate GMM densities
    X_class_0 = X_train[Y_train == -1]
    X_class_1 = X_train[Y_train == 1]
    theta_0_EM, weights_0 = EM_gaussian(X_class_0)
    theta_1_EM, weights_1 = EM_gaussian(X_class_1)

    # Predict using Bayes classifier for Gaussian mixture class conditional densities
    Y_pred_EM = bayes_mixture(theta_0_EM, theta_1_EM, weights_0, weights_1, X_test)
    Y_pred_EM[Y_pred_EM == 0] = -1
    Y_pred_EM[Y_pred_EM == 1] = 1

    # Calculate the accuracy and save the output
    GMM_accuracy = accuracy(Y_pred_EM, Y_test)
    output_file.write("Accuracy: {}\n\n".format(GMM_accuracy))

    # MLE implementation
    output_file.write("MLE Gaussian\n")

    # Separate class 0 and class 1 data and estimate mean and covariance using MLE
    X_class_0 = X_train[Y_train == -1]
    X_class_1 = X_train[Y_train == 1]
    mu_estimate_0, sigma_estimate_0 = MLE_gaussian(X_class_0, len(X_class_0))
    mu_estimate_1, sigma_estimate_1 = MLE_gaussian(X_class_1, len(X_class_1))

    # Wrapping parameters in a list
    theta_0 = [mu_estimate_0, sigma_estimate_0]
    theta_1 = [mu_estimate_1, sigma_estimate_1]

    # Predict using Bayes classifier for Gaussian - Gaussian class conditional densities
    Y_pred_MLE = bayes(theta_0, theta_1, X_test)
    Y_pred_MLE[Y_pred_MLE == 0] = -1
    Y_pred_MLE[Y_pred_MLE == 1] = 1

    # Calculate the accuracy and save the output
    MLE_accuracy = accuracy(Y_pred_MLE, Y_test)
    output_file.write("Accuracy: {}\n\n".format(MLE_accuracy))

    # Nearest neighbour implementation
    output_file.write("Nearest Neighbour\n")

    # Predict using nearest neighbour classifier
    Y_pred_nearest_neighbour = nearest_neighbour(X_class_0, X_class_1, X_test)
    Y_pred_nearest_neighbour[Y_pred_nearest_neighbour == 0] = -1
    Y_pred_nearest_neighbour[Y_pred_nearest_neighbour == 1] = 1

    # Calculate the accuracy and save the output
    nearest_neighbour_accuracy = accuracy(Y_pred_nearest_neighbour, Y_test)
    output_file.write("Accuracy: {}".format(nearest_neighbour_accuracy))

    output_file.close()

import random
from problem_1.MLE_gaussian import MLE_gaussian
from problem_1.MLE_exponential import MLE_exponential
from problem_1.bayes import bayes
from problem_1.accuracy import accuracy
from problem_1.bayes_gaussian_exponential import bayes_gaussian_exponential


# Function for problem 1c
def problem_1c(X_train, Y_train, X_test, Y_test):
    N_list = [5, 10, 25, 75, 100]
    output_file = open('./results/problem_1c.txt', "w")
    output_file.write("Class 1: Gaussian, Class 2: Gaussian\n")

    for N in N_list:
        # Random sample N data points and find the MLE of mean and covariance of class 0 Gaussian
        rand_list = random.sample(range(0, 100), N)
        X_class_0 = X_train[Y_train == -1]
        X_0 = X_class_0[rand_list]
        mu_estimate_0, sigma_estimate_0 = MLE_gaussian(X_0, N)

        # Random sample N data points and find the MLE of mean and covariance of class 1 Gaussian
        rand_list = random.sample(range(0, 100), N)
        X_class_1 = X_train[Y_train == 1]
        X_1 = X_class_1[rand_list]
        mu_estimate_1, sigma_estimate_1 = MLE_gaussian(X_1, N)

        # Wrapping parameters in a list
        theta_0 = [mu_estimate_0, sigma_estimate_0]
        theta_1 = [mu_estimate_1, sigma_estimate_1]

        # Predict using Bayes classifier for Gaussian - Gaussian class conditional densities
        Y_pred_bayes = bayes(theta_0, theta_1, X_test)
        Y_pred_bayes[Y_pred_bayes == 0] = -1
        Y_pred_bayes[Y_pred_bayes == 1] = 1

        # Calculate the accuracy
        bayes_accuracy = accuracy(Y_pred_bayes, Y_test)

        # Save the output
        output_file.write("{}, {}\n".format(N, bayes_accuracy))

    output_file.write("\nClass 1: Gaussian, Class 2: Exponential\n")
    for N in N_list:
        # Random sample N data points and find the MLE of mean and covariance of class 0 Gaussian
        rand_list = random.sample(range(0, 100), N)
        X_class_0 = X_train[Y_train == -1]
        X_0 = X_class_0[rand_list]
        mu_estimate_0, sigma_estimate_0 = MLE_gaussian(X_0, N)

        # Random sample N data points and find the MLE of mean and covariance of class 1 Exponential
        rand_list = random.sample(range(0, 100), N)
        X_class_1 = X_train[Y_train == 1]
        X_1 = X_class_1[rand_list]
        lambda_estimate_1 = MLE_exponential(X_1, N)

        # Wrapping parameters in a list
        theta_0 = [mu_estimate_0, sigma_estimate_0]
        theta_1 = lambda_estimate_1

        # Predict using Bayes classifier for Gaussian - Exponential class conditional densities
        Y_pred_bayes = bayes_gaussian_exponential(theta_0, theta_1, X_test)
        Y_pred_bayes[Y_pred_bayes == 0] = -1
        Y_pred_bayes[Y_pred_bayes == 1] = 1

        # Calculate the accuracy
        bayes_accuracy = accuracy(Y_pred_bayes, Y_test)

        # Save the output
        output_file.write("{}, {}\n".format(N, bayes_accuracy))

    output_file.close()

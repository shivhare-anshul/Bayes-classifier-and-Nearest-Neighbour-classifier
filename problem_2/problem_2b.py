import random
from problem_2.MLE_gaussian import MLE_gaussian
from problem_2.nearest_neighbour import nearest_neighbour
from problem_2.bayes import bayes
from problem_2.accuracy import accuracy


# Function for problem 2b
def problem_2b(X_train, Y_train, X_test, Y_test):
    N_list = [10, 20, 50, 200, 300, 500]
    output_file = open('./results/problem_2b.txt', "w")
    output_file.write("N, bayes, near_neigh\n")

    for N in N_list:
        # Random sample N data points and find the MLE of mean and covariance of class 0 Gaussian
        rand_list = random.sample(range(0, 500), N)
        X_class_0 = X_train[Y_train == -1]
        X_0 = X_class_0[rand_list]
        mu_estimate_0, sigma_estimate_0 = MLE_gaussian(X_0, N)

        # Random sample N data points and find the MLE of mean and covariance of class 1 Gaussian
        rand_list = random.sample(range(0, 500), N)
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

        # Predict using nearest neighbour classifier
        Y_pred_nearest_neighbour = nearest_neighbour(X_0, X_1, X_test)
        Y_pred_nearest_neighbour[Y_pred_nearest_neighbour == 0] = -1
        Y_pred_nearest_neighbour[Y_pred_nearest_neighbour == 1] = 1

        # Calculate the accuracy
        bayes_accuracy = accuracy(Y_pred_bayes, Y_test)
        nearest_neighbour_accuracy = accuracy(Y_pred_nearest_neighbour, Y_test)

        # Save the output
        output_file.write("{}, {}, {}\n".format(N, bayes_accuracy, nearest_neighbour_accuracy))

    output_file.close()

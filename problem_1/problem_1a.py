import random
from problem_1.MLE_gaussian import MLE_gaussian
from problem_1.nearest_neighbour import nearest_neighbour
from problem_1.bayes import bayes
from problem_1.accuracy import accuracy
import matplotlib.pyplot as plt
#import pandas as pd
import statistics
# Function for problem 1a
def problem_1a(X_train, Y_train, X_test, Y_test):
    N_list = [100]
    output_file = open('./results/problem_1a.txt', "w")
    output_file.write("N, bayes, near_neigh\n")

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
        print(theta_0)
        print(theta_1)

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
        
    for i in range(200):
        if Y_test[i]==Y_pred_bayes[i]:
            plt.scatter((X_test.iloc[[i],0]),(X_test.iloc[[i],1]),c='g',marker='o',s=10)
        else:
            plt.scatter((X_test.iloc[[i],0]),(X_test.iloc[[i],1]),c='r',marker='X',s=10)

    for i in range(200):
        if Y_test[i]==Y_pred_nearest_neighbour[i]:
            plt.scatter((X_test.iloc[[i],0]),(X_test.iloc[[i],1]),c='g',marker='o',s=10)
        else:
            plt.scatter((X_test.iloc[[i],0]),(X_test.iloc[[i],1]),c='r',marker='X',s=10)

    plt.show()

        

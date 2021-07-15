import numpy as np


# Function to calculate the accuracy
def accuracy(Y_pred, Y_test):
    accuracy_vector = (Y_pred == Y_test)
    accuracy_value = np.sum(accuracy_vector)/len(accuracy_vector)

    return accuracy_value

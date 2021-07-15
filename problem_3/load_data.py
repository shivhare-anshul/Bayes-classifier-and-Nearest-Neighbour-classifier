import numpy as np

MAX_LINES = 200
DIMENSIONS = 1


# Function to load data from file
def load_from_file(X, Y, filename):
    file = open(filename)
    idx = 0

    for line in file:
        line_list = line.rstrip().split(' ')

        # Remove None entries from list
        line_list = list(filter(None, line_list))

        # Skip blank lines
        if len(line_list) == 0:
            continue

        X[idx] = float(line_list[0])
        Y[idx] = int(line_list[1])
        idx += 1

    file.close()


# Function to load train and test data
def load_data(choice):
    # Matrices to store data
    X_train = np.zeros([MAX_LINES, DIMENSIONS])
    X_test = np.zeros([MAX_LINES, DIMENSIONS])
    Y_train = np.zeros(MAX_LINES)
    Y_test = np.zeros(MAX_LINES)

    # Load train dataset
    filename = 'datasets/P3' + choice + '_train_data.txt'
    load_from_file(X_train, Y_train, filename)

    # Load test dataset
    filename = 'datasets/P3' + choice + '_test_data.txt'
    load_from_file(X_test, Y_test, filename)

    return X_train, Y_train, X_test, Y_test

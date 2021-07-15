from problem_4.BOW_load_data import load_data
from problem_4.BOW_model_train import model_train
from problem_4.BOW_model_test import model_test


# Function for problem 4a
def problem_4a():
    # Load data and train model
    X_train, Y_train, X_test, Y_test = load_data()
    model = model_train(X_train, Y_train)

    # Calculate the accuracy
    accuracy = model_test(model, X_test, Y_test)

    # save the output
    output_file = open('./results/problem_4a.txt', "w")
    output_file.write("Accuracy: {}".format(accuracy))
    output_file.close()

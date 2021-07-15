from problem_4.accuracy import accuracy


# Function to test the model
def model_test(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    BOW_accuracy = accuracy(Y_pred, Y_test)

    return BOW_accuracy

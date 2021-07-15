from sklearn.naive_bayes import MultinomialNB


# Function to train the model
def model_train(X_train, Y_train):
    model = MultinomialNB()

    model.fit(X_train, Y_train)

    return model

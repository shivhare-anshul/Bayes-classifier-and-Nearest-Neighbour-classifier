import os
from problem_3.load_data import load_data
from problem_3.problem_3a import problem_3a
from problem_3.problem_3b import problem_3b


# Create results directory
os.makedirs('results', exist_ok=True)

# Problem 3a
X_train, Y_train, X_test, Y_test = load_data('a')
problem_3a(X_train, Y_train, X_test, Y_test)

# Problem 3b
X_train, Y_train, X_test, Y_test = load_data('b')
problem_3b(X_train, Y_train, X_test, Y_test)
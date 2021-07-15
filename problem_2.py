import os
from problem_2.load_data import load_data
from problem_2.problem_2a import problem_2a
from problem_2.problem_2b import problem_2b
from problem_2.problem_2c import problem_2c


# Create results directory
os.makedirs('results', exist_ok=True)

# Problem 2a
X_train, Y_train, X_test, Y_test = load_data('a')
problem_2a(X_train, Y_train, X_test, Y_test)

# Problem 2b
X_train, Y_train, X_test, Y_test = load_data('b')
problem_2b(X_train, Y_train, X_test, Y_test)

# Problem 2c
X_train, Y_train, X_test, Y_test = load_data('c')
problem_2c(X_train, Y_train, X_test, Y_test)
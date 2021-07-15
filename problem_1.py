import os
from problem_1.load_data import load_data
from problem_1.problem_1a import problem_1a
from problem_1.problem_1b import problem_1b
from problem_1.problem_1c import problem_1c

def main():
    
    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Problem 1a
    X_train, Y_train, X_test, Y_test = load_data('a')
    problem_1a(X_train, Y_train, X_test, Y_test)

    # Problem 1b
    X_train, Y_train, X_test, Y_test = load_data('b')
    problem_1b(X_train, Y_train, X_test, Y_test)

    # Problem 1c
    X_train, Y_train, X_test, Y_test = load_data('c')
    problem_1c(X_train, Y_train, X_test, Y_test)
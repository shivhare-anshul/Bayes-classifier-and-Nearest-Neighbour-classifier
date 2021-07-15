import os
import sys
import numpy as np
from problem_4.problem_4a import problem_4a
from problem_4.problem_4b import problem_4b

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# Create results directory
os.makedirs('results', exist_ok=True)

# Problem 4a
problem_4a()

# Problem 4b
problem_4b()
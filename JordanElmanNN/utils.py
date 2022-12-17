import numpy as np
from math import exp


alpha_for_function = 1


def elu(x):
    return x if x > 0 else alpha_for_function * (exp(x) - 1)


def derivative_elu(x):
    return 1 if x > 0 else exp(x) * alpha_for_function


def calculate_error(delta):
    return np.sum(pow(delta, 2) / 2) / delta.size


def random_array(height, width):
    return np.random.uniform(-1, 1, (height, width))


def elu(x):
    return x if x > 0 else alpha_for_function * (exp(x) - 1)


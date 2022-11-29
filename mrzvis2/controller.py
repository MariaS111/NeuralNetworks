import numpy as np


def func_of_activation(x):
    if x >= 0:
        res = 1
    else:
        res = -1
    return res


def activation_function(num):
    return np.sign(num)


def calculate(weight, matrix):
    weighted_matrix = weight * matrix
    return np.array(list(map(activation_function, weighted_matrix)))


def get_size_of_matrix(matrix):
    return matrix.flatten().size


def make_weight_from_patterns(patterns):
    size_of_matrix = get_size_of_matrix(patterns[0])
    if size_of_matrix / (2 * np.log(size_of_matrix)) < len(patterns):
        print("impossible")
    summa_of_squared_patterns = sum(pattern * pattern.T for pattern in patterns)
    weight_of_neural_net = summa_of_squared_patterns / size_of_matrix
    for i in range(weight_of_neural_net.__len__()):
        weight_of_neural_net[i, i] = 0
    return weight_of_neural_net


def multiplication(e, x):
    return tuple(
        tuple(func_of_activation(sum([e[i][k] * x[k][j] for k in range(len(x))])) for j in range(len(x[0]))) for i
        in range(len(e)))


def make_weights(e, x):
    return tuple(
        tuple(sum([e[i][k] * x[k][j] for k in range(len(x))]) for j in range(len(x[0]))) for i
        in range(len(e)))


def transposition(x):
    return tuple(
        tuple(x[i][j] for i in range(len(x))) for j in range(len(x[0])))


def zeros(x):
    return tuple(
        tuple(0 if k == j else x[j][k] for k in range(len(x[j]))) for j in range(len(x)))


def summ(m1, m2):
    return tuple(tuple(m1[j][i] + m2[j][i] for i in range(len(m1[j]))) for j in range(len(m1)))

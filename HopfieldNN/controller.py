import copy
import numpy as np


class Converter:
    def __init__(self):
        self.files = ('T', 'F')

    def create_pattern(self):
        res = []
        patterns, pattern = [], []
        for i in self.files:
            with open(f'sourse_patterns/{i}', 'r') as file:
                res.append(file.read())
        for i in res:
            pattern = i.replace('.', '-1 ').replace('+', '1 ')
            pattern = [[int(i)] for j in pattern.split('\n') for i in j.split()]
            patterns.append(pattern)
        return patterns

    def read_noisy_pattern(self):
        i = 'F2'
        with open(f'noisy_patterns/{i}', 'r') as file:
            res = file.read()
        result = res.replace('.', '-1 ').replace('+', '1 ')
        result = [[int(i)] for j in result.split('\n') for i in j.split()]
        return result

    def reverse_converter(self, data):
        return data.reshape(10, 10)

    def made_result_pattern(self, data):
        res = data.tolist()
        for i in range(len(res)):
            res[i] = [['+' if i[j] == 1 else '.' for j in range(len(i))] for i in res]
            res[i] = [''.join(i) for i in res[0]]
        return res


class NeuralNetwork:
    def __init__(self, patterns):
        self.patterns = patterns
        for i in range(len(self.patterns)):
            self.patterns[i] = np.array(self.patterns[i])

    def function_of_activation(self, num):
        return -1 if num < 0 else 1

    def multiply(self, weight, matrix):
        matrix = np.array(matrix)
        sign = np.vectorize(self.function_of_activation)
        result = sign(np.dot(weight, matrix))
        #c = Converter()
        for i in range(12):
            result = sign(np.dot(weight, self.asynchronous_execution(result, matrix)))
        return result
        #c.reverse_converter(result)

    def create_weight_matrix(self):
        data = self.patterns
        weights = np.array(np.zeros((self.patterns[0].size, self.patterns[0].size)))
        for i in range(len(data)):
            ups = np.dot((np.dot(weights, data[i]) - data[i]), (np.dot(weights, data[i]) - data[i]).T)
            weights += ups / (np.dot(data[i].T, data[i]) - np.dot(np.dot(data[i].T, weights), data[i]))
        return weights

    def asynchronous_execution(self, new_data, old_data):
        result = copy.deepcopy(new_data)
        for i in range(len(new_data)):
            if i < len(new_data) // 2:
                result[i][0] = new_data[i][0]
            else:
                result[i][0] = old_data[i][0]
        return result

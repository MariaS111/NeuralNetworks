from json import JSONEncoder
import numpy as np
from utils import *
import json


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class NeuralNet:
    def __init__(self, input_sequence, activation_function, derivative_elu, neurons=3, alpha=0.000001):
        self.old_output_layer = None
        self.old_hidden_layer = None
        self.weight_hidden_to_hidden = None
        self.weight_output_to_hidden = None
        self.weight_hidden_to_output = None
        self.weight_input_to_hidden = None
        self.bias_output = None
        self.biases_hidden = None
        self.learning_alpha = alpha
        self.neurons_hidden = neurons
        try:
            self.activation_function = np.vectorize(activation_function)
            self.derivative_of_activation_function = np.vectorize(derivative_elu)
            self.initialize_biases_and_weights(input_sequence)
        except:
            print("error i in intialize")

    def initialize_biases_and_weights(self, input_sequence):
        self.create_biases()
        self.create_weights(input_sequence.shape[0])

    def create_biases(self):
        self.biases_hidden = random_array(self.neurons_hidden, 1)
        self.bias_output = random_array(1, 1)

    def create_weights(self, input_size):
        self.weight_input_to_hidden = random_array(self.neurons_hidden, input_size)
        self.weight_hidden_to_output = random_array(1, self.neurons_hidden)
        self.weight_hidden_to_hidden = random_array(self.neurons_hidden, self.neurons_hidden)
        self.weight_output_to_hidden = random_array(self.neurons_hidden, 1)

    def learning(self, input_sequence, etalon, max_error, file):
        error = self.first_step(input_sequence, np.array(etalon))
        print(error)
        while max_error < error:
            value, error = self.next_step(input_sequence, np.matrix(etalon))
            print(error)
        print(value)
        numpydata ={'W1': self.weight_input_to_hidden, 'W2': self.weight_hidden_to_output, 'W3': self.weight_hidden_to_hidden,
                    'W4': self.weight_output_to_hidden, 'T1': self.biases_hidden, 'T2': self.bias_output, 'hidden': self.old_hidden_layer, 'out': self.old_output_layer}
        with open(f'trained_options/{file}.json', 'w') as f:
            json.dump(numpydata, f, cls=NumpyArrayEncoder)


    def first_step(self, input_sequence, etalon):
        self.old_hidden_layer = np.zeros((self.neurons_hidden, 1))
        self.old_output_layer = np.zeros((1, 1))
        weighted_hidden_layer = np.dot(self.weight_input_to_hidden,
                                       input_sequence) + self.old_hidden_layer + self.old_output_layer - self.biases_hidden
        hidden_layer = self.activation_function(weighted_hidden_layer)
        output_layer = np.dot(self.weight_hidden_to_output, hidden_layer) - self.bias_output
        delta_X = output_layer - etalon
        error = calculate_error(delta_X)
        self.step_of_learning(delta_X, weighted_hidden_layer, hidden_layer, self.old_hidden_layer, input_sequence,
                              self.old_output_layer, output_layer)
        self.old_hidden_layer = hidden_layer
        self.old_output_layer = output_layer
        return error

    def step_of_learning(self, error, weighted_layer, hidden_layer_new, hidden_layer_old, input_layer,
                         output_layer_old, output_layer_new):
        self.weight_hidden_to_output = self.weight_hidden_to_output - np.dot(np.dot(self.learning_alpha, error),
                                                                             hidden_layer_new.T)
        self.change_bias(error)
        self.calculate_new_weight(self.weight_input_to_hidden, weighted_layer, input_layer, error)
        self.calculate_new_weight(self.weight_hidden_to_hidden, weighted_layer, hidden_layer_old, error)
        self.calculate_new_weight(self.weight_output_to_hidden, weighted_layer, output_layer_old, error)
        self.change_biases(error, weighted_layer)
        return self.weight_hidden_to_output

    def calculate_new_weight(self, changeble_layer, weighted_layer, context_layer, error):
        for i in range(changeble_layer.shape[0]):
            for j in range(changeble_layer.shape[1]):
                changeble_layer[i][j] = changeble_layer[i][j] - np.dot(np.dot(np.dot(np.dot(self.learning_alpha, error),
                                                                                     self.weight_hidden_to_output[0][
                                                                                         i]),
                                                                              self.derivative_of_activation_function(
                                                                                  weighted_layer[i][0])),
                                                                       context_layer[j][0])

    def change_biases(self, error, weighted_layer):
        self.biases_hidden = self.biases_hidden + np.dot(np.dot(np.dot(self.learning_alpha, error),
                                                                self.weight_hidden_to_output),
                                                         self.derivative_of_activation_function(weighted_layer))

    def change_bias(self, error):
        self.bias_output = self.bias_output + np.dot(self.learning_alpha, error)

    def next_step(self, input_sequence, etalon):
        weighted_hidden_layer = np.dot(self.weight_input_to_hidden, input_sequence) + \
                                np.dot(self.weight_hidden_to_hidden, self.old_hidden_layer) + \
                                np.dot(self.weight_output_to_hidden, self.old_output_layer) - \
                                self.biases_hidden
        hidden_layer = self.activation_function(weighted_hidden_layer)
        output_layer = np.dot(self.weight_hidden_to_output, hidden_layer) - self.bias_output
        delta_X = output_layer - etalon
        error = calculate_error(delta_X)
        weights = self.step_of_learning(delta_X, weighted_hidden_layer, hidden_layer, self.old_hidden_layer,
                                        input_sequence,
                                        self.old_output_layer, output_layer)
        return output_layer, error

    def calculate(self, input_sequence, file):
        with open(f'trained_options/{file}.json', 'r') as f:
            data = json.load(f)
        weighted_hidden_layer = np.dot(data['W1'], input_sequence) + \
                                np.dot(data['W3'], data['hidden']) + \
                                np.dot(data['W4'], data['out']) - \
                                data['T1']
        hidden_layer = self.activation_function(weighted_hidden_layer)
        output_layer = np.dot(data['W2'], hidden_layer) - data['out']
        numpydata = {'W1': self.weight_input_to_hidden, 'W2': self.weight_hidden_to_output,
                     'W3': self.weight_hidden_to_hidden,
                     'W4': self.weight_output_to_hidden, 'T1': self.biases_hidden, 'T2': self.bias_output,
                     'hidden': self.old_hidden_layer, 'out': self.old_output_layer}
        with open(f'trained_options/{file}.json', 'w') as f:
            json.dump(numpydata, f, cls=NumpyArrayEncoder)
        print(output_layer)

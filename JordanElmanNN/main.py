from NeuralNet import *
from utils import *
import numpy as np
import json


if __name__ == '__main__':
    file = input('Enter name of file:')
    with open(f'source_options/{file}.json', 'r') as f:
        data = json.load(f)
    start = int(input('Enter index of start number:'))
    end = int(input('Enter index of last number:'))
    inpt = np.array([data['input'][0][start:end + 1]]).T
    JL = NeuralNet(inpt, elu, derivative_elu, 1)
    JL.learning(inpt, data['input'][0][end + 1], 0.0001, file)

    # for i in range(2):
    #     start += 1
    #     end += 1
    #     inpt = np.array([data['input'][0][start:end + 1]]).T
    #     JL.calculate(inpt, file)


from controller import *


if __name__ == '__main__':
    converter = Converter()
    patterns = converter.create_pattern()
    neuralnetwork = NeuralNetwork(patterns)
    pattern = input('Enter filename with noisy pattern:')
    resulted_pattern = converter.reverse_converter(neuralnetwork.multiply(neuralnetwork.create_weight_matrix(), converter.read_noisy_pattern(pattern)))
    for i in converter.made_result_pattern(resulted_pattern)[0]:
        print(i)

from controller import *


if __name__ == '__main__':
    converter = Converter()
    patterns = converter.create_pattern()
    neuralnetwork = NeuralNetwork(patterns)
    resulted_pattern = converter.reverse_converter(neuralnetwork.multiply(neuralnetwork.create_weight_matrix(), converter.read_noisy_pattern()))
    for i in converter.made_result_pattern(resulted_pattern)[0]:
        print(i)
        
from controller import *
from data import *
import numpy as np

inpt = np.matrix(([1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1],
                      [-1, -1, 1, -1, -1, -1, -1, 1, 1, -1],
                      [-1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1],
                      [-1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
                      [-1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
                      [-1, -1, 1, -1, -1, -1, -1, 1, -1, -1],
                      [-1, 1, 1, 1, 1, 1, 1, 1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],))

result_matrixes = []
weight = make_weight_from_patterns(patterns)

output_matrix = calculate(weight, inpt)
result_matrixes.append(output_matrix)

while len(result_matrixes) < 4:
    previous_matrix = result_matrixes[-1]
    output_matrix = calculate(weight, previous_matrix)
    result_matrixes.append(output_matrix)
while not (np.array_equal(result_matrixes[-4], result_matrixes[-2])
           and np.array_equal(result_matrixes[-3], result_matrixes[-1])):
    previous_matrix = result_matrixes[-1]
    output_matrix = calculate(weight, previous_matrix)
    result_matrixes.append(output_matrix)

previous_matrix = result_matrixes[-1]
print(previous_matrix)




import numpy as np

inputs = [[1, 2, 3, 4, 5],
          [4, 8, 5, 6, 1],
          [9, 3, 2, 4, 5]]

weights = [[0.2, 0.4, 0.6, 0.8, 1.0],
           [0.1, 0.2, 0.3, 0.4, 0.5],
           [0.1, 0.3, 0.6, 0.9, -0.2]]

biases = [3, 4, 5]

Inputs_shape = np.shape(inputs)
Weights_Shape = np.shape(weights)
Transpose_weight = np.array(weights).T
Transpose_weight_shape = np.shape(Transpose_weight)

print("Before transposing weights the share are = ", Inputs_shape, Weights_Shape)
print("After transpose the shapes are = ", Inputs_shape, Transpose_weight_shape)

final_output = np.dot(inputs, np.array(weights).T) + biases
print(final_output)
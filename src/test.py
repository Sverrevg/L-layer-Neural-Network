import math_operations

Z = [0.56, 0.87, -2.54]

A, cache = math_operations.softmax(Z)
print(math_operations.softmax_backward(A, cache))

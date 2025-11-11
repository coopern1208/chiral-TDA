import numpy as np

base = np.array([[0, 0, 1], [1, 0, 0.7], [0, 2, 0.2]])
arr = np.tile(base, (1000, 1, 1))
print(arr)
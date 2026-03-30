import numpy as np


xs = np.linspace(0.0, 1.0, 10 + 1)
print(xs)
nodes = xs[:, None]
print(nodes)

elements = np.column_stack((
    np.arange(10, dtype=int),
    np.arange(1, 10 + 1, dtype=int),
))

print(elements)

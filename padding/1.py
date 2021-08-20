import numpy as np

import matplotlib.pyplot as plt

x = np.random.rand(10, 10, 1)
def pad(X,pad):
    x_pad = np.pad(X,((pad,pad),(pad,pad),(0,0)),mode = 'constant',constant_values=0)
    return x_pad



x_pad = pad(x,2)

plt.imshow(x_pad)
plt.show()

import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def init_para(vecto_size):
    W = np.random.rand(vecto_size,1)
    b = 0.0
    return W,b
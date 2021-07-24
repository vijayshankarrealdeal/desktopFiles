import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0.0
    return w, b
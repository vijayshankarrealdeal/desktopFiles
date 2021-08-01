import numpy as np


def sigmoid(Z):
    cache = Z
    A = 1 / (1+np.exp(-Z))
    return A,cache

def relu(Z):
    cache = Z
    A = np.maximum(0,Z)
    return A,cache

def relu_backward(dA,cache):
    
    Z = cache
    dZ = np.array(dA,copy = True)
    dZ[Z<=0] = 0
    return dZ

def sigmoid_backward(dA,cache):
    Z = cache
    s = 1 / (1+np.exp(-Z))
    dZ = dA*s*(1-s)
    return dZ
    
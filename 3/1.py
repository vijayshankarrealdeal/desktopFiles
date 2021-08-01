import numpy as np
from utils import relu,sigmoid,sigmoid_backward,relu_backward
def init_deepLayers(layer_dims):
    L = len(layer_dims)
    parameters = {}
    for i in range(1,L):
        parameters['W'+str(i)] = np.random.randn(layer_dims[i],layer_dims[i-1])*0.001
        parameters['b'+str(i)] = np.zeros((layer_dims[i],1))
    
    return parameters

def one_step_forward(A,W,b):
     Z = np.dot(W,A) + b
     cache = (A,W,b)
     return Z,cache
    
def forward_prop(A_prev,W,b,activations):
    A = A_prev
    if activations == 'relu':
        Z,linear_caches = one_step_forward(A, W, b)
        A,activations_cahe = relu(Z)
    elif activations == 'sigmoid':
        Z,linear_caches = one_step_forward(A, W, b)
        A,activations_cahe = sigmoid(Z)
    cache = (linear_caches,activations_cahe)
    return A,cache

def L_forward(X,parameters):
    L = len(parameters)//2
    A = X
    caches = []
    for i in range(1,L):
        A_prev = A
        A,cahce = forward_prop(A_prev, parameters['W'+str(i)], parameters['b'+str(i)], 'relu')
        caches.append(cahce)
    
    AL,cache = forward_prop(A, parameters['W'+str(L)], parameters['b'+str(L)], 'sigmoid')
    caches.append(cache)
    return AL,caches

def count_cost(AL,Y):
    m = Y.shape[1]
    logFn = np.dot(Y,np.log(AL)) + np.dot((1-Y),np.log(1-AL))
    cost = (1/m)*np.sum(logFn)
    cost = np.squeeze(cost) 
    return cost        
        

def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = np.sum(dZ,axis = 1,keepdims = True)/m
    dA_prev = np.dot(W.T,dZ)
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,activation):
    linear_cache,activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA,dW,db = linear_backward(dZ, linear_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA,dW,db = linear_backward(dZ, linear_cache)
        
    return dA,dW,db

def L_backward(AL,Y,cache):
    grads = {}
    L = len(cache)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL =  - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = cache[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    for l in reversed(range(L-1)):
        current_cache = cache[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads
def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads['dW'+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads['db'+str(l+1)]        
    return parameters    
    
    
        



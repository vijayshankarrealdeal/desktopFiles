import numpy as np
import matplotlib.pyplot as plt
from planner_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


X,Y = load_planar_dataset()
#X[0] == Voilet means 0 in Y
#X[1] == Yellow means 1 in Y


#normalize
X = X/4
X

plt.scatter(X[0], X[1],c=Y)

#shapes
X.shape
#(2,400)
Y.shape
#(1,400)

#m Train size
m = X.shape[1]

def layer_size(X,Y):
    n_x = X.shape[0]
    n_h = X.shape[1] + 1
    n_y = Y.shape[0]
    return n_x, n_h , n_y

def init_parameters(n_x,n_h,n_y):
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    parmeters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2
        }
    
    return parmeters
        
def forward_prop(X,parmeters):
    W1 = parmeters['W1']
    b1 = parmeters['b1']
    W2 = parmeters['W2']
    b2 = parmeters['b2']
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1)+ b2
    A2 = sigmoid(Z2)
    
    cache = {
        "Z1":Z1,
        "A1":A1,
        "Z2":Z2,
        "A2":A2
        }
    
    return A2,cache

def cost(A2,Y):
    m = Y.shape[1]
    
    logprobs = np.dot(Y,np.log(A2.T)) + np.dot((1-Y),np.log(1-A2.T))
    cost = -(1/m) * np.sum(logprobs)
    return cost
    
    
def backprop(parmeters,X,Y,caches):
    
    m = Y.shape[1]
    A2 = caches['A2']
    A1 = caches['A1']
    W2 = parmeters['W2']
    
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) * np.sum(dZ2,axis = 1,keepdims = True)
    dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1,2))
    dW1 = (1/m) * np.dot(dZ1,X.T)
    db1 = (1/m) *   np.sum(dZ1,axis = 1,keepdims = True)
    
    grads = {
        'dW2':dW2,
        'db2':db2,
        'dW1':dW1,
        'db1':db1
        }
    return grads
    
def update(grads,parmeters,lr):
    W1 = parmeters['W1']
    W2 = parmeters['W2']
    b1 = parmeters['b1']
    b2 = parmeters['b2']
    W1 = W1 - lr*grads['dW1']
    b1 = b1 - lr*grads['db1']
    W2 = W2 - lr*grads['dW2']
    b2 = b2 - lr*grads['db2']
    parmeters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parmeters
    
def run(X,Y,epochs = 4000,lr = 0.001):
    n_x,n_h,n_y = layer_size(X, Y)   
    parmeters = init_parameters(n_x, n_h, n_y)
    costs = []
    for i in range(epochs):
        A2,cache = forward_prop(X,parmeters)
        cost_ = cost(A2,Y)
        if i % 10 == 0:
            print(cost_)
        costs.append(cost_)
        grads = backprop(parmeters,X,Y,cache)
        parmeters = update(grads, parmeters, lr)
    
    return costs,[i for i in range(epochs)]
        

cost_,epochs = run(X,Y,5000,0.001)
plt.plot(cost_,epochs) 
    

    
    
    
    
    
    
    
    
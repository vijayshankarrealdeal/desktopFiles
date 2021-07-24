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
    
n_x,n_h,n_y = layer_size(X, Y)   
parmeters = init_parameters(n_x, n_h, n_y)
    

    
    
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

    
    
    
    
    
    
    
    
    
    
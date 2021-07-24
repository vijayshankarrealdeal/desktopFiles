import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from helper_fn import sigmoid,initialize_with_zeros

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_sets_train = train_set_x_orig.shape[0]
m_sets_test = test_set_x_orig.shape[0]
pixel_height = train_set_x_orig.shape[1]

#########################
x_flatten_vector_train = train_set_x_orig.reshape(m_sets_train,-1).T
x_flatten_vector_test = test_set_x_orig.reshape(m_sets_test,-1).T

#########
X_train = x_flatten_vector_train/255.
X_test = x_flatten_vector_test/255.



def forward(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    cost = (-1/m)*np.sum(( Y*np.log(A) + (1-Y)*np.log(1-A)))
    print(cost)
    dw = (1/m)*np.dot(X,(A-Y).T)
    db =(1/m)*np.sum(A-Y) 
    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def run_op(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    
    for i in range(num_iterations):
        grads, cost = forward(w, b, X, Y)        
        dw = grads["dw"]
        db = grads["db"]        
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if i % 100 == 0:
            costs.append(cost)
        
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X) + b )        
    for i in range(A.shape[1]):
        
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
            
    return Y_prediction

def run(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    
    w, b = initialize_with_zeros(X_train.shape[0])    
    params, grads, costs = run_op(w, b, X_train, Y_train, num_iterations=100, learning_rate=0.009, print_cost=False)
    
    print(params)
    w = params["w"]
    b = params["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)    
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
        
    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    
    return d



d = run(X_train,train_set_y,X_test,test_set_y,num_iterations=2000, learning_rate=0.005)

    
wei = np.squeeze(d['w'])
y = [i for i in range(12288)]
plt.scatter(y,wei)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



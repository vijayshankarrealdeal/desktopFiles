import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from helper_fn import sigmoid,init_para

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

plt.imshow(train_set_x_orig[125])

m_sets_train = train_set_x_orig.shape[0]
m_sets_test = test_set_x_orig.shape[0]
pixel_height = train_set_x_orig.shape[1]

#########################
x_flatten_vector_train = train_set_x_orig.reshape(-1,m_sets_train)
x_flatten_vector_test = test_set_x_orig.reshape(-1,m_sets_test)

#########
X_train = x_flatten_vector_train/255.
X_test = x_flatten_vector_test/255.

def forward_pass(X,W,b):
    return np.dot(W,X) + b

def activation(z):
    return 

def backward(X,Y,A):
    return np.dot(X,(A-Y).T)

def forward(X,y,W,b):
    m = X.shape[0]
    z = forward(X, y, W, b)
    A = activation(z)
    cost = -1/m * np.sum( np.dot(y,np.log(A)),np.dot( (1-y),np.log(1-A)))
    print(cost)
    dw = (1/m)*backward(X, y, A)
    db = (1/m)*np.sum(A-y)   
    cost = np.squeeze(np.array(cost))
    grads = {
        'dw':dw,
        'db':db,
        }
    return grads,cost

def run_op(w, b, X, Y, num_iterations=100, learning_rate=0.009):
    
      
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        grads,cost = forward(X, Y, w, b)
        #gradient Decent
        dw = grads['dw']
        db = grads['db']
        w -= learning_rate*dw
        b -= learning_rate*db
        costs.append(cost)

    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params,costs,grads

def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
        #(≈ 1 line of code)
    A = sigmoid(np.dot(w.T,X) + b ) 
    for i in range(A.shape[1]):
                #(≈ 4 lines of code)
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0    
    return Y_prediction

def run(X_train,y_train,X_test,y_test,num_iterations=2000, learning_rate=0.05):
    W,b = init_para( X_train.shape[0])
    params,cost,grads = run_op(W, b, X_train,y_train)
    W = params["w"]
    b = params["b"]
    Y_prediction_test = predict(W, b, X_test)
    Y_prediction_train = predict(W, b, X_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
        
    
    d = {"costs": cost,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : W, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d



run(X_train,train_set_y,X_test,test_set_y,num_iterations=2000, learning_rate=0.005)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



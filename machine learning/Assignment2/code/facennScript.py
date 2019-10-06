'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize
import math
import time

t0 = time.time()

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

# Replace this with your sigmoid implementation
def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-1.0 * z))
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    
    #################
    
    training_bias=np.ones(len(training_data))
    training_data=np.column_stack([training_data,training_bias])

    # Hidden Layer
    netj=np.dot(training_data,np.transpose(w1))
    zj=sigmoid(netj)
    zj_bias=np.ones(len(training_data))
    zj=np.column_stack([zj,zj_bias])

    # Output Layer
    netl=np.dot(zj,np.transpose(w2))
    ol=sigmoid(netl)

    # Calculate delta
    y = np.zeros((len(training_data),n_class))
    for x in range(0,len(training_data)):
        y[x][math.floor(training_label[x])]=1
    deltaOl = ol - y

    # Calculate gradience
    gradienceOl = np.dot(np.transpose(deltaOl), zj)
    gradienceHlFirst2Parts = np.dot(deltaOl,w2) * ((1 - zj) * zj)
    gradienceHl = np.dot(np.transpose(gradienceHlFirst2Parts), training_data)
    gradienceHl = gradienceHl[0:n_hidden, :]

    # Calculate J
    J = (-1)*np.sum((np.multiply(y, np.log(ol))) + (np.multiply((1 - y), np.log(1 - ol))))/len(training_data)


   # Calculate Regularization values
    regularization = J + (lambdaval/(2*len(training_data)))*(np.sum(w1*w1) + np.sum(w2*w2))
    regularizedGradienceHl = ((gradienceHl) + (lambdaval * w1)) / len(training_data)
    regularizedGradienceOl = ((gradienceOl) + (lambdaval * w2)) / len(training_data)
    obj_val = regularization
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])
    obj_grad = np.concatenate((regularizedGradienceHl.flatten(), regularizedGradienceOl.flatten()), 0)

    return (obj_val, obj_grad)



    
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    
    # Your code here

    data = np.append(data,np.zeros([len(data),1]),1)
	
	# add bias input node
    data_size=data.shape[0]    
    
    #Feedforward
	
	# product of weights from input to hidden layer and input layer.
    p = np.dot(data,w1.T)
	
	# sigmoid of input layer
    
    z = sigmoid(p)
	
    # add bias hidden node
    
    z = np.append(z,np.zeros([len(z),1]),1)
	
    # product of weights from hidden to output layer and output of hidden layer.
    
    q = np.dot(z,w2.T)
    
    # sigmoid of output layer	
    
    o = sigmoid(q)    
    
    index = np.argmax(o,axis=1)
    
    # predict digits based on maximum value in the output layer.
    
    for i in range(data_size):
        index = np.argmax(o[i])
        labels = np.append(labels,index)

    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 0;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

t1 = time.time()
print ('\n Time taken: ' + str(t1 - t0) + ' seconds, with lambdaval: ' + str(lambdaval) + ' and hidden nodes: ' + str(n_hidden))

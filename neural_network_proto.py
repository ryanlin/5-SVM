#
# Project 3 Neural Networks
# File:    neural_network.py
# Authors: Ruilin Lin (Ryan), Jared Knutson
# Course:  CS491 (Topics) - Machine Learning
# Prof:    Dr. Emily Hand
#


import numpy as np
import matplotlib.pyplot as plt
import random
import math

## Comments are from the Project 3 assignment sheet. Delete as you like ##


# Helper function to evaluate the total loss on the dataset
# model is the current version of the model { ’W1 ’W1, ’b1 ’:b1 , ’W2 ’:W2, ’b2 ’:b2 ’}
# It's a dictionary.
# X is all the training data
# y is the training labels
def calculate_loss(model, X, y):
    N = len(X)
    C = len(X[0])
    yh = np.zeros((N,C))
    loss = 0

    # Calculate yh
    for j, x in enumerate(X):
        z = calculate_z(model,x)
        yh[j] = softmax(z)
    #print("z = ",z)
    #print ("yh = ",yh)

    for n in range(N):
        for i in range(C):
            if y[n] == 0 and i == 0:
                loss += calculate_cross_entropy(1,yh[n][i])
            elif y[n] == 0 and i == 1:
                loss += calculate_cross_entropy(0,yh[n][i])
            elif y[n] == 1 and i == 0:
                loss += calculate_cross_entropy(0,yh[n][i])
            elif y[n] == 1 and i == 1:
                loss += calculate_cross_entropy(1,yh[n][i])

    loss = -loss * (1/N)
    print("loss:",loss)
            
    return 0

def calculate_cross_entropy(y,yh):
    return y * math.log(yh)

# Helper function to predict an output (0 or 1)
# model is the current version of the model { ’W1 ’W1, ’b1 ’:b1 , ’W2 ’:W2, ’b2 ’:b2 ’}
# It's a dictionary.
# x is one sampe (without the label)
def predict(model, x):
    return [0,1]

# This function learns parameters for the neural network and returns the model.
# - X is the training data
# − y is the training labels
# − nn_hdim  Number of nodes in the hidden layer
# − num_passes: Number of passes through the training data for gradient descent
# − print_loss: If True, print the loss every 1000 iteration
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):

    print("--NUM H_DIMENSIONS: {0}--".format((nn_hdim)))

    W1 = np.zeros((len(X[0]), nn_hdim))
    W2 = np.zeros((nn_hdim, len(X[0])))
    b1 = np.zeros((nn_hdim))
    b2 = np.zeros((len(X[0])))
    model = {'W1':W1, 'W2':W2, 'b1':b1, 'b2':b2}

    # Initialize weights with random values
    for m in range(nn_hdim):
        model['b1'][m] = random.uniform(-1,1)
        for n in range(len(X[0])):
            model['b2'][n] = random.uniform(-2,2)
            model['W1'][n][m] = random.uniform(-2,2)
            model['W2'][m][n] = random.uniform(-2,2)

    loss = calculate_loss(model,X,y)

    return model

def calculate_z(model,x):

    a = np.dot(x,model['W1']) + model['b1']
    h = np.tanh(a)
    return np.dot(h,model['W2']) + model['b2']


def softmax(z):
    # Compute softmax values for each sets of scores in x.
    ex = np.exp(z - np.max(z))
    return ex / ex.sum(axis=0)


# Display your decision boundary
def plot_decision_boundary(pred_func, X, y):
  # Set min and max values and give it some padding
  x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
  y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
  h = 0.01

  # Generate a grid of points with distance h between them
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
  
  # Predict the function value for the whole grid
  Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Polot the contour and training examples
  plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

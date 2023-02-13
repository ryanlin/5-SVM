#
# Project 3 Neural Networks
# File:    neural_network.py
# Authors: Ruilin Lin (Ryan), Jared Knutson
# Course:  CS491 (Topics) - Machine Learning
# Prof:    Dr. Emily Hand
#


import numpy as np
import matplotlib.pyplot as plt

## Comments are from the Project 3 assignment sheet. Delete as you like ##


# Helper function to evaluate the total loss on the dataset
# model is the current version of the model { ’W1 ’W1, ’b1 ’:b1 , ’W2 ’:W2, ’b2 ’:b2 ’}
# It's a dictionary.
# X is all the training data
# y is the training labels
def calculate_loss(model, X, y):
  return 0

# Helper function to predict an output (0 or 1)
# model is the current version of the model { ’W1 ’W1, ’b1 ’:b1 , ’W2 ’:W2, ’b2 ’:b2 ’}
# It's a dictionary.
# x is one sampe (without the label)
def predict(mode, x):
  return 0

# This function learns parameters for the neural network and returns the model.
# - X is the training data
# − y is the training labels
# − nn_hdim  Number of nodes in the hidden layer
# − num_passes: Number of passes through the training data for gradient descent
# − print_loss: If True, print the loss every 1000 iteration
def build_model(X, y, nn_hdim, num_passes=20000, print_loss=False):
  return 0

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
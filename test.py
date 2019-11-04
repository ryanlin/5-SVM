#
# Project 3 Neural Networks
# File:    test.py
# Authors: Ruilin Lin (Ryan), Jared Knutson
# Course:  CS491 (Topics) - Machine Learning
# Prof:    Dr. Emily Hand
#

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import neural_network as nn
import random


#This is the dataset used to test our network
np.random.seed(0)
X, y = make_moons(200, noise=0.20)

nn.build_model(X, y, 1)
nn.build_model(X, y, 2)
nn.build_model(X, y, 3)

plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

# # Generate outputs, Use this code
# plt.figure(figsize=(16, 32))
# hidden_layer_dimensions = [1, 2, 3, 4]
# for i, nn_hdim in enumerate(hidden_layer_dimensions):
#   plt.subplot(5, 2, i+1)
#   plt.title('HiddenLayerSize%d' %nn_hdim)
#   model = nn.build_model(X, y, nn_hdim)
#   nn.plot_decision_boundary(lambda x: nn.predict(model, x), X, y)
# plt.show()

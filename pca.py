#
# Project 4 PCA
# File:    pca.py
# Authors: Ruilin Lin (Ryan), Huan Nguyen
# Course:  CS491 (Topics) - Machine Learning
# Prof:    Dr. Emily Hand
#

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

def compute_Z(X, centering=True, scaling=False):
  Z = X
  if centering:
    Z = Z - np.mean(Z,axis=0)
  if scaling:
    Z = Z / np.std(Z, axis=0)
  return Z

def compute_covariance_matrix(Z):
  return np.dot(Z.T, Z)   # (Z^T)(Z)

def find_pcs(COV):
  return linalg.eig(COV)

def project_data(Z, PCS, L, k, var):
  return 0


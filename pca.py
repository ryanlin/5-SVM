#
# Project 4 PCA
# File:    pca.py
# Authors: Ruilin Lin (Ryan), Huan Nguyen
# Course:  CS491 (Topics) - Machine Learning
# Prof:    Dr. Emily Hand
#

import numpy as np
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
  f = COV.shape[0]            # num features
  eig_vals = np.array((1,f))  # eigen values
  eig_vecs = np.array((f,f))  # eigen vectors

  return 0

def project_data(Z, PCS, L, k, var):
  return 0


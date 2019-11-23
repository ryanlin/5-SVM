#
# Project 4 PCA
# File:    pca.py
# Authors: Ruilin Lin (Ryan), Huan Nguyen
# Course:  CS491 (Topics) - Machine Learning
# Prof:    Dr. Emily Hand
#

import numpy as np
from numpy import linalg

def compute_Z(X, centering=True, scaling=False):
  """ Compute centered/scaled matrix Z
    Parameters:
      X : numpy array, holds samples
      centering : bool, if true, center 
      scaling : bool, if true, scale
    Returns:
      Z : numpy array, centered/scaled data
  """
  Z = X   # copy X over into Z
  if centering:
    Z = Z - np.mean(Z,axis=0)   # Z = z - mean(Z) , for all z
  if scaling:
    Z = Z / np.std(Z, axis=0)   # Z = z / std(Z) , for all z
  return Z

def compute_covariance_matrix(Z):
  """ Compute covariance matrix (Z^T)(Z)
    Parameters:
      Z : numpy array, centered/scaled data
    Returns:
      COV : numpy array, covariance matrix
  """
  return np.dot(Z.T, Z)   # (Z^T)(Z)

def find_pcs(COV):
  """ Computes Principal Components from COV. 
    Parameters:
      COV : numpy array, covariance matrix from data
    Returns:
      L : numpy array, eigenvalues (variance explained by respective eigenvector
      pcs : numpy array, eigenvectors (principal components)
  """
  # find eigenvalues (variance L) and eigenvectors (pcs)
  return linalg.eig(COV)

def project_data(Z, PCS, L, k, var):
  """ Projects data into space of k dimensions
    Parameters:
      Z : numpy array, centered/scaled data
      PCS : numpy array, principal components
      L : numpy array, variance explained by respective PC
      k : int, desired dimensionality for projection
      var : int, desired cumulative/total variance explained by projection
    Returns:
      Z_star : numpy array, projected data
  """
  if( k > 0):
    # If k > 0, project data with k principal comps
    Z_star = np.dot(Z, PCS[:k].T)
  else:
    # If k = 0, include k PCs until var is reached
    cumulative_var = L[0]
    k = 1
    while(cumulative_var < var):
      cumulative_var += L[i]
      k += 1
    Z_star = np.dot(Z, PCS[:k].T)
  
  return Z_star
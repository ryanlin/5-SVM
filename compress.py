#
# Project 4 PCA
# File:    compress.py
# Authors: Ruilin Lin (Ryan), Huan Nguyen
# Course:  CS491 (Topics) - Machine Learning
# Prof:    Dr. Emily Hand
#

import os
import numpy as np
import matplotlib.pyplot as plt
import pca

def compress_images(DATA,k):
  X_mean = np.mean(DATA.T,axis=0)
  Z = pca.compute_Z(DATA.T)
  COV = pca.compute_covariance_matrix(Z)
  COV = (COV + COV.T) / 2
  L, PCS = pca.find_pcs(COV)
  Z_star = pca.project_data(Z, PCS, L, k, 0) 
  DATA_compress = np.matmul(Z_star, PCS[:,:k].T) + X_mean
  output_dir = "./Output/"
  # check directory's existence
  if not os.path.exists(output_dir):
    # create directory
    try:
      os.mkdir(output_dir)
    except OSError:
      return 0 
	  #print ("Creation of the directory %s failed" % output_dir)
    #else:
      #print ("Successfully created the directory %s " % output_dir)
  # scale pixel value to range [0-255] and round off
  image_num = len(DATA_compress)
  for i in range(image_num):
    max_val = DATA_compress[i,:].max()
    min_val = DATA_compress[i,:].min()
    if ((min_val < 0) or (max_val > 255)):
      DATA_compress[i,:] = (DATA_compress[i,:] - min_val) * 255 / (max_val - min_val)
    DATA_compress[i,:] = np.rint(DATA_compress[i,:])   
    
    # display image
    image_compress = np.reshape(DATA_compress[i,:], (60, 48))
    # plt.imshow(image_compress, cmap='gray', interpolation='bilinear')
    # plt.show()

    # save to output folder
    file_name = os.path.join(output_dir, str(i) + "_k" + str(k) + ".png")
    plt.imsave(file_name, image_compress, cmap='gray', format='png')
  return 0

def readpgm(name):
  data = plt.imread(name, '.pgm')
  return data

def load_data(input_dir):
  data_full = []
  for file in os.listdir(input_dir):
    if file.endswith(".pgm"):
      file_name = os.path.join(input_dir, file)
      #print(file_name)
      data = readpgm(file_name)
      data_full.append(data.flatten())  
  return np.transpose(np.array(data_full, dtype=float))
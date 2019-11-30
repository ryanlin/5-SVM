import pca
import compress
import numpy as np

# pca test stuff
X = np.array ([ [-1, -1], [-1,1], [1, -1], [1, 1] ])
#X = np.array ([ [-1, -1], [-2, -2], [1, 1], [2, 2] ])
Z = pca.compute_Z(X)
COV = pca.compute_covariance_matrix(Z)
L, PCS = pca.find_pcs(COV)
Z_star = pca.project_data(Z, PCS, L, 1, 0)
#print(Z_star)

# compress test stuff
X = compress.load_data('Data/Train/')
compress.compress_images(X,100)
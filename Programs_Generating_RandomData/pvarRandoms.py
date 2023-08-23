from scipy.io import savemat
import numpy as np

p = np.random.poisson(0.2,size= 1000*3)
savemat('Matlab_Data/runme_generate_training_data_pvar.mat', {'p': p})
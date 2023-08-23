from scipy.io import savemat
import numpy as np
n = np.array(np.ceil(np.random.rand(1000)*6)) + 1
#should not be integers here

x = np.array(np.ceil(np.random.rand(1000*5) * 2))
savemat('../Matlab_Data/genLutBRandomData.mat', {'randomData': x, 'n': n})
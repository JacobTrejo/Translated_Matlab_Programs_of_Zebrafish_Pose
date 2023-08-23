from scipy.io import savemat
import numpy as np

x = np.array(np.ceil(np.random.rand(1000*11) * 2))
savemat('../Matlab_Data/bellyModelRandomData.mat', {'randomData': x})

from scipy.io import savemat
import numpy as np

x = np.array((np.random.rand(1000*22) * 2 + .001))
y = np.array((np.random.rand(1000*3 ) * 360))
savemat('../Matlab_Data/reorient_model_RandomData.mat', {'randomData': x, 'randomAngles': y})


from scipy.io import savemat
import numpy as np

x = np.array((np.random.rand(1000*5) * 2))
y = np.array((np.random.rand(1000*5) * 2))
z = np.array((np.random.rand(1000*5) * 2))
m = np.array((np.random.rand(1000*5) * 2))
savemat('../Matlab_Data/project_camera_copy_RandomData.mat', {'xvals': x, 'yvals': y, 'zvals': z, 'mvals': m})
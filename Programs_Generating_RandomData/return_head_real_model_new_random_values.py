from scipy.io import savemat
import numpy as np
a = np.array(np.random.rand(1000*19)*2*np.pi)
#should not be integers here
fl = np.array(np.random.rand(1000) * 2)

r = np.array(np.random.rand(1000*5))

x = np.array(np.random.rand(1000*3))
savemat('../Matlab_Data/return_head_RandomData.mat', {'x': x, 'a': a, 'fl': fl, 'r':r})



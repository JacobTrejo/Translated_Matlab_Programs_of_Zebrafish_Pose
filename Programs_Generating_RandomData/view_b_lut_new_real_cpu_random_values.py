from scipy.io import savemat
import numpy as np

r1 = np.array((np.random.rand(1000*10))) + 35
r2 = np.array((np.random.rand(1000*10))) + 35

c = np.array((np.random.rand(1000*2) * 6))

rands = np.array((np.random.rand(1000*2)))

savemat('../Matlab_Data/view_b_lut_new_real_cpu_RandomData.mat', {'r1': r1, 'r2': r2, 'c': c, 'rands':rands})
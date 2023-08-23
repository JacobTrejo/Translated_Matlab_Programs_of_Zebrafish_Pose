from scipy.io import savemat
import numpy as np

r = np.random.rand(1000*13)
fl = np.random.normal(3.8,0.15,size=1000)
n1 = np.random.normal(50,10,size=1000)
n2 = np.random.normal(20,10,size=1000*2)

#Things in patch_noise
c = np.random.randint(1, high=142, size=(1000*613*2))
I = np.random.randint(5, high=21,size=1000*613*2)
rL = np.random.rand(613)

savemat('../Matlab_Data/runme_generate_training_data_RandomData.mat', {'r': r, 'fl': fl, 'n1':n1, 'n2':n2, 'c':c, 'I':I, 'rL':rL})
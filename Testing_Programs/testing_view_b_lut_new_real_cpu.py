import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Translated_Python_Programs_Without_Randoms import view_s_lut_new_real_cpu_NORANDOM
from Translated_Python_Programs_Without_Randoms import view_b_lut_new_real_cpu_NORANDOM

outputMat = loadmat('../Matlab_Data/view_b_lut_new_real_cpu_MatlabOutput.mat')
outputArr = outputMat['matlabOutput']

dataMat = loadmat('../Matlab_Data/view_b_lut_new_real_cpu_RandomData.mat')
r1vals = dataMat['r1']
r1vals = r1vals[0]
r2vals = dataMat['r2']
r2vals = r2vals[0]
cvals = dataMat['c']
cvals = cvals[0]
randvals = dataMat['rands']
randvals = randvals[0]

x = 0
rstep = 10
cstep = 2
randstep = 2

proj = np.zeros((100,100))

arr = []

while x < 1000:
    ry = x * rstep
    cy = x * cstep
    rndy = x * randstep

    pt = [[r1vals[ry],r1vals[ry + 1],r1vals[ry + 2],r1vals[ry + 3],r1vals[ry + 4],r1vals[ry + 5],r1vals[ry + 6],r1vals[ry + 7],r1vals[ry + 8],r1vals[ry + 9]],[r2vals[ry],r2vals[ry + 1],r2vals[ry + 2],r2vals[ry + 3],r2vals[ry + 4],r2vals[ry + 5],r2vals[ry + 6],r2vals[ry + 7],r2vals[ry + 8],r2vals[ry + 9]]]
    pt = np.array(pt)
    pt = pt + 35

    crop_coor = [cvals[cy]+24,0,cvals[cy +1]+24]
    crop_coor = np.array(crop_coor)


    a = view_b_lut_new_real_cpu_NORANDOM.view_b_lut_new_real_cpu(crop_coor, pt, 1, proj, 100, 100, randvals[rndy], randvals[rndy + 1])
    jj = outputArr[:,:,x]
    diff = a - outputArr[:,:,x]
    sum = np.sum(diff)
    arr.append(sum)

    #print(sum)

    x = x +1


arr = np.array(arr)
plt.hist(arr, density=True, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(arr.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("view_b_lut_new_real_cpu Error Distribution")
plt.show()
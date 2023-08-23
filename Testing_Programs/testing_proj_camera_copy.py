import numpy as np
from scipy.io import loadmat
from Translated_Python_Programs import project_camera_copy
import matplotlib.pyplot as plt
outputMat = loadmat('../Matlab_Data/proj_camera_copy_MatlabOutput.mat')
outputa = outputMat['outputa']
outputb = outputMat['outputb']
outputc = outputMat['outputc']

dataMat = loadmat('../Matlab_Data/project_camera_copy_RandomData.mat')
xvals = dataMat["xvals"]
yvals = dataMat["yvals"]
zvals = dataMat["zvals"]
mvals = dataMat["mvals"]

xvals = xvals[0]
yvals = yvals[0]
zvals = zvals[0]
mvals = mvals[0]

x = 0
step = 5

cb = np.array([259,360,314,414])
cs1 = np.array([1186,1232,306,336])
cs2 = np.array([1211,1259,314,345])
indices = np.array([0,1,2,3,4])

arr = []

while x < 1000:
    y = x * step
    xarg = [xvals[y+0],xvals[y+1],xvals[y+2],xvals[y+3],xvals[y+4]]
    yarg = [yvals[y + 0], yvals[y + 1], yvals[y + 2], yvals[y + 3], yvals[y + 4]]
    zarg = [zvals[y + 0], zvals[y + 1], zvals[y + 2], zvals[y + 3], zvals[y + 4]]
    marg = [mvals[y + 0], mvals[y + 1], mvals[y + 2], mvals[y + 3], mvals[y + 4]]


    (a,b,c) = project_camera_copy.project_camera_copy(marg, xarg, yarg, zarg, "../Matlab_Data/proj_params_101019_corrected_new", indices, cb, cs1, cs2)
    # if x == 0 :
    #     j = a
    #     k = outputa[:,:,0]




    diffa = a - outputa[:,:,x]
    diffb = b - outputb[:,:,x]
    diffc = c - outputc[:,:,x]

    sum = np.sum(diffa)
    sum2 = np.sum(diffb)
    sum3 = np.sum(diffc)
    s = sum + sum2 + sum3

    arr.append(s)
    if s != 0.0:
            print('The sum below is not 0')
            print(x)

    print(sum + sum2 + sum3)


    x = x+1

arr = np.array(arr)

plt.hist(arr, density=True, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(arr.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("project_camera_copy Error Distribution")
plt.show()











# coors = np.array([xvals,yvals,zvals])
# (a,b,c) = calc_proj_w_refra_cpu_v3.calc_proj_w_refra_cpu(coors,'proj_params_101019_corrected_new')
#
#
# cb2 = min(a[0,:])
# cb3 = max(a[0,:])
# cb0 = min(a[1,:])
# cb1 = max(a[1,:])
#
# cs1_2 = min(b[0,:])
# cs1_3 = max(b[0,:])
# cs1_0 = min(b[1,:])
# cs1_1 = max(b[1,:])
#
# cs2_2 = min(c[0,:])
# cs2_3 = max(c[0,:])
# cs2_0 = min(c[1,:])
# cs2_1 = max(c[1,:])
#
#
#
# print(cb2,cb3,cb0,cb1)
# print(cs1_2,cs1_3,cs1_0,cs1_1)
# print(cs2_2,cs2_3,cs2_0,cs2_1)
import numpy as np
from scipy.io import loadmat
from Translated_Python_Programs_Without_Randoms import return_head_real_model_new_NORANDOMS

outputMat = loadmat('../Matlab_Data/return_head_MatlabOutputAll.mat')
# outputa = outputMat['outputa']

outputGray_b = outputMat['outputGray_b']
outputGray_s1 = outputMat['outputGray_s1']
outputGray_s2 = outputMat['outputGray_s2']

outputEyeb = outputMat['outputEyeb']
outputEyes1 = outputMat['outputEye2']
outputEyes2 = outputMat['outputEye3']

outputLast = outputMat['outputLast']





dataMat = loadmat('../Matlab_Data/return_head_RandomData.mat')
xvals = dataMat["x"]
avals = dataMat["a"]
flvals = dataMat["fl"]
rvals = dataMat["r"]

xvals = xvals[0]
avals = avals[0]
flvals = flvals[0]
rvals = rvals[0]

q = 0
xstep = 3
astep = 19
rstep = 5

cb = np.array([259,360,314,414])
cs1 = np.array([1186,1232,306,336])
cs2 = np.array([1211,1259,314,345])

arr = []

while q < 100:
    xy = q * xstep
    ay = q * astep
    ry = q * rstep

    xArr = [xvals[xy],xvals[xy+1],xvals[xy+2],avals[ay],avals[ay+1],avals[ay+2],avals[ay+3],avals[ay+4],avals[ay+5],avals[ay+6],avals[ay+7],avals[ay+8],avals[ay+9],avals[ay+10],avals[ay+11],avals[ay+12],avals[ay+13],avals[ay+14],avals[ay+15],avals[ay+16],avals[ay+17],avals[ay+18]]


    (graymodel_b, graymodel_s1, graymodel_s2, eye_b, eye_s1, eye_s2, eye_3d_coor) = return_head_real_model_new_NORANDOMS.return_head_real_model_new(xArr, flvals[q], "../Matlab_Data/proj_params_101019_corrected_new", cb, cs1, cs2, rvals[ry], rvals[ry + 1], rvals[ry + 2], rvals[ry + 3], rvals[ry + 4])

    diffGrayb = graymodel_b - outputGray_b[:,:,q]
    graybSum = np.sum(diffGrayb)
    diffGrays1 = graymodel_s1 - outputGray_s1[:,:,q]
    grays1Sum = np.sum(diffGrays1)
    diffGrays2 = graymodel_s2 - outputGray_s2[:,:,q]
    grays2Sum = np.sum(diffGrays2)

    diffEyeb = eye_b - outputEyeb[:,:,q]
    EyebSum = np.sum(diffEyeb)
    diffEyes1 = eye_s1 - outputEyes1[:,:,q]
    Eyes1Sum = np.sum(diffEyes1)
    diffEyes2 = eye_s2 - outputEyes2[:,:,q]
    Eyes2Sum = np.sum(diffEyes2)

    diffLast = eye_3d_coor - outputLast[:,:,q]
    LastSum = np.sum(diffLast)

    TotalSum = graybSum + grays1Sum + grays2Sum + EyebSum \
    + Eyes1Sum + Eyes2Sum + LastSum


    arr.append(TotalSum)

    # j = gray1
    # k = outputa[:,:,q]
    #
    # diff = gray1 - outputa[:,:,q]






    # sum = np.sum(diff**2)
    print('Trial: '+str(q))
    # print('TotalSum: '+str(TotalSum))



    q = q+1

import matplotlib.pyplot as plt
arr = np.array(arr)

plt.hist(arr, density=True, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(arr.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("return_head_model_new Error Distribution")
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
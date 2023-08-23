import numpy as np
from scipy.io import loadmat
from Translated_Python_Programs_Without_Randoms import return_graymodels_fish_NORANDOM
import time
import matplotlib.pyplot as plt

outputMat = loadmat('../Matlab_Data/return_graymodels_fish_MatlabOutputAll.mat')
outputGray_b = outputMat['outputGray_b']
outputGray_s1 = outputMat['outputGray_s1']
outputGray_s2 = outputMat['outputGray_s2']
outputCrop_b = outputMat['outputCrop_b']
outputCrop_s1 = outputMat['outputCrop_s1']
outputCrop_s2 = outputMat['outputCrop_s2']
outputAnnotated_b = outputMat['outputAnnotated_b']
outputAnnotated_s1 = outputMat['outputAnnotated_s1']
outputAnnotated_s2 = outputMat['outputAnnotated_s2']
outputEye_b = outputMat['outputEye_b']
outputEye_s1 = outputMat['outputEye_s1']
outputEye_s2 = outputMat['outputEye_s2']
outputCoor_3d = outputMat['outputCoor_3d']


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
rstep = 6

#startTime = time.time()
# diffArr = np.zeros((800))
diffArr = []
while q < 100:

    xy = q * xstep
    ay = q * astep
    ry = q * rstep

    xArr = [xvals[xy],xvals[xy+1],xvals[xy+2],avals[ay],avals[ay+1],avals[ay+2],avals[ay+3],avals[ay+4],avals[ay+5],avals[ay+6],avals[ay+7],avals[ay+8],avals[ay+9],avals[ay+10],avals[ay+11],avals[ay+12],avals[ay+13],avals[ay+14],avals[ay+15],avals[ay+16],avals[ay+17],avals[ay+18]]

    rArr = [rvals[ry] ,rvals[ry+1],rvals[ry+2],rvals[ry+3],rvals[ry+4],rvals[ry+5]]
    rArr = np.array(rArr)
    # if q == 6:
    #     #  1       2       3        4        5       6          7             8            9         10       11      12    13
    #     [gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2,annotated_b, annotated_s1, annotated_s2, eye_b, eye_s1, eye_s2,coor_3d] = \
    #       return_graymodels_fish_NORANDOM.return_graymodels_fish(\
    #          xArr, 1, 1, "../Matlab_Data/proj_params_101019_corrected_new", flvals[q], 100, 100, rArr)
    #     out = gray_s1
    #     #k = outputa[:,:,q]
    #
    #     diff = out - outputa[:,:,q]
    #     'block'

        #  1       2       3        4        5       6          7             8            9         10       11      12    13
    [gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2,annotated_b, annotated_s1, annotated_s2, eye_b, eye_s1, eye_s2,coor_3d] = \
      return_graymodels_fish_NORANDOM.return_graymodels_fish(\
         xArr, 1, 1, "../Matlab_Data/proj_params_101019_corrected_new", flvals[q], 100, 100, rArr)
    #k = outputa[:,:,q]

    # out = crop_s1
    # diff = out - outputCrop_s1[:,:,q]

    diffGray_b = gray_b - outputGray_b[:,:,q]
    Gray_bSum = np.sum(diffGray_b)
    diffGray_s1 = gray_s1 - outputGray_s1[:,:,q]
    Gray_s1Sum = np.sum(diffGray_s1)
    diffGray_s2 = gray_s2 - outputGray_s2[:,:,q]
    Gray_s2Sum = np.sum(diffGray_s2)

    diffCrop_b = crop_b - outputCrop_b[:,:,q]
    Crop_bSum = np.sum(diffCrop_b)
    diffCrop_s1 = crop_s1 - outputCrop_s1[:,:,q]
    Crop_s1Sum = np.sum(diffCrop_s1)
    diffCrop_s2 = crop_s2 - outputCrop_s2[:,:,q]
    Crop_s2Sum = np.sum(diffCrop_s2)

    diffAnnotated_b = annotated_b - outputAnnotated_b[:,:,q]
    Annotated_bSum = np.sum(diffAnnotated_b)
    diffAnnotated_s1 = annotated_s1 - outputAnnotated_s1[:,:,q]
    Annotated_s1Sum = np.sum(diffAnnotated_s1)
    diffAnnotated_s2 = annotated_s2 - outputAnnotated_s2[:,:,q]
    Annotated_s2Sum = np.sum(diffAnnotated_s2)

    diffEye_b = eye_b - outputEye_b[:,:,q]
    Eye_bSum = np.sum(diffEye_b)
    diffEye_s1 = eye_s1 - outputEye_s1[:,:,q]
    Eye_s1Sum = np.sum(diffEye_s1)
    diffEye_s2 = eye_s2 - outputEye_s2[:,:,q]
    Eye_s2Sum = np.sum(diffEye_s2)

    diffCoor_3d = coor_3d - outputCoor_3d[:,:,q]
    Coor_3dSum = np.sum(diffCoor_3d)

    TotalSum = Gray_bSum + Gray_s1Sum + Gray_s2Sum + Crop_bSum + Crop_s1Sum + Crop_s2Sum + Annotated_bSum + Annotated_s1Sum\
    + Annotated_s2Sum + Eye_bSum + Eye_s1Sum + Eye_s2Sum + Coor_3dSum


    #Beware of trial 7 sum not being around 0 due to one pixel being off
    print("Trial: "+ str(q+1))
    print('Total Sum: '+ str(TotalSum))
    diffArr.append(TotalSum)



    q = q+1

diffArr = np.array(diffArr)
plt.hist(diffArr, density=True, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(diffArr.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("return_graymodels_fish Error Distribution")
plt.show()


# y = np.argwhere(diffArr)
# print(y)










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
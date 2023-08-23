import numpy as np
from scipy.io import loadmat
from Translated_Python_Programs import reorient_model
import matplotlib.pyplot as plt

outputMat = loadmat('../Matlab_Data/reorient_modelMatlabOutput.mat')
outputArr = outputMat['matlabOutput']
#print("dataArr")
#print(outputArr)

dataMat = loadmat('../Matlab_Data/reorient_model_RandomData.mat')
dataArr = dataMat["randomData"]
dataArr = dataArr[0,:]

angleArr = dataMat["randomAngles"]
angleArr = angleArr[0,:]


differenceArr = np.empty((1000,16))
count = 0
step = 22
anglestep = 3
flag = True
xarr = np.zeros((1000))
while count < 1000:
    y = count * step
    angleY = count * anglestep

    firstArr = np.array([dataArr[y], dataArr[y + 1], dataArr[y + 2], dataArr[y + 3]])
    secondArr = np.array([dataArr[y+4], dataArr[y + 5], dataArr[y + 6], dataArr[y + 7]])
    thirdArr = np.array([dataArr[y+8], dataArr[y + 9], dataArr[y + 10], dataArr[y + 11]])
    fourthArr = np.array([dataArr[y+12], dataArr[y + 13], dataArr[y + 14], dataArr[y + 15]])
    fifthArr = np.array([dataArr[y+16], dataArr[y + 17], dataArr[y + 18]])
    sixthArr = np.array([dataArr[y+19], dataArr[y + 20], dataArr[y + 21]])


    (x,y,z,indices) = reorient_model.reorient_model(firstArr, secondArr, thirdArr, fourthArr, angleArr[angleY], angleArr[angleY + 1], angleArr[angleY + 2], fifthArr, sixthArr)
    # if flag:
    #     print('inputs')
    #     # print(firstArr)
    #     # print(secondArr)
    #     # print(thirdArr)
    #     # print(fourthArr)
    #     # print(angleArr[angleY])
    #     # print(angleArr[angleY+1])
    #     # print(angleArr[angleY+2])
    #     # print(fifthArr)
    #     # print(sixthArr)
    #     print(angleArr[angleY])
    #     print(angleArr[angleY+1])
    #     print(angleArr[angleY+2])
    #
    #
    #
    #     print("first Values")
    #     print(x)
    #     print(y)
    #     print(z)
    #     flag = False



    xDiff = outputArr[count, 0:4] - x[0,:]
    yDiff = outputArr[count, 4:8] - y[0,:]
    zDiff = outputArr[count, 8:12] - z[0,:]
    indicesDiff = outputArr[count, 12:16] - (indices + 1)

    differenceArr[count, 0:4] = xDiff
    differenceArr[count, 4:8] = yDiff
    differenceArr[count, 8:12] = zDiff
    differenceArr[count, 12:16] = indicesDiff


    print(xDiff[0,0:3])
    # xprime = (np.square(xDiff)).mean()
    # y = (np.square(yDiff)).mean()
    # z = (np.square(zDiff)).mean()
    # i = (np.square(indicesDiff).mean())

    xprime = (xDiff).mean()
    y = (yDiff).mean()
    z = (zDiff).mean()
    i = (indicesDiff.mean())

    totalDiff = (xprime + y + z + i)/4
    xarr[count] = totalDiff

    count = count + 1
mseArr = np.mean(np.square(differenceArr), axis=0)
print('mseArr')
print(mseArr)

plt.hist(xarr, density=True, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(xarr.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("Reorient Model Error Distribution")
plt.show()



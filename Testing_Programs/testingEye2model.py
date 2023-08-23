import numpy as np
from scipy.io import loadmat
from Translated_Python_Programs_Without_Randoms import eye2modelNoRandom
import matplotlib.pyplot as plt


outputMat = loadmat('../Matlab_Data/eye2ModelMatlabOutput.mat')
outputArr = outputMat['matlabOutput']
print("line 4")
dataMat = loadmat('../Matlab_Data/eye2ModelRandomData.mat')
dataArr = dataMat["randomData"]

differenceArr = np.empty((1000,4))
count = 0
step = 10
x = np.zeros((1000))
while count < 1000:
    y = count * step

    pythonOutput = eye2modelNoRandom.eye2model(dataArr[0,y], dataArr[0, y + 1], dataArr[0, y + 2], dataArr[0, y + 3], dataArr[0, y + 4], dataArr[0, y + 5], np.array([dataArr[0, y + 6], dataArr[0, y + 7], dataArr[0, y + 8], dataArr[0, y + 9]]))
    differenceArr[count, 0] = pythonOutput[0] - outputArr[count,0]
    differenceArr[count, 1] = pythonOutput[1][0] - outputArr[count, 1]
    differenceArr[count, 2] = pythonOutput[1][1] - outputArr[count, 2]
    differenceArr[count, 3] = pythonOutput[1][2] - outputArr[count, 3]

    diff1 = pythonOutput[0] - outputArr[count,0]
    diff2 = pythonOutput[1][0] - outputArr[count, 1]
    diff3 = pythonOutput[1][1] - outputArr[count, 2]
    diff4 = pythonOutput[1][2] - outputArr[count, 3]

    # diff1 = diff1**2
    # diff2 = diff2**2
    # diff3 = diff3**2
    # diff4 = diff4**2


    diff = (diff1 + diff2 + diff3 + diff4)/4
    x[count] = diff

    count += 1
mseArr = np.mean(np.square(differenceArr), axis=0)
print(mseArr)
print(mseArr[0]/4)

plt.hist(x, density=True, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("Eye2 Model Error Distribution")
plt.show()


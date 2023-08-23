import numpy as np
from scipy.io import loadmat
from Translated_Python_Programs_Without_Randoms import headmodelNoRandom
import matplotlib.pyplot as plt


outputMat = loadmat('../Matlab_Data/headModelMatlabOutput.mat')
outputArr = outputMat['matlabOutput']

dataMat = loadmat('../Matlab_Data/headModelRandomData.mat')
dataArr = dataMat["randomData"]

differenceArr = np.empty((1000,1))
count = 0
step = 10
x = np.zeros(1000)
while count < 1000:
    y = count * step

    pythonOutput = headmodelNoRandom.headmodel(dataArr[0,y], dataArr[0, y + 1], dataArr[0, y + 2], dataArr[0, y + 3], dataArr[0, y + 4], dataArr[0, y + 5], dataArr[0, y + 6], dataArr[0, y + 7], dataArr[0, y + 8], dataArr[0, y + 9])
    differenceArr[count, 0] = pythonOutput - outputArr[count,0]
    diff = pythonOutput - outputArr[count,0]
    #diff = diff **2
    x[count] = diff
    count += 1
mseArr = np.mean(np.square(differenceArr), axis=0)
print(mseArr)
plt.hist(x, density=True, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("Head Model Error Distribution")
plt.show()



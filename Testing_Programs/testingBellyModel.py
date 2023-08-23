import numpy as np
from scipy.io import loadmat
from Translated_Python_Programs_Without_Randoms import bellymodelNoRandom
import matplotlib.pyplot as plt

outputMat = loadmat('../Matlab_Data/bellyModelMatlabOutput.mat')
outputArr = outputMat['matlabOutput']

dataMat = loadmat('../Matlab_Data/bellyModelRandomData.mat')
dataArr = dataMat["randomData"]

differenceArr = np.empty((1000,1))
count = 0
step = 11
x = np.zeros((1000))
print('zeros')
print(x.shape)

while count < 1000:
    y = count * step

    pythonOutput = bellymodelNoRandom.bellymodel(dataArr[0,y], dataArr[0, y + 1], dataArr[0, y + 2], dataArr[0, y + 3], dataArr[0, y + 4], dataArr[0, y + 5], dataArr[0, y + 6], dataArr[0, y + 7], dataArr[0, y + 8], dataArr[0, y + 9], dataArr[0, y + 10])
    dif = pythonOutput - outputArr[count,0]
    #dif = dif**2

    x[count] = dif
    differenceArr[count, 0] = pythonOutput - outputArr[count,0]
    count += 1
mseArr = np.mean(np.square(differenceArr), axis=0)
print(mseArr)

plt.hist(x, density=False, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("Belly Model Error Distribution")
plt.show()



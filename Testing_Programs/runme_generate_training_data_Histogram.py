import numpy as np
import  matplotlib.pyplot as plt
from scipy.io import loadmat

testingOutput = loadmat('../Matlab_Data/runme_generate_training_data_100Times.mat')
diffArr = testingOutput['allImgDifferences']
diffArr = diffArr[0]
pixelsInImage = 59643

AverageList = []
diffCount = 0
counter = 0

for difference in diffArr:
    diffCount += ((float(difference) **2) ** .5)
    counter +=1
    if counter == pixelsInImage:
        counter = 0
        avg = float(diffCount)/float(pixelsInImage)
        diffCount = 0

        print(avg)
        AverageList.append(avg)

AverageList = np.array(AverageList)

plt.hist(x=AverageList,bins=30,color='#0504aa')
plt.ylabel('Count')
plt.xlabel('Average Pixel Difference Between Images')
plt.title("Average Pixel Difference Between First 100 Images")
plt.axvline(np.mean(AverageList), color='k', linestyle='dashed', linewidth=1)


plt.show()
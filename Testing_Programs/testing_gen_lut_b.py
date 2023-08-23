import numpy as np
from scipy.io import loadmat

from Translated_Python_Programs_Without_Randoms import gen_lut_b_tail_no_random

outputMat = loadmat('../Matlab_Data/output.mat')
outputArr = outputMat['output']


dataMat = loadmat('../Matlab_Data/genLutBRandomData.mat')
dataArr = dataMat["randomData"]

nArr = dataMat['n']


count = 0
step = 5
diff = np.zeros((1,1000))
arr = []

while count <1000:
    x = count * step
    #if count == 4:
        #print((int(nArr[0,count]),dataArr[0,x],dataArr[0,x+1],dataArr[0,x+2],dataArr[0,x+3],dataArr[0,x+4]))

    a = gen_lut_b_tail_no_random.gen_lut_b_tail(int(nArr[0,count]), dataArr[0,x], dataArr[0, x + 1], dataArr[0, x + 2], dataArr[0, x + 3], dataArr[0, x + 4])


    total = 0
    count2 = 0
    for y in range(0,19):
        for z in range(0,19):
            total = total + (outputArr[0,count*19*19+count2] - a[y,z])
            count2 = count2+1
    total = total/(19*19)
    arr.append(total)
    #print('count '+str(count))

    # if count == 26:
    #     print((int(nArr[0,count]),dataArr[0,x],dataArr[0,x+1],dataArr[0,x+2],dataArr[0,x+3],dataArr[0,x+4]))
    count = count +1
    #print('count'+str(count))

    #print('total'+str(total))

#endTime = time.time()

#print((endTime-startTime)/1000)
import matplotlib.pyplot as plt
arr = np.array(arr)

plt.hist(arr, density=True, bins=20)  # density=False would make counts
plt.ylabel('Amount')
plt.xlabel('Error')
plt.axvline(arr.mean(), color='k', linestyle='dashed', linewidth=1)
plt.title("gen_lut_b_tail Error Distribution")
plt.show()

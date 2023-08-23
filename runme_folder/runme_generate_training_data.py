import time

from scipy.io import savemat
import numpy as np
import os
import cv2 as cv
from Translated_Python_Programs_Without_Randoms import return_graymodels_fish_NORANDOM
from skimage.util import random_noise
from scipy.io import loadmat
import multiprocessing

def getDiskSE(r):
    matlabDisksLoaded = loadmat('../Matlab_Data/Matlab_Disks.mat')
    matlabDisks = matlabDisksLoaded['disks']

    return matlabDisks[0:(2*r -1),0:(2*r -1),r-5]


def roundHalfUp(a):
    return (np.floor(a)+ np.round(a - (np.floor(a)-1))-1)

def uint8(a):
    a = roundHalfUp(a)

    if np.ndim(a) == 0:
        if a <0:
            a = 0
        if a > 255:
            a = 255
    else:
        a[a>255]=255
        a[a<0]=0
    return a

#Got to check if this is also right
def imGaussNoise(image,mean,var):
    # row,col,ch= image.shape
    # sigma = var**0.5
    # gauss = np.random.normal(mean,sigma,(row,col,ch))
    # gauss = gauss.reshape(row,col,ch)
    # noisy = image + gauss

    row,col= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


pi = np.pi
#addpath('../')
proj_params = 'Matlab_Data/proj_params_101019_corrected_new'
lut_b_tail = 'lut_b_tail.mat'
lut_s_tail = 'lut_s_tail.mat'
dataset_name = '100_percent'
fish_shapes = 'generated_pose_100_percent.mat'
patchy_noise = 1

# % Load data
# load(proj_params)
lut_s_tail_mat = loadmat(lut_b_tail)
lut_b_tail = lut_s_tail_mat['lut_b_tail']

lut_s_tail_mat = loadmat(lut_s_tail)
lut_s_tail = lut_s_tail_mat['lut_s_tail']

# load(lut_s_tail);
data_dir = ['training_data_3D_', dataset_name]

x_all_data = loadmat(fish_shapes)
x_all_data = x_all_data['generated_pose']
x_all_data = np.array(x_all_data)

idx = 1
imageSizeX = 141
imageSizeY = 141
path = os.path.join("../", data_dir[0] + dataset_name)
path = os.path.join(path,'annotations_' + dataset_name + '_pose')
if not os.path.exists(path):
  os.makedirs(path)



path = os.path.join("../", data_dir[0] + dataset_name)
path = os.path.join(path,'annotations_' + dataset_name + '_crop_coor')
if not os.path.exists(path):
  os.makedirs(path)



path = os.path.join("../", data_dir[0] + dataset_name)
path = os.path.join(path,'annotations_' + dataset_name + '_eye_coor')
if not os.path.exists(path):
  os.makedirs(path)



path = os.path.join("../", data_dir[0] + dataset_name)
path = os.path.join(path,'images')
if not os.path.exists(path):
  os.makedirs(path)

path = os.path.join("../", data_dir[0] + dataset_name)
path = os.path.join(path,'annotations_' + dataset_name + '_coor_3d')
if not os.path.exists(path):
  os.makedirs(path)


#### need to find replacement ? #####
# myStream = RandStream('mlfg6331_64')
# RandStream.setGlobalStream(myStream)
######################################

from scipy.io import loadmat

dataMat = loadmat('../Matlab_Data/runme_generate_training_data_pvar.mat')
pArr = dataMat["p"]
pArr = pArr[0]

dataMat = loadmat('../Matlab_Data/runme_generate_training_data_RandomData.mat')
r = dataMat["r"]
r = r[0]

fl = dataMat["fl"]
fl = fl[0]

n1 = dataMat["n1"]
n1 = n1[0]

n2 = dataMat["n2"]
n2 = n2[0]

c = dataMat["c"]
c = c[0]

I = dataMat["I"]
I = I[0]

rL = dataMat["rL"]
rL = rL[0]


#Outputs
outputMat = loadmat('../Matlab_Data/runme_generate_training_data_MatlabOutput_WithPatchNoise.mat')

imOutput = outputMat['outputImg']
Crop_coorOutput = outputMat['outputCrop_coor']
PoseOutput = outputMat['outputPose']
Eye_coorOutput = outputMat['outputEye_coor']
Coor_3dOutput = outputMat['outputCoor_3d']





# np.random.seed(1337)







zeros_mat = np.zeros((141, 141))
startTime = time.time()

rIdx = 0
flIdx = 0
n1Idx = 0
n2Idx = 0

#variables for patch_noise
pIdx = 0
cIdx = 0
rLIdx = 0
iIdx = 0


allImgDifferences = []
averageDiff = []

startTime = time.time()
for q in range(0,500):

    x = np.copy(x_all_data[idx-1,:])
    x[12] = x_all_data[idx-1, 17-1]
    x[0] = (r[rIdx] - 0.5) * 40
    x[1] = (r[rIdx+1] - 0.5) * 40
    x[2] = (r[rIdx+2] - 0.5) * 35 + 72.5
    x[4: 12] = x_all_data[idx-1, 0: 8]
    x = x[0:13]
    temp = 99
    temp2 = np.concatenate((x,x_all_data[idx-1, 8: 16],[temp]),axis=0)
    #x[13: 21] = x_all_data[idx-1, 8: 16]
    temp2[21] = x_all_data[idx-1, 18-1]
    x = temp2

    fishlen = fl[flIdx]
    flIdx +=1

    # fishlen = np.random.normal(3.8, 0.15)

    seglen = fishlen * 0.09


    x[3] = r[rIdx+3] * 2 * pi

    rIn = [r[rIdx], r[rIdx+1], r[rIdx+2], r[rIdx+3], r[rIdx+4], r[rIdx+5]]
    rIn = np.array(rIn)
    [gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2, c_b, c_s1, c_s2, eye_b, eye_s1, eye_s2, coor_3d] = return_graymodels_fish_NORANDOM.return_graymodels_fish(x, lut_b_tail, lut_s_tail, proj_params, fishlen, imageSizeX, imageSizeY,rIn)
    cent_b = np.array([0, 0])
    cent_s1 = np.array([0,0])
    cent_s2 = np.array([0,0])

    #eye_coor = np.array([eye_b, eye_s1, eye_s2])
    eye_coor = np.concatenate((eye_b,eye_s1,eye_s2),axis=1)


    #check if this is the right round function
    #filter_size = 2 * roundHalfUp(np.random.rand(3)) + 3
    filter_size = 2 * roundHalfUp(np.array([r[rIdx],r[rIdx+1],r[rIdx+2]])) + 3

    # sigma = np.random.rand(3) + 0.5
    sigma = np.array([r[rIdx+4],r[rIdx+5],r[rIdx+6]]) + 0.5


###########NEED TO FIND SUITABLE REPLACEMENT ###################################
    #gray_b = imgaussfilt(gray_b, sigma[0], 'FilterSize', filter_size[0])
    kernel = cv.getGaussianKernel(int(filter_size[0]), sigma[0])
    gray_b = cv.filter2D(gray_b,-1,kernel)

    gray_b = uint8(gray_b)

    #gray_s1 = imgaussfilt(gray_s1, sigma[1], 'FilterSize', filter_size[1])
    kernel = cv.getGaussianKernel(int(filter_size[1]), sigma[1])
    gray_s1 = cv.filter2D(gray_s1,-1,kernel)

    gray_s1 = uint8(gray_s1)

    #gray_s2 = imgaussfilt(gray_s2, sigma[2], 'FilterSize', filter_size[2])
    kernel = cv.getGaussianKernel(int(filter_size[2]), sigma[2])
    gray_s2 = cv.filter2D(gray_s2,-1,kernel)

    gray_s2 = uint8(gray_s2)
# Got to check I did it right #


# Got to check the code below to see if it is right

    #Converting to range [0,1]
    maxB = max(gray_b.flatten())
    gray_b = gray_b / maxB

    #gray_b = gray_b / max(gray_b.flatten())

    gray_b = imGaussNoise(gray_b, (r[rIdx+7] * n1[n1Idx]) / 255, (r[rIdx+8] * 50 + 20) / 255 ** 2)
    n1Idx += 1
    #Converting Back
    gray_b = gray_b * (maxB/ max(gray_b.flatten()))

    #gray_b = gray_b * (255/max(gray_b.flatten()))

    gray_b = uint8(gray_b)



    #Converting to range [0,1]

    maxS1 = max(gray_s1.flatten())
    gray_s1 = gray_s1/maxS1

    #gray_s1 = gray_s1 / max(gray_s1.flatten())

    gray_s1 = imGaussNoise(gray_s1, (r[rIdx+9] * n2[n2Idx]) / 255, (r[rIdx+10] * 50 + 10) / 255 ** 2)
    n2Idx += 1
    #Converting Back
    gray_s1 = gray_s1 * (maxS1/max(gray_s1.flatten()))

    #gray_s1 = gray_s1 * (255/ max(gray_s1.flatten()))
    gray_s1 = uint8(gray_s1)



    #Converting to range [0,1]
    maxS2 = max(gray_s2.flatten())
    gray_s2 = gray_s2 / maxS2

    #gray_s2 = gray_s2 / max(gray_s2.flatten())
    gray_s2 = imGaussNoise(gray_s2, (r[rIdx+11] * n2[n2Idx]) / 255, (r[rIdx+12] * 50 + 10) / 255 ** 2)
    n2Idx += 1
    #Converting Back
    gray_s2 = gray_s2 * (maxS2 / max(gray_s2.flatten()))

    #gray_s2 = gray_s2* (255/ max(gray_s2.flatten()))
    gray_s2 = uint8(gray_s2)

    rIdx += 13



########################################################################################


    if (patchy_noise):
        # pvar = np.random.poisson(0.2)
        pvar = pArr[pIdx]
        if (pvar > 0):
            for i in range(1,int(np.floor(pvar+1))):
                # centers = np.random.randint(1, high=142, size=(2))
                centers = np.array([c[cIdx], c[cIdx + 1]])
                cIdx += 2

                var_mat = np.copy(zeros_mat)

                ####need to check if -1 correct###
                var_mat[centers[0]-1, centers[1]-1] = 1

                ############off by a little bit ##########################
                #se = strel('disk', ( (2 * np.random.randint(5, high=21))+1, (2 * np.random.randint(5, high=21))+1 )   )
                #var_mat = imdilate(var_mat, se)
                #se = cv.getStructuringElement(cv.MORPH_ELLIPSE, ( (2 * I[iIdx])+1, (2 * I[iIdx])+1 ))
                se = getDiskSE(I[iIdx])

                var_mat = cv.dilate(var_mat,se)
                iIdx += 1


                #copyVar_mat = np.copy(var_mat)
                #copyVar_mat[copyVar_mat>0] = 1

                #gray_b = imnoise(gray_b, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 20) / 255 ** 2)
                #copyGray_b = np.copy(gray_b)

                maxB = max(gray_b.flatten())
                gray_b = gray_b / max(gray_b.flatten())

                gray_b = random_noise(gray_b, mode='localvar', local_vars=(var_mat * 3 * (rL[rLIdx] * 60 + 20) / 255 ** 2)+.00000000000000001)

                gray_b = gray_b * (maxB/ max((gray_b.flatten())))
                #gray_b = gray_b * (255/max(gray_b.flatten()))

                #gray_b[copyVar_mat>0] = copyGray_b[copyVar_mat>0]

                rLIdx += 1
                ###############################################################
        # pvar = np.random.poisson(0.2)

        pIdx += 1
        pvar = pArr[pIdx]
        if (pvar > 0):
            for i in range(1,int(np.floor(pvar+1))):
                # centers = np.random.randint(1, high=142, size=(2))
                centers = np.array([c[cIdx], c[cIdx + 1]])
                cIdx += 2

                var_mat = np.copy(zeros_mat)
                ############need to find equivalent ##########################
                # se = cv.getStructuringElement(cv.MORPH_ELLIPSE, ( (2 * I[iIdx])+1, (2 * I[iIdx])+1 ))
                se = getDiskSE(I[iIdx])
                iIdx += 1


                var_mat = np.copy(zeros_mat)
                var_mat[centers[0]-1, centers[1]-1] = 1
                var_mat = cv.dilate(var_mat,se)
                #gray_s1 = imnoise(gray_s1, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 10) / 255 ^ 2)

                maxS1 = max(gray_s1.flatten())
                gray_s1 = gray_s1 / max(gray_s1.flatten())

                gray_s1 = random_noise(gray_s1, mode='localvar', local_vars=(var_mat * 3 * (rL[rLIdx] * 60 + 10) / 255 ** 2)+.00000000000000001)

                gray_s1 = gray_s1 * (maxS1/max(gray_s1.flatten()))
                #gray_s1 = gray_s1 * (255/max(gray_s1.flatten()))

                rLIdx += 1
                ###############################################################

        # pvar = np.random.poisson(0.2)
        pIdx += 1
        pvar = pArr[pIdx]
        if (pvar > 0):
            for i in range(1,int(np.floor(pvar+1))):
                # centers = np.random.randint(1, high=142, size=(2))
                centers = np.array([c[cIdx], c[cIdx + 1]])
                cIdx += 2

                var_mat = np.copy(zeros_mat)
                #se = strel('disk', np.random.randint(5, high=21))
                #se = cv.getStructuringElement(cv.MORPH_ELLIPSE, ((2 * I[iIdx]) + 1, (2 * I[iIdx]) + 1))
                se = getDiskSE(I[iIdx])
                iIdx += 1

                var_mat = np.copy(zeros_mat)
                var_mat[centers[0] - 1, centers[1] - 1] = 1
                #var_mat = imdilate(var_mat, se)
                var_mat = cv.dilate(var_mat, se)

                #gray_s1 = imnoise(gray_s1, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 10) / 255 ^ 2)

                maxS2 = max(gray_s2.flatten())
                gray_s2 = gray_s2 / max(gray_s2.flatten())
                gray_s2 = random_noise(gray_s2, mode='localvar',local_vars=(var_mat * 3 * (rL[rLIdx] * 60 + 10) / 255 ** 2)+.00000000000000001)

                gray_s2 = gray_s2 * (maxS2/max(gray_s2.flatten()))
                #gray_s2 = gray_s2 * (255/max(gray_s2.flatten()))

                rLIdx +=1
        pIdx += 1


    gray_b = gray_b * (255 / float(max(gray_b.flatten())))
    gray_s1 = gray_s1 * (255 / float(max(gray_s1.flatten())))
    gray_s2 = gray_s2 * (255 / float(max(gray_s2.flatten())))

    im = np.concatenate((gray_b,gray_s1,gray_s2),axis=0)

    directory = '/Users/jacob/PycharmProjects/eye1model/training_data_3D_100_percent/images'
    os.chdir(directory)

    filename = 'im_'+ str(idx) + '.png'

    cv.imwrite(filename,im)



    crop_coor = np.concatenate((crop_b,crop_s1,crop_s2),axis=0)
    pose = np.concatenate((c_b,c_s1,c_s2),axis=1)

    if (np.any(eye_coor)==False or np.any(coor_3d)==False or np.any(pose)==False):
        print(x)
        #print(coor_mf_matname)

    # saves all variables in this file under this path#
    # path = '/Users/jacob/PycharmProjects/eye1model/training_data_3D_100_percent/annotations_100_percent_coor_3d'
    # os.chdir(path)
    # savemat('coor_3d_'+str(idx)+'.mat', {'coor_3d': coor_3d})
    #
    # path = '/Users/jacob/PycharmProjects/eye1model/training_data_3D_100_percent/annotations_100_percent_pose'
    # os.chdir(path)
    # savemat('pose_ann'+str(idx)+'.mat', {'pose': pose})
    #
    # path = '/Users/jacob/PycharmProjects/eye1model/training_data_3D_100_percent/annotations_100_percent_crop_coor'
    # os.chdir(path)
    # savemat('crop_coor_ann_'+str(idx)+'.mat', {'crop_coor': crop_coor})
    #
    # path = '/Users/jacob/PycharmProjects/eye1model/training_data_3D_100_percent/annotations_100_percent_eye_coor'
    # os.chdir(path)
    # savemat('eye_coor_ann_'+str(idx)+'.mat', {'eye_coor': eye_coor})
    #



    idx = idx + 1
    if (idx% 1) == 0:
        #print(idx)
        3+3
    if (idx == 1):
        print('Finished 500000')


    diffIm = im - imOutput[:,:,q]

    directory = '/Users/jacob/PycharmProjects/eye1model/training_data_3D_100_percent/images'
    os.chdir(directory)

    filename = 'diffIm_'+ str(idx-1) + '.png'

    cv.imwrite(filename,diffIm)


    path = '/'
    os.chdir(path)

    print(idx)
#
#
#
#
#
#     allImgDifferences = np.concatenate((im.flatten(), allImgDifferences), axis=0)
#
#     diffPose = pose - PoseOutput[:,:,q]
#     diffPoseSum = np.sum(diffPose)
#     diffCrop_coor = crop_coor - Crop_coorOutput[:,:,q]
#     diffCrop_coorSum = np.sum(diffCrop_coor)
#     diffEye_coor = eye_coor - Eye_coorOutput[:,:,q]
#     diffEye_coorSum = np.sum(diffEye_coor)
#     diffCoor_3d = coor_3d - Coor_3dOutput[:,:,q]
#     diffCoor_3dSum = np.sum(diffCoor_3d)
#
    imSum = np.sum((diffIm**2)**.5)
    imSum = imSum / 59643
    averageDiff.append(imSum)


    print('Index: '+str(q))
    print('Average Difference between the Images: '+str(imSum)+' Max Difference: '+str(max(diffIm.flatten())))
#     print("Difference in Pose: "+str(diffPoseSum))
#     print("Difference in Crop_coor: "+str(diffCrop_coorSum))
#     print("Difference in Eye_coor: "+str(diffEye_coorSum))
#     print("Difference in Coor_3d: "+str(diffCoor_3dSum))
#
# allImgDifferences = np.array(allImgDifferences)
averageDiff = np.array(imSum)
savemat('Matlab_Data/runme_generate_training_data_AverageDiff.mat', {'averageDiff': averageDiff})


# savemat('Matlab_Data/runme_generate_training_data_RandomBackGround_2.mat', {'allImgDifferences': allImgDifferences})

endTime = time.time()
print('Average Time: '+str((endTime-startTime)/50))









#Run in parallel
#
# if __name__ == '__main__':
#
#     startTime = time.time()
#
#     pool_obj = multiprocessing.Pool()
#     pool_obj.map(genImage, range(0, 20))
#     pool_obj.close()
#
#     endTime = time.time()
#
#     print('Finished running')
#     print("Time it took to finish Running")
#     print(endTime-startTime)
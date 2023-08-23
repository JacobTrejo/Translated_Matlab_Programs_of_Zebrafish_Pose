# % % Script written by Aniket
# Ravan(asravan2 @ illinois.edu)
#
#  This script evaluates correlation coefficients on
#  the real preprocessed images
#
#  User inputs
#
#  lut_b_tail - string input; path to larval tail lookup table bottom camera
#
#  lut_s_tail - string input; path to larval tail lookup table side camera
#
#  proj_params - string input; path to camera projection parameters
#
#  data_folder - string input; path to folder containing preprocessed real images for evaluation
#
#  experiment - string input; training experiment
#
#  model_prediction_path - string input;
#
#  User outputs
#
#  A mat file containing the correlation coefficients for projections of the bottom camera and the two side cameras
#
#  User inputs

import numpy as np
import math as m
import cv2 as cv
import imageio.v3 as iio
from scipy.io import loadmat
from scipy.io import savemat
from scipy.interpolate import splprep, splev
from return_graymodels_fish import return_graymodels_fish

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

def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, m.cos(theta), -m.sin(theta)],
                      [0, m.sin(theta), m.cos(theta)]])


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])


def Rz(theta):
    return np.matrix([[m.cos(theta), -m.sin(theta), 0],
                      [m.sin(theta), m.cos(theta), 0],
                      [0, 0, 1]])

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

def mask_real_image(im_real,im_gray):
    #I will make this better soon by storing structuring elements in a file
    diskSe = [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
    thresh, throwaway = cv.threshold(im_gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    bw = np.copy(im_gray)
    bw[bw >= thresh] = 255
    im_gray[bw < thresh] = 0

    mask = cv.dilate(bw, diskSe)

    im_masked = uint8(mask) * im_real
    return im_masked




lut_b_tail = 'lut_b_tail.mat'

lut_s_tail = 'lut_s_tail.mat'

proj_params = 'proj_params_101019_corrected_new.mat'

experiment = '100_percent_er_complete_data'

data_folder = '../real_data/complete_dataset/complete_data_er/'

model_prediction_path = '../generate_manuscript_figures/data_files_for_figures/network_predictions'

# Load data
# I need to find where they are used
#loadmat(lut_b_tail)
#loadmat(lut_s_tail)
#loadmat(proj_params)

#Is this conversion fine ?
#asynchronized_videos = {'010920_1221', '101419_1145', '101619_1813', }
asynchronized_videos = ['010920_1221', '101419_1145', '101619_1813']

# Generate string defining filenames path

split_experiment_name = experiment.split('_')

filename_string = ''

#for i = 3:length(split_experiment_name)
for i in range(2,len(split_experiment_name)):
    filename_string += split_experiment_name[i]
    filename_string += '_'

filename_string = filename_string[:-1]


# reconstructed_files_mat = ['reconstructed_pose_', experiment]
reconstructed_files_mat = 'reconstructed_pose_' + experiment


# filenames_path = ['all_filenames_', filename_string];
filenames_path = 'all_filenames_' + filename_string

#I probably need to get the array
reconstructed_poses = loadmat(model_prediction_path + '/' + reconstructed_files_mat + '.mat')
filenames_mat = loadmat(model_prediction_path + '/' + filenames_path + '.mat')

#Start evaluation

asynchronized_video_count = 0

s = np.linspace(0,1,10)

pb = np.zeros(len(filenames_mat))
ps1 = np.zeros(len(filenames_mat))
ps2 = np.zeros(len(filenames_mat))

for file in range(0,len(filenames_mat)):

    if file % 500 == 0:
        print('File '+ str(file) + ' of ' + str(len(filenames_mat)))

    coor_3d = (np.squeeze(reconstructed_poses[file,:,:])).astype(float)

    #crop_coor = importdata([data_folder, 'annotations_crop_coor/crop_coor_ann_', filenames_mat(file, 1:end - 4), '.mat']);
    crop_coor = loadmat(data_folder + 'annotations_crop_coor/crop_coor_ann_' + filenames_mat[file, 1:-4] + '.mat')

    c = coor_3d

    #Why the sum ?
    if (np.sum( asynchronized_videos == filenames_mat[file, 1:-11] )):
        asynchronized_video_count +=1
        continue



    # im = importdata([data_folder, 'images_real/im_', filenames_mat[file, 1: -4], '.png']);
    im = iio.imread(data_folder +'images_real/im_' + filenames_mat[file, 1: -4] + '.png')

    coor_3d_shifted = coor_3d

    #Might use np.interp

    #coor_3d_shifted[:, 0: 10] = interparc(s, coor_3d[0, 0: 10], coor_3d[1, 0: 10], coor_3d[2, 0: 10], 'spline')
    # might need to pass in s as the parameter u
    tck, u = splprep([coor_3d[0, 0: 10], coor_3d[1, 0: 10], coor_3d[2, 0: 10]], s=0)
    coor_3d_shifted[:, 0: 10] = splev(u, tck)


    coor_3d[:, 0: 10] = coor_3d_shifted[:, 0: 10]

    # Convert pose to params

    coor_3d_shifted = coor_3d_shifted - coor_3d_shifted[:, 0]

    diff_vec = coor_3d_shifted[:, 1: 10] - coor_3d_shifted[:, 0: 9]

    heading_vec = diff_vec[:, 0]

    heading_vec_2d = heading_vec[0:2]

    #theta_0 = atan2(heading_vec_2d(2), heading_vec_2d(1))
    theta_0 = np.arctan2(heading_vec_2d[1], heading_vec_2d[0])


    if (theta_0 < 0):
        theta_0 = 2 * np.pi - np.abs(theta_0)


    phi_0 = -np.arctan2(heading_vec[2], np.linalg.norm(heading_vec_2d))

    coor_3d_fish_frame = Ry(-phi_0) * Rz(-theta_0) * coor_3d_shifted

    eye_vec = coor_3d_fish_frame[:, 10] - coor_3d_fish_frame[:, 11]

    gamma_01 = np.arctan2(eye_vec[2], eye_vec[1])

    ref_vec_for_roll = Rz(gamma_01 + np.pi) * np.array([[1], [0], [0]])

    gamma_02 = np.arctan2(ref_vec_for_roll[1], ref_vec_for_roll[0])

    coor_3d_fish_frame_1 = Rx(-gamma_01) * coor_3d_fish_frame

    coor_3d_fish_frame_2 = Rx(-gamma_02) * coor_3d_fish_frame

    if (np.abs(gamma_01) < np.abs(gamma_02)):
        gamma_0 = gamma_01

        coor_3d_fish_frame = coor_3d_fish_frame_1

    else:

        gamma_0 = gamma_02

        coor_3d_fish_frame = coor_3d_fish_frame_2

    #might wanna re check the indices
    diff_vec_fish_frame = coor_3d_fish_frame[:, 1: 10] - coor_3d_fish_frame[:, 0: 9]


    theta = np.zeros(8)
    phi = np.zeros(8)

    for i in range(0,8):

        theta[i] = np.arctan2(diff_vec_fish_frame[1, i + 1], diff_vec_fish_frame[0, i + 1])

        phi[i] = -np.arctan2(diff_vec_fish_frame[2, i + 1], np.linalg.norm(diff_vec_fish_frame[0:2, i + 1]))


    x = np.zeros(22)
    x[0: 3] = coor_3d[:, 3]

    x[3] = theta_0

    x[12] = phi_0

    x[21] = gamma_0

    x[4: 12] = np.diff(np.concatenate(([0], theta)))

    x[13: 21] = np.diff(np.concatenate(np.concatenate(([0], phi))))

    seglen = np.linalg.norm(diff_vec[:, 0])

    #Rendering the images

    fishlen = seglen / 0.09

    imageSizeX = 141
    imageSizeY = 141

    Crop_b = crop_coor[0:4] + 1
    Crop_s1 = crop_coor[4:8] + 1
    Crop_s2 = crop_coor[8:12] + 1

    [gray_b, gray_s1, gray_s2, c_b, c_s1, c_s2, eye_b, eye_s1, eye_s2, coor_3d] = \
        return_graymodels_fish(x, Crop_b, Crop_s1, Crop_s2,
        proj_params, fishlen, imageSizeX, imageSizeY)

    #CHECK THE INDEX
    im_b = im[0:141,:]

    im_s1 = im[141:282,:]

    im_s2 = im[282:,:]

    im_b_masked = mask_real_image(im_b, gray_b)

    im_s1_masked = mask_real_image(im_s1, gray_s1)

    im_s2_masked = mask_real_image(im_s2, gray_s2)

    pb[file] = corr2_coeff(im_b_masked(im_b_masked > 0), gray_b(im_b_masked > 0))

    ps1[file] = corr2_coeff(im_s1_masked(im_s1_masked > 0), gray_s1(im_s1_masked > 0))

    ps2[file] = corr2_coeff(im_s2_masked(im_s2_masked > 0), gray_s2(im_s2_masked > 0))

    rgb_b = np.zeros( np.concatenate((list(im_b.shape), [3])) )

    rgb_b[:,:, 0] = uint8(im_b_masked - gray_b)

    rgb_b[:,:, 1] = uint8(gray_b - im_b_masked)

    rgb_s1 = np.zeros( np.concatenate((list(im_s1_masked.shape), [3])) )

    rgb_s1[:,:, 0] = uint8(im_s1_masked - gray_s1)

    rgb_s1[:,:, 1] = uint8(gray_s1 - im_s1_masked)

    rgb_s2 = np.zeros( np.concatenate((list(im_s2_masked.shape), [3])) )

    rgb_s2[:,:, 0] = uint8(im_s2_masked - gray_s2)

    rgb_s2[:,:, 1] = uint8(gray_s2 - im_s2_masked)

    rgb_im = np.concatenate((rgb_b, rgb_s1, rgb_s2),axis=1)



#Save outputs

savemat('../generate_manuscript_figures/data_files_for_figures/correlation_coefficients/correlation_coefficients_'+ \
      experiment, {'pb':pb, 'ps1':ps1, 'ps2':ps2})

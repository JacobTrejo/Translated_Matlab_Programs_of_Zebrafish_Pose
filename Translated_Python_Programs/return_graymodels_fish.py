#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import numpy as np
from Translated_Python_Programs import calc_proj_w_refra_cpu_v3

from Translated_Python_Programs import view_s_lut_new_real_cpu, view_b_lut_new_real_cpu, return_head_real_model_new

def roundHalfUp(a):
    return (np.floor(a)+ np.round(a - (np.floor(a)-1))-1)

def return_graymodels_fish(x,lut_b_tail,lut_s_tail,proj_params,fishlen,imageSizeX,imageSizeY):
    # initial guess of the position
    # seglen is the length of each segment
    seglen = fishlen*0.09
    # alpha: azimuthal angle of the rotated plane
    # gamma: direction cosine of the plane of the fish with z-axis
    # theta: angles between segments along the plane with direction cosines
    # alpha, beta and gamma
    hp = np.array([[x[0]],[x[1]],[x[2]]])
    dtheta = x[3:12]
    theta = np.cumsum(dtheta)
    dphi = x[12:21]
    phi = np.cumsum(dphi)

    vec = seglen*np.array([np.cos(theta)*np.cos(phi), np.sin(theta) *np.cos(phi), -np.sin(phi)])

    # vec_ref_1 is parallel to the camera sensor of b and s2
    # vec_ref_2 is parallel to s1
    vec_ref_1 = np.array([[seglen],[0],[0]])
    vec_ref_2 = np.array([[0],[seglen],[0]])


    pt_ref = np.concatenate((hp + vec_ref_1, hp + vec_ref_2), axis=1)


    z = np.array([[0], [0], [0]])
    frank = np.concatenate((z, vec), axis=1)
    pt = np.cumsum(frank,axis=1)

    pt = np.concatenate((pt + np.tile(hp, (1, 10)),pt_ref),axis=1)

    # use cen_3d as the 4th point on fish
    hinge = pt[:,2]
    vec_13 = pt[:,0] - hinge
    temp1 = vec_13[0]
    temp2 = vec_13[1]
    temp3 = vec_13[2]
    vec_13 = np.array([[temp1],[temp2],[temp3]])
    vec_13 = np.tile(vec_13, (1, 12))

    pt = pt + vec_13

    [coor_b,coor_s1,coor_s2] = calc_proj_w_refra_cpu_v3.calc_proj_w_refra_cpu(pt, proj_params)

    # keep the corresponding vec_ref for each
    coor_b_shape = np.shape(coor_b)
    coor_b = coor_b[:,0:coor_b_shape[1]-2]
    idxs = [*range(coor_s1.shape[1])]
    idxs.pop(coor_s1.shape[1]-2)  # this removes elements from the list
    coor_s1 = coor_s1[:, idxs]
    coor_s2 = coor_s2[:,0:coor_s2.shape[1]-1]



    # Re-defining cropped coordinates for training images of dimensions
    # imageSizeY x imageSizeX
    crop_b = np.array([0.0,0.0,0.0,0.0])
    crop_b[0] = roundHalfUp(coor_b[1,2]) - (imageSizeY - 1)/2 + roundHalfUp((np.random.rand() - 0.5)*40)
    crop_b[1] = crop_b[0] + imageSizeY - 1
    crop_b[2] = roundHalfUp(coor_b[0,2]) - (imageSizeX - 1)/2 + roundHalfUp((np.random.rand() - 0.5)*40)
    crop_b[3] = crop_b[2] + imageSizeX - 1



    crop_s1 = np.array([0.0,0.0,0.0,0.0])
    crop_s1[0] = roundHalfUp(coor_s1[1,2]) - (imageSizeY - 1)/2 + roundHalfUp((np.random.rand() - 0.5)*40)
    crop_s1[1] = crop_s1[0] + imageSizeY - 1
    crop_s1[2] = roundHalfUp(coor_s1[0,2]) - (imageSizeX - 1)/2 + roundHalfUp((np.random.rand() - 0.5)*40)
    crop_s1[3] = crop_s1[2] + imageSizeX - 1

    crop_s2 = np.array([0.0,0.0,0.0,0.0])
    crop_s2[0] = roundHalfUp(coor_s2[1,2]) - (imageSizeY - 1)/2 + roundHalfUp((np.random.rand() - 0.5)*40)
    crop_s2[1] = crop_s2[0] + imageSizeY - 1
    crop_s2[2] = roundHalfUp(coor_s2[0,2]) - (imageSizeX - 1)/2 + roundHalfUp((np.random.rand() - 0.5)*40)
    crop_s2[3] = crop_s2[2] + imageSizeX - 1

    (projection_b,projection_s1,projection_s2,eye_b,eye_s1,eye_s2,eye_coor_3d) = return_head_real_model_new.return_head_real_model_new(x, fishlen, proj_params, crop_b, crop_s1, crop_s2)

    temp = np.copy(coor_b)
    gray_b = view_b_lut_new_real_cpu.view_b_lut_new_real_cpu(crop_b, temp, lut_b_tail, projection_b, imageSizeX, imageSizeY)
    temp = np.copy(coor_s1)
    gray_s1 = view_s_lut_new_real_cpu.view_s_lut_new_real_cpu(crop_s1, temp, lut_s_tail, projection_s1, imageSizeX, imageSizeY)
    temp = np.copy(coor_s2)
    gray_s2 = view_s_lut_new_real_cpu.view_s_lut_new_real_cpu(crop_s2, temp, lut_s_tail, projection_s2, imageSizeX, imageSizeY)


    annotated_b = np.zeros((2,coor_b.shape[1]))
    annotated_b[0,:] = coor_b[0,:] - crop_b[2] + 1
    annotated_b[1,:] = coor_b[1,:] - crop_b[0] + 1

    annotated_s1 = np.zeros((2,coor_s1.shape[1]))
    annotated_s1[0,:] = coor_s1[0,:] - crop_s1[2] + 1
    annotated_s1[1,:] = coor_s1[1,:] - crop_s1[0] + 1

    annotated_s2 = np.zeros((2,coor_s2.shape[1]))
    annotated_s2[0,:] = coor_s2[0,:] - crop_s2[2] + 1
    annotated_s2[1,:] = coor_s2[1,:] - crop_s2[0] + 1

    annotated_b = annotated_b[:,0:10]
    annotated_s1 = annotated_s1[:,0:10]
    annotated_s2 = annotated_s2[:,0:10]

    eye_b[0,:] = eye_b[0,:] - crop_b[2] + 1
    eye_b[1,:] = eye_b[1,:] - crop_b[0] + 1
    eye_s1[0,:] = eye_s1[0,:] - crop_s1[2] + 1
    eye_s1[1,:] = eye_s1[1,:] - crop_s1[0] + 1
    eye_s2[0,:] = eye_s2[0,:] - crop_s2[2] + 1
    eye_s2[1,:] = eye_s2[1,:] - crop_s2[0] + 1

    #Subtract 1 for accordance with python's format
    eye_b = eye_b - 1
    eye_s1 = eye_s1 - 1
    eye_s2 = eye_s2 - 1

    annotated_b = annotated_b - 1
    annotated_s1 = annotated_s1 - 1
    annotated_s2 = annotated_s2 - 1

    crop_b = crop_b - 1
    crop_s1 = crop_s1 - 1
    crop_s2 = crop_s2 - 1

    coor_3d = pt[:,0:10]

    coor_3d = np.concatenate((coor_3d,eye_coor_3d),axis=1)

    return gray_b, gray_s1, gray_s2, crop_b, crop_s1, crop_s2,annotated_b, annotated_s1, annotated_s2, eye_b, eye_s1, eye_s2,coor_3d
if 0:
    pi = np.pi
    x = [0, 0, 0, .3* pi, .4 * pi, .5 * pi, .6 * pi, .7 * pi, .8* pi, .9 * pi, pi, 1.1* pi, 1.2*  pi, 1.3 * pi, 1.4 * pi, 1.5* pi, 1.6 * pi,1.7 *  pi, 1.8 *  pi, 1.9 * pi , 2*  pi, .1*  pi]
    x = np.array(x)

    r = [.1,.2,.3,.4,.5,.6]
    r = np.array(r)
    startTime = time.time()
    a = return_graymodels_fish(x,1,1,"../Matlab_Data/proj_params_101019_corrected_new",1,100,100,r)

    endTime = time.time()

    print("TIME IT TOOK")
    print(endTime-startTime)
    print('a')
    print(a)

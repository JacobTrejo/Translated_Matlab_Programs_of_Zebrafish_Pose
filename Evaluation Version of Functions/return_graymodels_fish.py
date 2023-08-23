import numpy as np
import math as m
from calc_proj_w_refra_cpu_v3 import calc_proj_w_refra_cpu
from return_head_real_model_new import return_head_real_model_new
from view_s_lut_new_real_cpu import view_s_lut_new_real_cpu
from view_b_lut_new_real_cpu import view_b_lut_new_real_cpu

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

def roundHalfUp(a):
    return (np.floor(a)+ np.round(a - (np.floor(a)-1))-1)

def return_graymodels_fish(x,crop_b,crop_s1,crop_s2,proj_params,fishlen,imageSizeX,imageSizeY):
    seglen = fishlen*0.09
    hp = np.array([[x[0]], [x[1]], [x[2]]])
    dtheta = x[4:12]

    theta = np.cumsum(np.concatenate(([0],dtheta)))
    dphi = x[13:21]
    phi = np.cumsum(np.concatenate(([0],dphi)))

    vec = seglen*np.array([np.cos(theta)*np.cos(phi), np.sin(theta) *np.cos(phi), -np.sin(phi)])

    theta_0 = x[3]
    phi_0 = x[12]
    gamma_0 = x[21]

    vec_ref_1 = np.array([[seglen], [0], [0]])
    vec_ref_2 = np.array([[0], [seglen], [0]])

    pt_ref = np.concatenate((hp + vec_ref_1, hp + vec_ref_2), axis=1)

    z = np.array([[0], [0], [0]])
    frank = np.concatenate((z, vec), axis=1)
    pt = np.cumsum(frank,axis=1)

    pt = Rz(theta_0) @ Ry(phi_0) @ Rx(gamma_0) @ pt

    pt = np.array(pt)
    pt = np.concatenate((pt + np.tile(hp, (1, 10)),pt_ref),axis=1)

    hinge = pt[:, 2]
    vec_13 = pt[:, 0] - hinge

    vec_13 = np.array([[vec_13[0]],[vec_13[1]],[vec_13[2]]])
    vec_13 = np.tile(vec_13, (1, 12))

    pt = pt + vec_13
    pt = np.array(pt)
    [coor_b,coor_s1,coor_s2] = calc_proj_w_refra_cpu(pt, proj_params)

    #There is a better way to do this I will comeback to it later
    #coor_b(:, end - 1: end) = []
    coor_b_shape = np.shape(coor_b)
    coor_b = coor_b[:, 0:coor_b_shape[1] - 2]

    #coor_s1(:,end-1) = [];
    idxs = [*range(coor_s1.shape[1])]
    idxs.pop(coor_s1.shape[1] - 2)  # this removes elements from the list
    coor_s1 = coor_s1[:, idxs]

    #coor_s2(:,end) = [];
    coor_s2 = coor_s2[:,0:coor_s2.shape[1]-1]

    #SHOULD DOUBLE CHECK THIS FUNCTION IS NOT DIFFRENT#########
    [projection_b, projection_s1, projection_s2, eye_b, eye_s1, eye_s2, eye_coor_3d] = \
    return_head_real_model_new(x, fishlen, proj_params, crop_b, crop_s1, crop_s2)

    ######Should Pass in copys of arrays#############
    temp = np.copy(coor_b)
    gray_b = view_b_lut_new_real_cpu(crop_b, temp, projection_b, imageSizeX, imageSizeY)
    temp = np.copy(coor_s1)
    gray_s1 = view_s_lut_new_real_cpu(crop_s1, temp, projection_s1, imageSizeX, imageSizeY)
    temp = np.copy(coor_s2)
    gray_s2 = view_s_lut_new_real_cpu(crop_s2, temp, projection_s2, imageSizeX, imageSizeY)

    annotated_b = np.zeros((2, coor_b.shape[1]))
    annotated_b[0, :] = coor_b[0, :] - crop_b[2] + 1
    annotated_b[1, :] = coor_b[1, :] - crop_b[0] + 1

    annotated_s1 = np.zeros((2, coor_s1.shape[1]))
    annotated_s1[0, :] = coor_s1[0, :] - crop_s1[2] + 1
    annotated_s1[1, :] = coor_s1[1, :] - crop_s1[0] + 1

    annotated_s2 = np.zeros((2, coor_s2.shape[1]))
    annotated_s2[0, :] = coor_s2[0, :] - crop_s2[2] + 1
    annotated_s2[1, :] = coor_s2[1, :] - crop_s2[0] + 1

    annotated_b = annotated_b[:, 0:10]
    annotated_s1 = annotated_s1[:, 0:10]
    annotated_s2 = annotated_s2[:, 0:10]

    eye_b[0, :] = eye_b[0, :] - crop_b[2] + 1
    eye_b[1, :] = eye_b[1, :] - crop_b[0] + 1
    eye_s1[0, :] = eye_s1[0, :] - crop_s1[2] + 1
    eye_s1[1, :] = eye_s1[1, :] - crop_s1[0] + 1
    eye_s2[0, :] = eye_s2[0, :] - crop_s2[2] + 1
    eye_s2[1, :] = eye_s2[1, :] - crop_s2[0] + 1

    eye_b = eye_b - 1
    eye_s1 = eye_s1 - 1
    eye_s2 = eye_s2 - 1

    annotated_b = annotated_b - 1
    annotated_s1 = annotated_s1 - 1
    annotated_s2 = annotated_s2 - 1

    crop_b = crop_b - 1
    crop_s1 = crop_s1 - 1
    crop_s2 = crop_s2 - 1

    coor_3d = pt[:, 0:10]

    coor_3d = np.concatenate((coor_3d, eye_coor_3d), axis=1)

    return gray_b, gray_s1, gray_s2,annotated_b, annotated_s1, annotated_s2, eye_b, eye_s1, eye_s2,coor_3d

if 1:
    pi = np.pi
    x = [0, 0, 0, .3* pi, .4 * pi, .5 * pi, .6 * pi, .7 * pi, .8* pi, .9 * pi, pi, 1.1* pi, 1.2*  pi, 1.3 * pi, 1.4 * pi, 1.5* pi, 1.6 * pi,1.7 *  pi, 1.8 *  pi, 1.9 * pi , 2*  pi, .1*  pi]
    x = np.array(x)

    r = [.1,.2,.3,.4,.5,.6]
    r = np.array(r)
    cb = np.array([259, 360, 314, 414])
    cs1 = np.array([1186, 1232, 306, 336])
    cs2 = np.array([1211, 1259, 314, 345])

    (gray_b, gray_s1, gray_s2,annotated_b, annotated_s1, annotated_s2, eye_b, eye_s1, eye_s2, coor_3d) = return_graymodels_fish(x,cb,cs1,cs2,"../Matlab_Data/proj_params_101019_corrected_new",1,100,100)

    print(gray_b)





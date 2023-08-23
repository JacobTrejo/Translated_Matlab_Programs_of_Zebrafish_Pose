#Written by Aniket Ravan
# 5th of May 2019
# Last edit on 4th of September 2019
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import math as m

from eye1model import eye1model
from eye2model import eye2model
from bellymodel import bellymodel
from headmodel import headmodel
from project_camera_copy import project_camera_copy
from reorient_model import reorient_model
from reorient_belly_model import reorient_belly_model
from  calc_proj_w_refra_cpu_v3 import calc_proj_w_refra_cpu


def mat2gray(imgArg):
    img = np.copy(imgArg)
    minimum = np.min(img)
    maximum = np.max(img)
    difference = maximum - minimum

    img = img - minimum
    img = img / difference

    return img

def sigmoid(xArg,scaling):
    x = np.copy(xArg)
    y = 1 / (1 + np.exp(-scaling * x))
    return y



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
def return_head_real_model_new(x,fishlen,proj_params,cb,cs1,cs2):


    # Calculate the 3D points pt from model parameters
    seglen = fishlen * 0.09
    size_lut_3d = 2 # Represents the length of the box in which the 3D fish is constructed
    inclination = x[12]
    heading = x[3]
    hp = np.array([[x[0]],[x[1]],[x[2]]])
    dtheta = x[3:12]
    theta = np.cumsum(dtheta)
    dphi = x[12:21]
    phi = np.cumsum(dphi)
    roll = x[21]

    vec_unit = seglen* np.array([[np.cos(theta) * np.cos(phi)], [np.sin(theta) * np.cos(phi)], [-np.sin(phi)]])
    vec_unit = vec_unit[:,0,:]
    # vec_ref_1 is parallel to the camera sensor of b and s2
    # vec_ref_2 is parallel to s1
    vec_ref_1 = np.array([[seglen], [0], [0]])
    vec_ref_2 = np.array([[0], [seglen], [0]])
    pt_ref = np.array([hp + vec_ref_1, hp + vec_ref_2])
    pt_ref = np.transpose(pt_ref[:, :, 0])

    z = np.array([[0], [0], [0]])
    temp = np.concatenate((z, vec_unit), axis=1)
    pt = np.cumsum(temp, axis=1)

    pt = np.concatenate((pt + np.tile(hp, (1, 10)), pt_ref), axis=1)

    # Construct the larva
    # Locate center of the head to use as origin for rotation of the larva.
    # This is consistent with the way in which the parameters of the model are
    # computed during optimization
    resolution = 75
    x_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)
    y_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)
    z_c = (np.linspace(0, 1, num=resolution) * size_lut_3d)

    [x_c, y_c, z_c] = np.meshgrid(x_c, y_c, z_c)

    x_c = x_c.transpose()
    x_c = x_c.flatten()

    y_c = y_c.transpose()
    y_c = y_c.flatten()

    z_c = z_c.transpose()
    z_c = z_c.flatten()

    pt_original = np.zeros((3,3))
    pt_original[:,1] = np.array([size_lut_3d/2, size_lut_3d/2, size_lut_3d/2])
    pt_original[:,0] = pt_original[:,1] - np.array([seglen,0,0])

    #This line was modified with the use of temp
    temp = Rz(x[4]) * Ry(x[13]) * np.array([[seglen], [0], [0]])
    pt_original[:,2] = pt_original[:,1] + np.array([temp[0,0],temp[1,0],temp[2,0]])
    hinge = pt_original[:,2] # COM of the fish
    # Calculate the 3D fish
    # eye_br = 120.8125; % 150.8125;
    # head_br = 15.953318957123471;
    # belly_br = 17.05897936938326;
    # eye_br = eye_br*4.1; head_br = head_br*3.1; belly_br = belly_br*3.6;
    # %eye_br = eye_br*0.033; head_br = head_br*0.021; belly_br = belly_br*0.028;
    #%% Translate the model to overlap with the cropped image
    vec_13 = hp - pt[:,2].reshape(-1,1)
    vec_13 = np.tile(vec_13,(1,12))
    pt = pt + vec_13
    ref_vec = pt[:,2] - hinge
    eye_br = 13
    head_br = 13
    belly_br = 13
    # Render and project the 3D fish

    random_vector_eye = np.random.rand(5)

    [eye1_model,eye1_c] = eye1model(x_c, y_c, z_c, seglen, x[4], x[13], eye_br, size_lut_3d, random_vector_eye)
    [eye2_model,eye2_c] = eye2model(x_c, y_c, z_c, seglen, x[4], x[13], eye_br, size_lut_3d, random_vector_eye)
    [belly_model, belly_c] = bellymodel(x_c, y_c, z_c, seglen, x[4], x[13], belly_br, size_lut_3d)
    head_model = headmodel(x_c, y_c, z_c, seglen, x[4], x[13], head_br, size_lut_3d)
    # model = max(max(max(eye1_model,eye2_model),head_model),belly_model);
    # [model_X, model_Y, model_Z, indices] = reorient_model(model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge);
    # [graymodel_b, graymodel_s1, graymodel_s2] = project_camera_copy(model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2);

    # Eye1
    [model_X, model_Y, model_Z, indices] = reorient_model(eye1_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge)

    #This if probably necessay, can be fixed to made better later
    model_X = np.array(model_X)
    model_X = model_X[0, :]

    model_Y = np.array(model_Y)
    model_Y = model_Y[0, :]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0, :]

    [eye1_b, eye1_s1, eye1_s2] = project_camera_copy(eye1_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2)

    # Eye2
    [model_X, model_Y, model_Z, indices] = reorient_model(eye2_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge)
    model_X = np.array(model_X)
    model_X = model_X[0, :]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0, :]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0, :]
    [eye2_b, eye2_s1, eye2_s2] = project_camera_copy(eye2_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2)

    # Head
    [model_X, model_Y, model_Z, indices] = reorient_model(head_model,x_c,y_c,z_c,heading,inclination,roll,ref_vec,hinge)
    model_X = np.array(model_X)
    model_X = model_X[0, :]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0, :]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0, :]
    [head_b, head_s1, head_s2] = project_camera_copy(head_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2)

    # Bellymodel
    [model_X, model_Y, model_Z, indices] = reorient_belly_model(belly_model,
                                                                x_c,y_c,z_c,heading,x[4]+0.3*x[5],inclination,x[13]+0.3*x[14],roll,ref_vec,pt_original[:,0],hinge)
    model_X = np.array(model_X)
    model_X = model_X[0, :]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0, :]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0, :]

    [belly_b, belly_s1, belly_s2] = project_camera_copy(belly_model,model_X,model_Y,model_Z,proj_params,indices,cb,cs1,cs2);


    eye_br = 112
    head_br = 67
    belly_br = 85

    #eye_scaling = max(eye_br/double(max(max(eye1_b))),eye_br/double(max(max(eye2_b))));
    #eye1_b = eye1_b*(eye_scaling);
    #eye2_b = eye2_b*(eye_scaling);

    eye1_b = mat2gray(eye1_b)
    eye2_b = mat2gray(eye2_b)
    eye1_b = 2*(sigmoid(eye1_b, 2.5) - 0.5)*eye_br
    eye2_b = 2*(sigmoid(eye2_b, 2.5) - 0.5)*eye_br
    #head_b = head_b*(head_br/double(max(max(head_b))))
    head_b = mat2gray(head_b)
    head_b = 2*(sigmoid(head_b,3) - 0.5)*head_br
    #belly_b = belly_b*(belly_br/double(max(max(belly_b))));
    belly_b = mat2gray(belly_b)
    belly_b = 2*(sigmoid(belly_b,3) - 0.5)*belly_br

    #eye_scaling = max(eye_br/double(max(max(eye1_s1))),eye_br/double(max(max(eye2_s1))));
    #eye1_s1 = eye1_s1*(eye_br/double(max(max(eye1_s1))));
    eye1_s1 = mat2gray(eye1_s1)
    eye2_s1 = mat2gray(eye2_s1)
    eye1_s1 = 2*(sigmoid(eye1_s1,2.8) - 0.5)*eye_br
    eye2_s1 = 2*(sigmoid(eye2_s1,2.8) - 0.5)*eye_br
    #eye2_s1 = eye2_s1*(eye_br/double(max(max(eye2_s1))));
    #head_s1 = head_s1*(head_br/double(max(max(head_s1))));
    head_s1 = mat2gray(head_s1)
    head_s1 = 2*(sigmoid(head_s1,2) - 0.5)*head_br
    #belly_s1 = belly_s1*(belly_br/double(max(max(belly_s1))));
    belly_s1 = mat2gray(belly_s1)
    belly_s1 = 2*(sigmoid(belly_s1,2) - 0.5)*belly_br

    # eye_scaling = max(eye_br/double(max(max(eye1_s2))),eye_br/double(max(max(eye2_s2))));
    #eye1_s2 = eye1_s2*(eye_br/double(max(max(eye1_s2))));
    #eye2_s2 = eye2_s2*(eye_br/double(max(max(eye2_s2))));
    eye1_s2 = mat2gray(eye1_s2)
    eye2_s2 = mat2gray(eye2_s2)
    eye1_s2 = 2*(sigmoid(eye1_s2,2.8) - 0.5)*eye_br
    eye2_s2 = 2*(sigmoid(eye2_s2,2.8) - 0.5)*eye_br
    #head_s2 = head_s2*(head_br/double(max(max(head_s2))));
    head_s2 = mat2gray(head_s2)
    head_s2 = 2*(sigmoid(head_s2,2) - 0.5)*head_br
    #belly_s2 = belly_s2*(belly_br/double(max(max(belly_s2))));
    belly_s2 = mat2gray(belly_s2)
    belly_s2 = 2*(sigmoid(belly_s2,2) - 0.5)*belly_br



    graymodel_b = np.maximum(np.maximum(np.maximum(eye1_b,eye2_b),head_b),belly_b)
    graymodel_s1 = np.maximum(np.maximum(np.maximum(eye1_s1,eye2_s1),head_s1),belly_s1)
    graymodel_s2 = np.maximum(np.maximum(np.maximum(eye1_s2,eye2_s2),head_s2),belly_s2)

    [eyeCenters_X, eyeCenters_Y, eyeCenters_Z, throwaway] = reorient_model(np.array([]),
                np.array([eye1_c[0], eye2_c[0]]), np.array([eye1_c[1], eye2_c[1]]),
                np.array([eye1_c[2], eye2_c[2]]), heading, inclination, roll, ref_vec, hinge)

    eyeCenters_X = np.array(eyeCenters_X)
    eyeCenters_X = eyeCenters_X[0, :]

    eyeCenters_Y = np.array(eyeCenters_Y)
    eyeCenters_Y = eyeCenters_Y[0, :]

    eyeCenters_Z = np.array(eyeCenters_Z)
    eyeCenters_Z = eyeCenters_Z[0, :]

    # [bellyCenter_X, bellyCenter_Y, bellyCenter_Z] = reorient_belly_model([],...
    #     belly_c(1),belly_c(2),belly_c(3),heading,x(5)+0.3*x(6),inclination,x(14)+0.3*x(6),roll,...
    #     ref_vec,pt_original(:,1),hinge);

    [eye_b, eye_s1, eye_s2] = calc_proj_w_refra_cpu(np.array([eyeCenters_X, eyeCenters_Y, eyeCenters_Z]), proj_params)
    eye_3d_coor = np.array([eyeCenters_X, eyeCenters_Y, eyeCenters_Z])

    return graymodel_b, graymodel_s1, graymodel_s2, eye_b, eye_s1, eye_s2, eye_3d_coor
if 1:
    cb = np.array([259, 360, 314, 414])
    cs1 = np.array([1186, 1232, 306, 336])
    cs2 = np.array([1211, 1259, 314, 345])
    # x = np.random.rand(23) * .5 + .75
    # x = [0.78703773, 0.82178072, 1.17047819, 1.18592876, 1.21606674, 1.20760616, 0.88265817, 0.96133755, 1.0370321,  1.21312716, 0.76610274, 0.91362852, 0.9436858,  1.16572985, 1.19384144, 0.84790873, 1.14112188, 1.11211386, 1.17385469, 0.96597181, 1.00944152, 1.05469039, 1.22351609]
    # x = np.array(x)


    pi = np.pi
    x = [0, 0, 0, .3* pi, .4 * pi, .5 * pi, .6 * pi, .7 * pi, .8* pi, .9 * pi, pi, 1.1* pi, 1.2*  pi, 1.3 * pi, 1.4 * pi, 1.5* pi, 1.6 * pi,1.7 *  pi, 1.8 *  pi, 1.9 * pi , 2*  pi, .1*  pi]
    x = np.array(x)

    (grayb, grays1, grays2, eyeb, eyes1, eyes2, eyeCoor) = return_head_real_model_new(x,1,"../Matlab_Data/proj_params_101019_corrected_new",cb,cs1,cs2)

    print(grayb)

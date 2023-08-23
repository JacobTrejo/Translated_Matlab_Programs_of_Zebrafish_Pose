#Written by Aniket Ravan
# 5th of May 2019
#Last edit on 4th of September 2019
import numpy as np
import cv2 as cv
import scipy

from Translated_Python_Programs import bellymodel, calc_proj_w_refra_cpu_v3, eye1model, eye2model, headmodel, \
    project_camera_copy, reorient_model


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
    vec_ref_1 = np.array([[seglen],[0],[0]])
    vec_ref_2 = np.array([[0],[seglen],[0]])
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

    [x_c,y_c,z_c] = np.meshgrid(x_c,y_c,z_c)

    x_c = x_c.transpose()
    x_c = x_c.flatten()

    y_c = y_c.transpose()
    y_c = y_c.flatten()

    z_c = z_c.transpose()
    z_c = z_c.flatten()

    pt_original = np.zeros((3,3))
    pt_original[:,1] = np.array([size_lut_3d/2, size_lut_3d/2, size_lut_3d/2])
    pt_original[:,0] = pt_original[:,1] - np.array([seglen,0,0])
    pt_original[:, 2] = pt_original[:, 1] + np.array([seglen, 0, 0])


    hinge = pt_original[:,2]

    vec_13 = pt[:,0] - pt[:, 2]
    vec_13 = np.array([[vec_13[0]],[vec_13[1]],[vec_13[2]]])

    vec_13 = np.tile(vec_13,(1,12))

    pt = pt + vec_13

    ref_vec = pt[:,2] - hinge
    eye_br = 13
    head_br = 13
    belly_br = 13

    random_vector_eye = np.random.rand(5)

    [eye1_model, eye1_c] = eye1model.eye1model(x_c, y_c, z_c, seglen, eye_br, size_lut_3d, random_vector_eye)
    [eye2_model, eye2_c] = eye2model.eye2model(x_c, y_c, z_c, seglen, eye_br, size_lut_3d, random_vector_eye)

    belly_model = bellymodel.bellymodel(x_c, y_c, z_c, seglen, belly_br, size_lut_3d)

    head_model = headmodel.headmodel(x_c, y_c, z_c, seglen, head_br, size_lut_3d)



    model_X, model_Y, model_Z, indices = reorient_model.reorient_model(eye1_model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)
    model_X = np.array(model_X)
    model_X = model_X[0,:]


    model_Y = np.array(model_Y)
    model_Y = model_Y[0,:]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0,:]

    [eye1_b, eye1_s1, eye1_s2] = project_camera_copy.project_camera_copy(eye1_model, model_X, model_Y, model_Z, proj_params, indices, cb, cs1, cs2)


    [model_X, model_Y, model_Z, indices] = reorient_model.reorient_model(eye2_model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)
    model_X = np.array(model_X)
    model_X = model_X[0,:]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0,:]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0,:]
    [eye2_b, eye2_s1, eye2_s2] = project_camera_copy.project_camera_copy(eye2_model, model_X, model_Y, model_Z, proj_params, indices, cb, cs1, cs2)

    [model_X, model_Y, model_Z, indices] = reorient_model.reorient_model(head_model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)
    model_X = np.array(model_X)
    model_X = model_X[0,:]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0,:]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0,:]
    [head_b, head_s1, head_s2] = project_camera_copy.project_camera_copy(head_model, model_X, model_Y, model_Z, proj_params, indices, cb, cs1, cs2)

    [model_X, model_Y, model_Z, indices] = reorient_model.reorient_model(belly_model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge)
    model_X = np.array(model_X)
    model_X = model_X[0,:]
    model_Y = np.array(model_Y)
    model_Y = model_Y[0,:]
    model_Z = np.array(model_Z)
    model_Z = model_Z[0,:]
    [belly_b, belly_s1, belly_s2] = project_camera_copy.project_camera_copy(belly_model, model_X, model_Y, model_Z, proj_params, indices, cb, cs1, cs2)

    eye_br = 114 + (np.random.rand() - 0.5) * 5
    head_br = 72 + (np.random.rand() - 0.5) * 5
    belly_br = 83 + (np.random.rand() - 0.5) * 5


    eye_scaling = np.maximum(eye_br / float(max(eye1_b.max(axis=0))), eye_br / float(max(eye2_b.max(axis=0))))
    eye1_b = eye1_b * (eye_scaling)
    eye2_b = eye2_b * (eye_scaling)

    head_b = head_b * (head_br / float(max(head_b.max(axis=0))))
    belly_b = belly_b * (belly_br / float(max(belly_b.max(axis=0))))


    eye_scaling = max(eye_br / float(max(eye1_s1.max(axis=0))), eye_br / float(max(eye2_s1.max(axis=0))))
    eye1_s1 = eye1_s1 * (eye_scaling)


    eye2_s1 = eye2_s1 * (eye_br / float(max(eye2_s1.max(axis=0))))
    head_s1 = head_s1 * (head_br / float(max(head_s1.max(axis=0))))
    belly_s1 = belly_s1 * (belly_br / float(max(belly_s1.max(axis=0))))
    eye_scaling = max(eye_br / float(max(eye1_s2.max(axis=0))), eye_br / float(max(eye2_s2.max(axis=0))))
    eye1_s2 = eye1_s2 * (eye_scaling)
    eye2_s2 = eye2_s2 * (eye_scaling)
    head_s2 = head_s2 * (head_br / float(max(head_s2.max(axis=0))))
    belly_s2 = belly_s2 * (belly_br / float(max(belly_s2.max(axis=0))))

    graymodel_b = np.maximum(np.maximum(np.maximum(eye1_b, eye2_b), head_b), belly_b)
    graymodel_s1 = np.maximum(np.maximum(np.maximum(eye1_s1, eye2_s1), head_s1), belly_s1)
    graymodel_s2 = np.maximum(np.maximum(np.maximum(eye1_s2, eye2_s2), head_s2), belly_s2)

    [eyeCenters_X, eyeCenters_Y, eyeCenters_Z, throwaway] = reorient_model.reorient_model(np.array([]), np.array([eye1_c[0], eye2_c[0]]), np.array([eye1_c[1], eye2_c[1]]), np.array([eye1_c[2], eye2_c[2]]), heading, inclination, roll, ref_vec, hinge)

    eyeCenters_X = np.array(eyeCenters_X)
    eyeCenters_X = eyeCenters_X[0,:]

    eyeCenters_Y = np.array(eyeCenters_Y)
    eyeCenters_Y = eyeCenters_Y[0,:]

    eyeCenters_Z = np.array(eyeCenters_Z)
    eyeCenters_Z = eyeCenters_Z[0,:]

    [eye_b, eye_s1, eye_s2] = calc_proj_w_refra_cpu_v3.calc_proj_w_refra_cpu(np.array([eyeCenters_X, eyeCenters_Y, eyeCenters_Z]), proj_params)

    eye_3d_coor = np.array([eyeCenters_X, eyeCenters_Y, eyeCenters_Z])

    # Had to be switched since the available python enviorments on HAL do not have scipy
    # beware that cv can introduce some errors on the corners of images when testing this function
    # graymodel_b = (scipy.signal.medfilt2d(graymodel_b))
    # graymodel_s1 = (scipy.signal.medfilt2d(graymodel_s1))
    # graymodel_s2 = (scipy.signal.medfilt2d(graymodel_s2))


    graymodel_b = np.float32(graymodel_b)
    graymodel_b = cv.medianBlur(graymodel_b, 3)
    graymodel_s1 = np.float32(graymodel_s1)
    graymodel_s1 = cv.medianBlur(graymodel_s1,3)
    graymodel_s2 = np.float32(graymodel_s2)
    graymodel_s2 = cv.medianBlur(graymodel_s2,3)

    return graymodel_b, graymodel_s1, graymodel_s2, eye_b, eye_s1, eye_s2, eye_3d_coor
if 0:
    cb = np.array([259, 360, 314, 414])
    cs1 = np.array([1186, 1232, 306, 336])
    cs2 = np.array([1211, 1259, 314, 345])
    x = np.random.rand(23) * .5 + .75
    x = [0.78703773, 0.82178072, 1.17047819, 1.18592876, 1.21606674, 1.20760616, 0.88265817, 0.96133755, 1.0370321,  1.21312716, 0.76610274, 0.91362852, 0.9436858,  1.16572985, 1.19384144, 0.84790873, 1.14112188, 1.11211386, 1.17385469, 0.96597181, 1.00944152, 1.05469039, 1.22351609]
    x = np.array(x)


    pi = np.pi
    x = [0, 0, 0, .3* pi, .4 * pi, .5 * pi, .6 * pi, .7 * pi, .8* pi, .9 * pi, pi, 1.1* pi, 1.2*  pi, 1.3 * pi, 1.4 * pi, 1.5* pi, 1.6 * pi,1.7 *  pi, 1.8 *  pi, 1.9 * pi , 2*  pi, .1*  pi]
    x = np.array(x)

    (b, x, y, z, q) = return_head_real_model_new(x,1,"proj_params_101019_corrected_new",cb,cs1,cs2)

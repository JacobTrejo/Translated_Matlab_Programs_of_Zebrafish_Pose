import numpy as np
import math as m

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

def reorient_model(model, x_c, y_c, z_c, heading, inclination, roll, ref_vec, hinge):

    if (np.any(model) == False):
        length = max(np.shape(x_c))
        indices = np.linspace(0,length-1,num=(length),dtype= int)
    else:
        indices = np.argwhere(model)
        indices = indices[:,0]

    R = Rz(heading) @ Ry(inclination) @ Rx(roll)

    new_coor = R @ np.array([x_c[indices] - hinge[0], y_c[indices] - hinge[1], z_c[indices] - hinge[2]])

    X = new_coor[0,:] + hinge[0] + ref_vec[0]
    Y = new_coor[1,:] + hinge[1] + ref_vec[1]
    Z = new_coor[2,:] + hinge[2] + ref_vec[2]

    return X, Y, Z, indices


if 0:
    result = reorient_model(np.array([1,2,3,4]),np.array([1,2,3,4]),np.array([5,6,7,8]),np.array([9,10,11,12]),10,20,30,np.array([1,2,3]),np.array([1,2,3]))







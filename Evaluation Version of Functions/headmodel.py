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
def headmodel(x, y, z, seglen, theta1,phi1, brightness, size_lut):
    head_w = seglen * 0.6962
    head_l = seglen * 0.7675 # 0.8196; % 1 / sqrt(2) 7475
    head_l = seglen * 0.8575
    head_h = seglen * 0.6426 # 0.7622; % 0.7 / sqrt(2)
    head_h = seglen * 0.8226
    head_h = seglen * 0.7626
    c_head = 1.1971 # 1.1296;

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2 , size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    temp = Rz(theta1) * Ry(-phi1) * np.array([[seglen], [0], [0]])
    pt_original[:, 2] = pt_original[:, 1] + [temp[0,0],temp[1,0],temp[2,0]]
    # R = rotz(heading) * roty(inclination) * rotx(roll);
    head_c = [c_head * pt_original[0,0] + (1 - c_head) * pt_original[0,1],
    c_head * pt_original[1,0] +
              (1 - c_head) * pt_original[1,1],
    pt_original[2,1] - seglen / (8.3590 + (np.random.rand() - 0.5) * 0.05)] # 3.4609];
    # head_c = head_c - pt_original(:, 2);
    # head_c = R * head_c + pt_original(:, 2);

    XX = x - head_c[0]
    YY = y - head_c[1]
    ZZ = z - head_c[2]
    # rot_mat = rotx(-roll) * roty(-inclination) * rotz(-heading);
    # XX = rot_mat(1, 1) * XX + rot_mat(1, 2) * YY + rot_mat(1, 3) * ZZ;
    # YY = rot_mat(2, 1) * XX + rot_mat(2, 2) * YY + rot_mat(2, 3) * ZZ;
    # ZZ = rot_mat(3, 1) * XX + rot_mat(3, 2) * YY + rot_mat(3, 3) * ZZ;

    head_model = np.exp(-2 * (XX * XX / (2 * head_l ** 2) + YY * YY / (2 * head_w ** 2) +
                              ZZ * ZZ / (2 * head_h ** 2) - 1))
    head_model = head_model * brightness

    return head_model

if 0:
    x = [1 ,2, 3]
    y = [1 ,2, 3]
    z = [1,2,3]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    print(headmodel(x,y,z,.9,1.2,.7,4, 1))
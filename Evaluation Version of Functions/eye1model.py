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

def eye1model(x, y, z, seglen, theta1, phi1, brightness, size_lut, rnd):
    d_eye = seglen * (0.8356)# 1.4332; %0.83
    c_eyes = 1.5230 #1.3015;
    eye1_w = seglen * 0.2097 #0.1671; % 0.35
    eye1_w = seglen * 0.2597
    eye1_w = seglen * 0.2997
    eye1_l = seglen * 0.3306 #0.2507; % 0.45 3006
    eye1_l = seglen * 0.4306
    eye1_h = seglen * 0.2496 #0.2661; % 0.35
    eye1_h = seglen * 0.3096
    # R = rotz(heading)*roty(inclination)*rotx(roll);

    pt_original = np.empty([3,3])

    pt_original[:, 1] = np.array([size_lut / 2, size_lut / 2, size_lut / 2])
    pt_original[:, 0] = np.array(pt_original[:, 1] - [seglen, 0, 0])
    temp = Rz(theta1) @ Ry(-phi1) @ np.array([[seglen], [0], [0]])
    pt_original[:, 2] = pt_original[:,1] + np.array([temp[0,0],temp[1,0],temp[2,0]])

    eye1_c = [c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1], c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2, pt_original[2, 1] - seglen / 6.3049]
    # eye1_c = eye1_c - pt_original(:,2);
    # eye1_c = R*eye1_c + pt_original(:,2);

    XX = x - eye1_c[0]
    YY = y - eye1_c[1]
    ZZ = z - eye1_c[2]
    # rot_mat = rotx(-roll)*roty(-inclination)*rotz(-heading);
    # XX = rot_mat(1,1)*XX + rot_mat(1,2)*YY + rot_mat(1,3)*ZZ;
    # YY = rot_mat(2,1)*XX + rot_mat(2,2)*YY + rot_mat(2,3)*ZZ;
    # ZZ = rot_mat(3,1)*XX + rot_mat(3,2)*YY + rot_mat(3,3)*ZZ;

    eye1_model = np.exp(-1.2*(XX*XX/(2*eye1_l ** 2) + YY*YY/(2*eye1_w**2) + ZZ*ZZ/(2*eye1_h**2) - 1))
    eye1_model = eye1_model*brightness

    return eye1_model, eye1_c
if 0:
    x = [1 ,2, 3]
    y = [1 ,2, 3]
    z = [1,2,3]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    print(eye1model(x,y,z,.9,1.2,.7,4, 1, 1))


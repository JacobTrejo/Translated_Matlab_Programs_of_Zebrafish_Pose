import numpy as np


def eye2model(x, y, z, seglen, brightness, size_lut, rnd):
    d_eye = seglen * (0.8556 + (rnd[0] - 0.5)*0.05) # 1.4332 #0.83
    c_eyes = 1.4230 #1.3015

    eye2_w = seglen * (0.3197 + (rnd[1] - 0.5)*0.02)

    eye2_l = seglen * (0.4756 + (rnd[2] - 0.5)*0.02)

    eye2_h = seglen * (0.2996 + (rnd[3] - 0.5)*0.02)

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2, size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    pt_original[:,2] = pt_original[:,1] + [seglen, 0, 0]
    eye2_c = [c_eyes*pt_original[0,0] + (1-c_eyes)*pt_original[0,1], c_eyes*pt_original[1,0] + (1-c_eyes)*pt_original[1,1] - d_eye/2, pt_original[2,1] - seglen/7.3049]


    XX = x - eye2_c[0]
    YY = y - eye2_c[1]
    ZZ = z - eye2_c[2]

    eye2_model = np.exp(-1.2*(XX*XX/(2*eye2_l**2) + YY*YY/(2*eye2_w**2) + ZZ*ZZ/(2*eye2_h**2) - 1))
    eye2_model = eye2_model*brightness

    return eye2_model, eye2_c

if 0:
    print(eye2model(1,2,3,4,5,6,np.array([1,2,3,4])))



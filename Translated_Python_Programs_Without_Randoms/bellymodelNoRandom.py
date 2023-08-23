import numpy as np

def bellymodel(x, y, z, seglen, brightness, size_lut,rand1,rand2,rand3,rand4,rand5):
    belly_w = seglen * (0.499 + (rand1 - 0.5)*0.03)
    belly_l = seglen * (1.2500 + (rand2 - 0.5)*0.07)

    belly_h = seglen * (0.7231 + (rand3 - 0.5)*0.03)
    c_belly = 1.0541 + (rand4 - 0.5)*0.03

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2, size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    pt_original[:,2] = pt_original[:,1] + [seglen, 0, 0]
    belly_c = [c_belly*pt_original[0,1] + (1-c_belly)*pt_original[0,2], c_belly*pt_original[1,1] + (1-c_belly)*pt_original[1,2], pt_original[2,1] - seglen/(6+(rand5 - 0.5)*0.05)]


    XX = x - belly_c[0]
    YY = y - belly_c[1]
    ZZ = z - belly_c[2]

    belly_model = np.exp(-2*(XX * XX / (2 * belly_l**2) + YY*YY/(2* belly_w**2) + ZZ*ZZ/(2* belly_h**2) - 1))
    belly_model = belly_model*brightness
    return belly_model

if 1==0:
    print(bellymodel(1,2,3,4,5,6,7,8,9,10,20))

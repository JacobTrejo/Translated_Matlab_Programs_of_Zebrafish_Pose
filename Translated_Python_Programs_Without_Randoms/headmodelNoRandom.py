import numpy as np
def headmodel(x, y, z, seglen, brightness, size_lut, rand1, rand2, rand3, rand4):
    head_w = seglen * (0.6962 + (rand1 - 0.5)*0.03)
    head_l = seglen * (0.8475 + (rand2 - 0.5)*0.03)
    head_h = seglen * (0.7926 + (rand3 - 0.5)*0.03)
    c_head = 1.1771

    pt_original = np.empty((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2 , size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    pt_original[:,2] = pt_original[:,1] + [seglen, 0, 0]

    head_c = [c_head*pt_original[0,0] + (1-c_head)*pt_original[0,1], c_head*pt_original[1,0] + (1-c_head)*pt_original[1,1], pt_original[2,1] - seglen/(9.3590 + (rand4 - 0.5)*0.05)]

    XX = x - head_c[0]
    YY = y - head_c[1]
    ZZ = z - head_c[2]


    head_model = np.exp(-2*(XX*XX/(2*head_l**2) + YY*YY/(2*head_w**2) + ZZ*ZZ/(2*head_h**2) - 1))
    head_model = head_model*brightness

    return head_model



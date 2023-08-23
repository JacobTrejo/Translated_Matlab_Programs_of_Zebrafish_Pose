import numpy as np
from Translated_Python_Programs_Without_Randoms import gen_lut_s_tail_no_random


def roundHalfUp(a):
    return (np.floor(a)+ np.round(a - (np.floor(a)-1))-1)

def uint8(a):
    a = roundHalfUp(a)

    if np.ndim(a) == 0:
        if a <0:
            a = 0
        if a > 255:
            a = 255
    else:
        a[a>255]=255
        a[a<0]=0
    return a

def linIndxTo2DIndx(num, arrShape):
    rows = arrShape[0]
    columns = arrShape[1]
    columnIndx = num / rows
    if np.floor(columnIndx) == columnIndx and columnIndx != 0:
        columnIndx = columnIndx - 1
    else:
        columnIndx = np.floor(columnIndx)

    rowIndx = (num % rows) - 1

    if rowIndx == -1:
        rowIndx = rows - 1

    return (int(rowIndx),int(columnIndx))


def view_s_lut_new_real_cpu(crop_coor, pt,lut_tail,projection,imageSizeX,imageSizeY,r1,r2):

    # Find the coefficients of the line that defines the refracted ray





    vec_pt = pt[:,1:10] - pt[:,0:9]

    segslen = (np.sum(vec_pt*vec_pt,0))**(1/2)

    segslen = np.tile(segslen, (2, 1))

    vec_pt_unit = vec_pt /segslen

    theta_prime = np.arctan2(vec_pt_unit[1, :], vec_pt_unit[0, :])
    theta = np.zeros((2, max(theta_prime.shape)))
    theta[0, :] = theta_prime
    theta[1:] = theta_prime



    # shift pts to the cropped images

    pt[0,:] = pt[0,:] - crop_coor[2] + 1

    pt[1,:] = pt[1,:] - crop_coor[0] + 1


    ###These three are all uint8
    imblank = np.zeros((imageSizeY,imageSizeX))

    imblank_cpu = np.zeros((imageSizeY,imageSizeX))

    bodypix = imblank_cpu





    headpix = uint8(uint8(projection/1.8)*5.2)



    # tail

    size_lut = 15

    size_half = (size_lut+1)/2

    seglen = segslen


    seglen[seglen<0.2] = 0.1


    seglen[seglen>7.9] = 8



    seglenidx = roundHalfUp(seglen/0.2)

    coor_t = np.floor(pt)

    dt = np.floor((pt - coor_t)*5) + 1

    at = np.mod(np.floor(theta*90/np.pi),180) + 1

    #Uncomment to fix
    # seglenidx = seglenidx[0,:]

    for ni in range(0,7):

        n = ni+2

        tailpix = imblank

        newIndex = linIndxTo2DIndx(n+1, seglenidx.shape)
        tail_model = gen_lut_s_tail_no_random.gen_lut_s_tail(ni + 1, seglenidx[newIndex], dt[0,n], dt[1,n], at[0,n], r1)
        # tail_model = gen_lut_s_tail_no_random.gen_lut_s_tail(ni + 1, seglenidx[n], dt[0,n], dt[1,n], at[0,n], r1)
        j = (int(max(1, coor_t[1, n] - (size_half - 1))) - 1, int(min(imageSizeY, coor_t[1, n] + (size_half - 1))), int(max(0, coor_t[0, n] - (size_half - 1))) - 1, int(min(imageSizeX, coor_t[0, n] + (size_half - 1))))
        k = (int(max((size_half + 1) - coor_t[1, n], 1))-1, int(min(imageSizeY - coor_t[1, n] + size_half, size_lut)), int(max((size_half + 1 ) - coor_t[0, n], 1))-1, int(min(imageSizeX - coor_t[0, n] + size_half, size_lut)))

        tailpix[int(max(1, coor_t[1, n] - (size_half - 1))) - 1: int(min(imageSizeY, coor_t[1, n] + (size_half - 1))), int(max(1, coor_t[0, n] - (size_half - 1))) - 1: int(min(imageSizeX, coor_t[0, n] + (size_half - 1)))] = tail_model[int(max((size_half + 1) - coor_t[1, n], 1))-1: int(min(imageSizeY - coor_t[1, n] + size_half, size_lut)), int(max((size_half + 1 ) - coor_t[0, n], 1))-1: int(min(imageSizeX - coor_t[0, n] + size_half, size_lut))]

        bodypix = np.maximum(bodypix, tailpix)
        bodypix = uint8(bodypix)




    graymodel = np.maximum(headpix,uint8((r2)*bodypix))
    graymodel = uint8(graymodel)

    return graymodel

if 0:
    pt = [[0.2490,    0.7227,    0.0044,    0.9092,    0.9923,    0.9971,    0.8367,    0.3786,    0.0464,    0.5091], [0.1512,    0.1215,    0.9262,    0.3204,    0.1095,    0.7785,    0.4049,    0.0245,    0.1281,    0.6583]]
    pt = np.array(pt)
    pt = pt + 35

    (imageSizeX, imageSizeY) = (100,100)
    proj = np.zeros((imageSizeX, imageSizeY))

    crop_coor = [30,0,25]

    a = view_s_lut_new_real_cpu(crop_coor,pt,1,proj,imageSizeX,imageSizeY,.3,.2)

    print('heeelooo')

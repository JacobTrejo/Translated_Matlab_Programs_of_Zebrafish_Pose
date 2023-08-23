# generate look up table for tail
import numpy as np
import scipy


def roundHalfUp(a):
    return (np.floor(a) + np.round(a - (np.floor(a) - 1)) - 1)

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

def ind2sub(arr):
    ys = []
    xs = []
    sz = arr.shape
    for x in range(0, sz[0]):
        for y in range(0, sz[1]):
            if arr[y, x] > 0:
                ys.append(y)
                xs.append(x)
    return np.array(ys), np.array(xs)

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(x-float(mean))**2/(2*var))
    return num/denom


def gen_lut_s_tail(n, seglenidx, d1, d2, a,rand):
    size_lut = 15

    size_half = (size_lut + 1) / 2

    imblank = np.zeros((size_lut, size_lut))

    imageSizeX = size_lut

    imageSizeY = size_lut

    random_number = rand

    # size of the balls in the model

    temp = [2.5, 2.4, 2.3, 2.2, 1.8, 1.5, 1.3, 1.2]
    temp = np.array(temp)
    ballsize = random_number * temp

    # thickness of the sticks in the model
    temp = [8, 7, 6, 5, 4, 3, 2.5, 2.5]
    temp = np.array(temp)
    thickness = random_number * temp

    # brightness of the tail

    b_tail = [0.5, 0.45, 0.4, 0.32, 0.28, 0.24, 0.22, 0.20]
    b_tail = np.array(b_tail)

    x = np.linspace(1, imageSizeX, imageSizeX)
    y = np.linspace(1, imageSizeY, imageSizeY)

    [columnsInImage0, rowsInImage0] = np.meshgrid(x, y)

    radius = ballsize[n]

    th = thickness[n]

    # p_max = scipy.stats.norm.pdf(0, loc=0, scale=th)
    p_max = normpdf(0,0,th)


    bt_gradient = b_tail[n] / b_tail[n - 1]

    seglen = 0.2 * seglenidx

    bt = b_tail[n - 1] * (1 - 0.02 * seglenidx)

    centerX = size_half + d1 / 5

    centerY = size_half + d2 / 5

    columnsInImage = columnsInImage0

    rowsInImage = rowsInImage0

    ballpix = (rowsInImage - centerY) ** 2 + (columnsInImage - centerX) ** 2 <= radius ** 2

    ballpix = uint8(uint8(uint8(uint8(ballpix) * 255) * bt) * 0.85)

    t = 2 * np.pi * (a - 1) / 180

    pt = np.zeros((2, 2))

    R = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]

    vec = np.matmul(R, np.array([[seglen], [0]]))

    pt[:, 0] = np.array([size_half + d1 / 5, size_half + d2 / 5])

    pt[:, 1] = pt[:, 0] + vec[:, 0]
    stickpix = imblank

    columnsInImage = columnsInImage0

    rowsInImage = rowsInImage0

    if (pt[0, 1] - pt[0, 0]) != 0:

        slope = (pt[1, 1] - pt[1, 0]) / (pt[0, 1] - pt[0, 0])

        # vectors perpendicular to the line segment

        # th is the thickness of the sticks in the model

        vp = np.array([[-slope], [1]]) / np.linalg.norm(np.array([[-slope], [1]]))

        # one vertex of the rectangle

        # POSSIBLE SOURCE OF ERROR
        V1 = pt[:, 1] - vp[:, 0] * th

        # two sides of the rectangle

        s1 = 2 * vp * th
        s2 = pt[:, 0] - pt[:, 1]

        # find the pixels inside the rectangle

        r1 = rowsInImage - V1[1]

        c1 = columnsInImage - V1[0]

        # inner products

        ip1 = r1 * s1[1] + c1 * s1[0]

        ip2 = r1 * s2[1] + c1 * s2[0]

        stickpix_bw = (ip1 > 0) * (ip1 < np.dot(s1[:, 0], s1[:, 0])) * (ip2 > 0) * (ip2 < np.dot(s2, s2))



    else:
        stickpix_bw = (rowsInImage < max(pt[1, 1], pt[1, 0])) * (rowsInImage > min(pt[1, 1], pt[1, 0])) * (
                    columnsInImage < pt[0, 1] + th) * (columnsInImage > pt[0, 1] - th)

    # the brightness of the points on the stick is a function of its

    # distance to the segment

    #[ys, xs] = ind2sub(stickpix_bw)
    idx_bw = np.argwhere(stickpix_bw >0)
    ys = idx_bw[:, 0]
    xs = idx_bw[:, 1]

    px = pt[0, 1] - pt[0, 0]

    py = pt[1, 1] - pt[1, 0]

    pp = px * px + py * py

    # the distance between a pixel and the fish backbone

    d_radial = np.zeros((max(ys.shape), 1))

    # the distance between a pixel and the anterior end of the

    # segment (0 < d_axial < 1)

    b_axial = np.zeros((max(ys.shape), 1))

    for i in range(0, max(ys.shape)):
        u = (((xs[i] + 1) - pt[0, 0]) * px + ((ys[i] + 1) - pt[1, 0]) * py) / pp

        dx = pt[0, 0] + u * px - xs[i] - 1

        dy = pt[1, 0] + u * py - ys[i] - 1

        d_radial[i] = dx * dx + dy * dy

        b_axial[i] = 1 - (1 - bt_gradient) * u * 0.9

    #b_stick = scipy.stats.norm.pdf(d_radial, 0, th) / p_max * 255
    b_stick = normpdf(d_radial, 0, th) / p_max * 255


    b_stick = uint8(b_stick)


    for i in range(0, max(ys.shape)):
        stickpix[ys[i], xs[i]] = uint8(b_stick[i] * b_axial[i])

    stickpix = stickpix * bt
    stickpix = uint8(stickpix)

    graymodel = np.maximum(ballpix, stickpix)
    graymodel = uint8(graymodel)

    return graymodel

if 0:
    a = gen_lut_s_tail(5, 6, 1, 1, 0, 1)
    print(a)

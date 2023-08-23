import numpy as np


def interparc(t,x,y,z):

    pxyz = np.zeros((max(x.shape),3))
    pxyz[:,0] = x
    pxyz[:,1] = y
    pxyz[:,2] = z

    pt = np.zeros((max(t.shape),3))

    #row vector instead of column vector like in matlab
    chordlen = (np.sum(np.diff(pxyz,axis=0) **2,axis=1))**.5
    chordlen = chordlen / np.sum(chordlen)

    cumarc = np.concatenate(([0], np.cumsum(chordlen)))

    spl = [0,0,0]
    spld = spl

    diffarray = [[3,0,0],
                 [0,2,0],
                 [0,0,1],
                 [0,0,0]]

    diffarray = np.array(diffarray)

    for x in range(0,3):
        spl[x] =




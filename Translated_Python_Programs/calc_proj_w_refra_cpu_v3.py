import numpy as np
from scipy.io import loadmat

# This function does triangulation and refraction calibration
#  the input is the 3D coordinates of the model
#  the output is the 2D coordinates in three views after refraction
#  camera a corresponds to view s1
#  camera b corresponds to b
#  camera c corresponds to s2
#
#  The center of the tank is (35.2,27.5,35.7);
#  The sensor of camera a is parallel to x-y plane
#  camera b parallel to x-z
#  camera c parallel to y-z

def calc_proj_w_refra_cpu(coor_3d,proj_params):

    mat = loadmat(proj_params)
    proj_params_mat = mat['proj_params']

    fa1p00 = proj_params_mat[0,0]
    fa1p10 = proj_params_mat[0,1]
    fa1p01 = proj_params_mat[0,2]
    fa1p20 = proj_params_mat[0,3]
    fa1p11 = proj_params_mat[0,4]
    fa1p30 = proj_params_mat[0,5]
    fa1p21 = proj_params_mat[0,6]
    fa2p00 = proj_params_mat[1,0]
    fa2p10 = proj_params_mat[1,1]
    fa2p01 = proj_params_mat[1,2]
    fa2p20 = proj_params_mat[1,3]
    fa2p11 = proj_params_mat[1,4]
    fa2p30 = proj_params_mat[1,5]
    fa2p21 = proj_params_mat[1,6]
    fb1p00 = proj_params_mat[2,0]
    fb1p10 = proj_params_mat[2,1]
    fb1p01 = proj_params_mat[2,2]
    fb1p20 = proj_params_mat[2,3]
    fb1p11 = proj_params_mat[2,4]
    fb1p30 = proj_params_mat[2,5]
    fb1p21 = proj_params_mat[2,6]
    fb2p00 = proj_params_mat[3,0]
    fb2p10 = proj_params_mat[3,1]
    fb2p01 = proj_params_mat[3,2]
    fb2p20 = proj_params_mat[3,3]
    fb2p11 = proj_params_mat[3,4]
    fb2p30 = proj_params_mat[3,5]
    fb2p21 = proj_params_mat[3,6]
    fc1p00 = proj_params_mat[4,0]
    fc1p10 = proj_params_mat[4,1]
    fc1p01 = proj_params_mat[4,2]
    fc1p20 = proj_params_mat[4,3]
    fc1p11 = proj_params_mat[4,4]
    fc1p30 = proj_params_mat[4,5]
    fc1p21 = proj_params_mat[4,6]
    fc2p00 = proj_params_mat[5,0]
    fc2p10 = proj_params_mat[5,1]
    fc2p01 = proj_params_mat[5,2]
    fc2p20 = proj_params_mat[5,3]
    fc2p11 = proj_params_mat[5,4]
    fc2p30 = proj_params_mat[5,5]
    fc2p21 = proj_params_mat[5,6]

    npts = coor_3d.shape[1]

    coor_b = np.zeros((2,npts))
    coor_s1 = np.zeros((2, npts))
    coor_s2 = np.zeros((2, npts))

    coor_b[0,:] = fa1p00 + fa1p10*coor_3d[2,:] + fa1p01*coor_3d[0,:] + fa1p20*(coor_3d[2,:]**2) + fa1p11*coor_3d[2,:]*coor_3d[0,:] + fa1p30*(coor_3d[2,:]**3) + fa1p21*(coor_3d[2,:]**2)*coor_3d[0,:]
    coor_b[1,:] = fa2p00 + fa2p10*coor_3d[2,:] + fa2p01*coor_3d[1,:] + fa2p20*(coor_3d[2,:]**2) + fa2p11*coor_3d[2,:]*coor_3d[1,:] + fa2p30*(coor_3d[2,:]**3) + fa2p21*(coor_3d[2,:]**2)*coor_3d[1,:]
    coor_s1[0,:] = fb1p00 + fb1p10*coor_3d[0,:] + fb1p01*coor_3d[1,:] + fb1p20*(coor_3d[0,:]**2) + fb1p11*coor_3d[0,:]*coor_3d[1,:] + fb1p30*(coor_3d[0,:]**3) + fb1p21*(coor_3d[0,:]**2)*coor_3d[1,:]
    coor_s1[1,:] = fb2p00 + fb2p10*coor_3d[0,:] + fb2p01*coor_3d[2,:] + fb2p20*(coor_3d[0,:]**2) + fb2p11*coor_3d[0,:]*coor_3d[2,:] + fb2p30*(coor_3d[0,:]**3) + fb2p21*(coor_3d[0,:]**2)*coor_3d[2,:]
    coor_s2[0,:] = fc1p00 + fc1p10*coor_3d[1,:] + fc1p01*coor_3d[0,:] + fc1p20*(coor_3d[1,:]**2) + fc1p11*coor_3d[1,:]*coor_3d[0,:] + fc1p30*(coor_3d[1,:]**3) + fc1p21*(coor_3d[1,:]**2)*coor_3d[0,:]
    coor_s2[1,:] = fc2p00 + fc2p10*coor_3d[1,:] + fc2p01*coor_3d[2,:] + fc2p20*(coor_3d[1,:]**2) + fc2p11*coor_3d[1,:]*coor_3d[2,:] + fc2p30*(coor_3d[1,:]**3) + fc2p21*(coor_3d[1,:]**2)*coor_3d[2,:]

    return coor_b, coor_s1, coor_s2




# if True:
#     x = np.array([0])
#     x = x * (2/10)
#     y = np.array([0])
#     y  = y * (2/8)
#     z = np.array([72.5])
#     z = z * (1)
#     d = np.array([x[:],y[:],z[:]])
#     print(x)
#     print(y)
#     print(z)
#     results = calc_proj_w_refra_cpu(d,"../Matlab_Data/proj_params_101019_corrected_new")
#     print(results)

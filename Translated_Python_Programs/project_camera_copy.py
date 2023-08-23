from Translated_Python_Programs import calc_proj_w_refra_cpu_v3
import numpy as np
def project_camera_copy(model, X, Y, Z, proj_params, indices, cb, cs1, cs2):

    (coor_b, coor_s1, coor_s2) = calc_proj_w_refra_cpu_v3.calc_proj_w_refra_cpu(np.array([X, Y, Z]), proj_params)

    coor_b[0,:] = coor_b[0,:] - cb[2]
    coor_b[1,:] = coor_b[1,:] - cb[0]
    coor_s1[0,:] = coor_s1[0,:] - cs1[2]
    coor_s1[1,:] = coor_s1[1,:] - cs1[0]
    coor_s2[0,:] = coor_s2[0,:] - cs2[2]
    coor_s2[1,:] = coor_s2[1,:] - cs2[0]

    projection_b = np.zeros((int(cb[1] - cb[0] + 1) , int(cb[3] - cb[2] + 1)))
    projection_s1 = np.zeros((int(cs1[1] - cs1[0] +1),int( cs1[3] - cs1[2] +1)))
    projection_s2 = np.zeros((int(cs2[1] - cs2[0] +1),int( cs2[3] - cs2[2] +1)))

    sz_b = np.shape(projection_b)
    sz_s1 = np.shape(projection_s1)
    sz_s2 = np.shape(projection_s2)

    count_mat_b = np.zeros(np.shape(projection_b)) + 0.0001
    count_mat_s1 = np.zeros(np.shape(projection_s1)) + 0.0001
    count_mat_s2 = np.zeros(np.shape(projection_s2)) + 0.0001



    length = max(indices.shape)
    # for x in range(0,length):
    #     if np.floor(coor_b[1,x]) > sz_b[0]-1 or np.floor(coor_b[0,x]) > sz_b[1]-1 or np.floor(coor_b[1,x]) < 0 or np.floor(coor_b[0,x] ) < 0:
    #         continue
    #     # is model getting the correct indices, it could depend on the input
    #     #print(np.floor(coor_b[1, x]), np.floor(coor_b[0, x]))
    #     projection_b[int(np.floor(coor_b[1, x])), int(np.floor(coor_b[0, x]))] = projection_b[int(np.floor(coor_b[1, x])), int(np.floor(coor_b[0, x]))] + model[indices[x]]
    #     count_mat_b[int(np.floor(coor_b[1, x])), int(np.floor(coor_b[0, x]))] = count_mat_b[int(np.floor(coor_b[1, x])), int(np.floor(coor_b[0, x]))] + 1

    x = np.linspace(0,length-1,length)
    x = x.astype(int)


    fval = np.logical_or(np.floor(coor_b[1,x]) > sz_b[0]-1, np.floor(coor_b[0,x]) > sz_b[1]-1)
    sval = np.logical_or(np.floor(coor_b[1,x]) < 0,np.floor(coor_b[0,x] ) < 0)

    finval = np.logical_not(np.logical_or(fval,sval))
    model = np.array(model)

    index1 = (np.floor(coor_b[1, x[finval]])).astype(int)
    index2 = (np.floor(coor_b[0, x[finval]])).astype(int)

    values = model[(indices[x[finval]]).astype(int)]

    np.add.at(projection_b, (index1, index2), values)
    np.add.at(count_mat_b, (index1, index2), 1)

    #projection_b = projection_b / count_mat_b
    projection_b = np.divide(projection_b,count_mat_b)





    i = np.linspace(0,length-1,length)
    i = i.astype(int)

    fval = np.logical_or(np.floor(coor_s1[1, i]) > sz_s1[0]-1,np.floor(coor_s1[0, i]) > sz_s1[1]-1)
    sval = np.logical_or(np.floor(coor_s1[1, i]) < 0,np.floor(coor_s1[0, i]) < 0)
    finval = np.logical_not(np.logical_or(fval,sval))

    index1 = (np.floor(coor_s1[1, i[finval]])).astype(int)
    index2 = (np.floor(coor_s1[0, i[finval]])).astype(int)

    values = model[(indices[i[finval]]).astype(int)]

    np.add.at(projection_s1,(index1,index2),values)
    np.add.at(count_mat_s1,(index1,index2),1)

    #projection_s1 = projection_s1 / count_mat_s1
    projection_s1 = np.divide(projection_s1,count_mat_s1)




    x = np.linspace(0, length - 1, length)
    x = x.astype(int)

    fval = np.logical_or(np.floor(coor_s2[1, x]) > sz_s2[0] - 1, np.floor(coor_s2[0, x]) > sz_s2[1] - 1)
    sval = np.logical_or(np.floor(coor_s2[1, x]) < 0, np.floor(coor_s2[0, x]) < 0)

    finval = np.logical_not(np.logical_or(fval, sval))

    index1 = (np.floor(coor_s2[1, x[finval]])).astype(int)
    index2 = (np.floor(coor_s2[0, x[finval]])).astype(int)

    values = model[(indices[x[finval]]).astype(int)]

    np.add.at(projection_s2, (index1, index2), values)
    np.add.at(count_mat_s2, (index1, index2), 1)

    #projection_s2 = projection_s2 / count_mat_s2
    projection_s2 = np.divide(projection_s2,count_mat_s2)

    return projection_b,projection_s1,projection_s2

if 1==0:
    indices = np.array([0,1,2,3])
    model = np.array([1.5,2,3,5])
    x = np.array([1.7,2.3,2.9,3])
    x = x *(2/3)
    y = np.array([2.6,3,3.5,4])
    y=y * (2/4)
    z = np.array([2.9,3.9,4.1,4.5])
    z = z * (2/4.5)
    cb = np.array([322,356,369,412])
    cs1 = np.array([0,7,0,7])
    cs2 = np.array([0,7,0,7])

    result = project_camera_copy(model,x,y,z,"../Matlab_Data/proj_params_101019_corrected_new",indices,cb,cs1,cs2)
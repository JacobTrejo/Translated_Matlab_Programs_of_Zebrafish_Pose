import numpy as np

def bellymodel(x, y, z, seglen, theta1, phi1, brightness, size_lut):

    #belly_w = seglen * (0.3527); %0.5030;5527
    belly_w = seglen * 0.45
    belly_l = seglen * 1.3000 #1.5970;
    #belly_h = seglen * 0.6531 #0.6294; % 0.55
    #belly_h = seglen * (0.7431);
    belly_h = seglen * 0.6631
    c_belly_wrt_1 = 1-0.475 # 1.0541;
    # R = rotz(heading)*roty(inclination)*rotx(roll);

    pt_original = np.zeros((3,3))
    pt_original[:,1] = [size_lut/2, size_lut/2, size_lut/2]
    pt_original[:,0] = pt_original[:,1] - [seglen, 0, 0]
    pt_original[:,2] = pt_original[:,1] + [seglen, 0, 0] #This should not be rotated
    # The belly is rotated twice in reorient_belly_model instead

    belly_c = [c_belly_wrt_1*pt_original[0,0] + (1-c_belly_wrt_1)*pt_original[0,2], c_belly_wrt_1*pt_original[1,0] +
               (1-c_belly_wrt_1)*pt_original[1,2], pt_original[2,0] - seglen/6.5] #3.5602]; % Changed from 6   7.0257
    # belly_c = belly_c - pt_original(:,2);
    # belly_c = R*belly_c  + pt_original(:,2);
    # figure,
    # plot3(pt_original(1,1), pt_original(2,1), pt_original(3,1), 'Marker', 'o',...
    #     'color', 'k','MarkerFaceColor','k')
    # hold on
    # plot3(pt_original(1,2), pt_original(2,2), pt_original(3,2), 'Marker', 'o',...
    #     'color', 'k')
    # plot3(pt_original(1,3), pt_original(2,3), pt_original(3,3),'Marker',  'o',...
    #     'color', 'k')
    # plot3(belly_c(1), belly_c(2), belly_c(3), 'Marker', '*',...
    #     'color', 'r')
    # plot3(pt_original(1,[1,3]), pt_original(2,[1,3]), pt_original(3,[1,3]),...
    #     'color', 'k')
    # axis('equal')
    XX = x - belly_c[0]
    YY = y - belly_c[1]
    ZZ = z - belly_c[2]
    # rot_mat = rotx(-roll)*roty(-inclination)*rotz(-heading);
    # XX = rot_mat(1,1)*XX + rot_mat(1,2)*YY + rot_mat(1,3)*ZZ;
    # YY = rot_mat(2,1)*XX + rot_mat(2,2)*YY + rot_mat(2,3)*ZZ;
    # ZZ = rot_mat(3,1)*XX + rot_mat(3,2)*YY + rot_mat(3,3)*ZZ;

    belly_model = np.exp(-2*(XX*XX/(2*belly_l**2) + YY*YY/(2*belly_w**2) +
                             ZZ*ZZ/(2*belly_h**2) - 1))
    belly_model = belly_model*brightness
    return belly_model, belly_c

if 0:
    x = [1, 2, 3]
    y = [1, 2, 3]
    z = [1, 2, 3]

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    print(bellymodel(x, y, z, 4, 5, 6, 7, 8))







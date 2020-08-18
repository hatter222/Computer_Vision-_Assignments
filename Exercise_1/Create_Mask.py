import numpy as np
from Ransac_Plane import ransac
import matplotlib.pyplot as plt
from scipy import ndimage


def create_mask(init_data, data_floor, cloud, type):
    rows, cols, channel = cloud.shape
#****************************************************** Setting Parameters for FLOOR **************************************************************
    if type == 'Floor' or type == 'floor' or type == 'floor_2':
        data_for_mask = np.copy(init_data)
        threshold = 0.04
        Max_iteration = 80
        kernel = np.ones((11, 11))
#****************************************************Setting Parameters for TOP**********************************************************************
    if type == 'Top' or type == 'top':
        init_for_top = np.empty((channel, (rows * cols)))
        x_cloud = init_data[:, :, 0]
        y_cloud = init_data[:, :, 1]
        z_cloud = init_data[:, :, 2]
        init_for_top[0, :] = x_cloud.flatten()
        init_for_top[1, :] = y_cloud.flatten()
        init_for_top[2, :] = z_cloud.flatten()
        temp = np.zeros(init_for_top.shape, dtype=bool)
        # print(temp.shape)
       # for i in range(0, init_for_top.shape[0]):
        #    for j in range(0, init_for_top.shape[1]):
         #       if init_for_top[i, j] == 1.0:
          #          temp[i, j] = True
        temp[:]=init_for_top==1.0
        # print(temp)
        init_for_top = np.multiply(data_floor, ~temp)
        threshold = 0.004
        Max_iteration = 80
        data_for_mask = np.copy(init_for_top)
        kernel = np.ones((9, 9))
#******************************************************** Calling RANSAC Fucntion ***************************************************************************

    best_inliers, best_coeffd, best_mask_floor, best_normal = ransac(data_for_mask, Max_iteration, threshold, type)
    print('number of inliers = ' + type + ": ", best_inliers)
#*************************************************** Data Reshaping and Morphologial operation **************************************************************
    mask = np.copy(cloud)
    filtered_mask = np.copy(cloud)
    x_plane = np.reshape(best_mask_floor[0, :], (rows, cols))
    y_plane = np.reshape(best_mask_floor[1, :], (rows, cols))
    z_plane = np.reshape(best_mask_floor[2, :], (rows, cols))

    if type == 'floor_2':
        # to get the binary mask
        x_plane = np.abs(x_plane) <= threshold
        y_plane = np.abs(y_plane) <= threshold
        z_plane = np.abs(z_plane) <= threshold

    mask[:, :, 0] = np.copy(x_plane)
    mask[:, :, 1] = np.copy(y_plane)
    mask[:, :, 2] = np.copy(z_plane)

    x_plane = ndimage.morphology.binary_closing(x_plane, kernel)
    y_plane = ndimage.morphology.binary_closing(y_plane, kernel)
    z_plane = ndimage.morphology.binary_closing(z_plane, kernel)
    filtered_mask[:, :, 0] = np.copy(x_plane)
    filtered_mask[:, :, 1] = np.copy(y_plane)
    filtered_mask[:, :, 2] = np.copy(z_plane)


    if type == 'Top' or type == 'top':
        return filtered_mask, best_normal, best_coeffd,mask
    else:
        return filtered_mask,mask

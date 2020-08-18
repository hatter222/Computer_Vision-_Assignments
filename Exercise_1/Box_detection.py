import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import scipy.io as sio
import cv2
from scipy.spatial import distance as dist
from Create_Mask import create_mask
from Dimension_detection import detect_corners
from scipy.spatial.distance import cdist
from Height_of_the_box import find_height

def boxdetection(cloud_path,data):
    cloud = data[cloud_path]
    # print (cloud.shape)

    rows, cols, channel = cloud.shape
    # print(rows)
    # print(cols)
    # print(channel)

    #*******************************************************Median filtering of cloud***************************************************************************

    x_cloud = ndimage.median_filter(cloud[:, :, 0], 11)
    y_cloud = ndimage.median_filter(cloud[:, :, 1], 11)
    z_cloud = ndimage.median_filter(cloud[:, :, 2], 11)
    data_cloud = np.dstack([x_cloud, y_cloud, z_cloud])  # dint work

    temp = np.empty_like(data_cloud)
    temp[:, :, 0] = np.copy(x_cloud)
    temp[:, :, 1] = np.copy(y_cloud)
    temp[:, :, 2] = np.copy(z_cloud)
    np.copyto(data_cloud, temp)


    '''
    b = np.reshape(cloud, (217088, 3))
    temp = np.empty_like(b)
    temp[:, 0] = x_cloud.flatten()
    temp[:, 1] = y_cloud.flatten()
    temp[:, 2] = z_cloud.flatten()

    # print(temp.shape)

    x = b[:, 0]  # subsample x = b[::16, 0]
    y = b[:, 1]  # subsample y = b[::16, 0]
    z = b[:, 2]  # subsample z = b[::16, 0]'''

    ''''# ignoring the points which the z-component of a vector is 0
    list_del = [i for i, v in enumerate(z) if v == 0]
    z = [z[i] for i in range(len(z)) if (i not in list_del)]
    x = [x[i] for i in range(len(x)) if (i not in list_del)]
    y = [y[i] for i in range(len(y)) if (i not in list_del)]

    v, t = x.shape

    ttt = np.empty((v, 3))
    ttt[:, 0] = np.copy(x)
    ttt[:, 1] = np.copy(y)
    ttt[:, 2] = np.copy(z)

    print(ttt.shape)'''
#****************************************************** DATA MANIPULATION**********************************************************
    data_floor = np.empty((channel, (rows * cols)))
    data_floor[0, :] = x_cloud.flatten()
    data_floor[1, :] = y_cloud.flatten()
    data_floor[2, :] = z_cloud.flatten()
    # np.copyto(init_for_top,init_data)
    # print(data_floor.shape)

#***************************************************** COMPUTING INLIERS ***********************************************************
    print("Computing Inliers by RANSAC ----------------------")
    floor_mask_2,uf_floor_mask = create_mask(data_floor, None, cloud, 'floor')
    # floor_mask_2 = create_mask(data_floor,None,cloud,'floor_2')
    top_mask, normal, coeff,uf_top_mask = create_mask(floor_mask_2, data_floor, cloud, 'top')
# ************************************************************* CONNECTED COMPONENT ***************************************************************************
    temp_label, num_label = ndimage.label(top_mask[:, :, 0])
    max_label = np.amax(temp_label)
    idx = np.empty_like(temp_label, dtype=bool)
    # print(temp_label.shape)
    #for i in range(0, temp_label.shape[0]):
     #   for j in range(0, temp_label.shape[1]):
      #      if temp_label[i, j] == max_label:
       #         idx[i, j] = True
    idx[:]= temp_label==max_label
    box_mask = np.copy(floor_mask_2)
    box_mask[idx, 0] = 0.5
    box_mask[idx, 1] = 0.5
    box_mask[idx, 2] = 0.5



#******************************************************** Detection of Dimensions ***************************************************************************
    edge,grey,box =  detect_corners(top_mask, data_cloud)

#********************************************************* Detect the Height********************************************************************************
    x_temp = np.copy(box_mask[:, :, 0])
    x_temp[box_mask[:, :, 0] == 0] = 1
    x_temp[box_mask[:, :, 0] == 1] = 0

    # x =find_height(data_cloud,floor_mask,normal,coeff)
    # x = sp.distance.cdist(top_mask,floor_mask,metric='euclidean')
    num = rows + cols
    height = np.linalg.norm(top_mask[:, :, 0] - x_temp) / num
    # print("height:", height)

    y_temp = np.copy(box_mask[:, :, 1])
    y_temp[box_mask[:, :, 1] == 0] = 1
    y_temp[box_mask[:, :, 1] == 1] = 0

    height2 = np.linalg.norm(top_mask[:, :, 1] - y_temp) / num
    # print("height:", height2)

    z_temp = np.copy(box_mask[:, :, 2])
    z_temp[box_mask[:, :, 2] == 0] = 1
    z_temp[box_mask[:, :, 2] == 1] = 0

    height3 = np.linalg.norm(top_mask[:, :, 2] - y_temp) / num
    # print("height:", height3)

    final_height = (height + height2 + height3) / 3.0
    print("height _by_first_method_euclidean:", final_height)

    #find_height(data_cloud, floor_mask_2, normal, coeff)

    x_dist = dist.cdist(top_mask[:, :, 0], x_temp)
    y_dist = dist.cdist(top_mask[:, :, 1], y_temp)
    z_dist = dist.cdist(top_mask[:, :, 2], z_temp)

    height_3 = (np.average(x_dist) + np.average(y_dist) + np.average(z_dist)) / 3
    print("height_by_2nd_method :", height3)

#************************************************** Plotting all the Results ***************************************************************************
    fig = plt.figure()
    plt1 = fig.add_subplot(121)
    plt1.imshow(cloud)
    plt.title("original cloud")
    plt2 = fig.add_subplot(122)
    plt2.imshow(data_cloud)
    plt.title(" Median Filtered cloud")

    fig2 = plt.figure()
    plt3 = fig2.add_subplot(121)
    plt3.imshow(uf_floor_mask)
    plt.title('Original Floor Mask')
    plt4 = fig2.add_subplot(122)
    plt4.imshow(floor_mask_2)
    plt.title("Filtered Floor Mask")

    fig3 = plt.figure()
    plt5 = fig3.add_subplot(121)
    plt5.imshow(uf_top_mask)
    plt.title('Original Top  Mask')
    plt6 = fig3.add_subplot(122)
    plt6.imshow(top_mask)
    plt.title("Filtered Top Mask")

    fig4 = plt.figure()
    plt7 = fig4.add_subplot(121)
    plt7.imshow(box_mask)
    plt.title('box with both top and floor plane')
    plt8 = fig4.add_subplot(122)
    plt8.imshow(edge)
    plt.title("Top Plane with edges")

    fig5 = plt.figure()
    plt9 = fig5.add_subplot(121)
    plt9.imshow(grey)
    plt.title('Top Plane with detected corners')
    plt10 = fig5.add_subplot(122)
    plt10.imshow(box)
    plt.title('Cloud with detected Cornera')
    plt.show()
#***************************__START OF THE PROGRAM ___**********************************************************************************************************
i=1
s = str(i)
path_1 = 'data1/example'+s+'kinect.mat'
cloud_path_1 = 'cloud'+s
print("For the Data: ", cloud_path_1, " in example :", path_1)
data_1 = sio.loadmat(path_1)
# a = sio.whosmat('data1/example4kinect.mat')
# print(a)
boxdetection(cloud_path_1, data_1)





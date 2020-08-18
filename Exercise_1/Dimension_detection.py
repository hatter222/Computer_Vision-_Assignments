from skimage import feature
from skimage import morphology as morph
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage,misc
import numpy as np
import  scipy.spatial.distance as dist
import  scipy
import math

import PIL.Image as pilim
def detect_corners(top_mask, data_cloud):
    edges = feature.canny(top_mask[:, :, 0], sigma=5)
    rows,cols,channels = top_mask.shape
    filtered_edge = scipy.ndimage.morphology.binary_dilation(edges)
    gray = np.float32(filtered_edge)
    box = np.float32(data_cloud)
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.2, 100)
    #print(corners)
    #print(len(corners))
    for corner in corners:
        x, y = corner.ravel()
        #print(x, y)
        cv2.circle(gray, (x, y), 5, (36, 255, 12), -1)
        cv2.circle(box,(x,y),5,(36, 255, 12), -1)

    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    top_l, bottom_l, bottom_r,topr = corners[0], corners[1], corners[2],corners[3]  # corners[3]

    width_A = math.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = math.sqrt(((bottom_l[0] - top_l[0]) ** 2) + ((bottom_l[1] - top_l[1]) ** 2))
    width_C = math.sqrt(((bottom_r[0] - top_l[0]) ** 2) + ((bottom_r[1] - top_l[1]) ** 2))
    distance = np.sort([width_A, width_B, width_C])
    print('Pixels range calculation of Dimensions-------------------------------------')
    print('Width of the box (in pixels): = ', distance[0])
    print('Length of the box (in pixels): = ', distance[1])

    #x = dist.euclidean(bottom_l,bottom_r)
    #y= dist.euclidean(bottom_r,top_l)
    #z = dist.euclidean(top_l,bottom_l)
    #print(x,y,z)
    plt.show()
    c1= data_cloud[int(top_l[0]),int(top_l[1]),:]
    c4= data_cloud[int(bottom_r[0]),int(bottom_r[1]),:]
    c2= data_cloud[int(bottom_l[0]),int(bottom_l[1]),:]
    c3 = data_cloud[int(topr[0]),int(topr[1]),:]
   # print(c1,c2,c3,c4)

    l1 = (abs(np.linalg.norm(c1-c2))+abs(np.linalg.norm(c3-c4)))/2
    l2 = (abs(np.linalg.norm(c4-c2))+abs(np.linalg.norm(c1-c4)))/2
    l3 = (abs(np.linalg.norm(c1-c3))+abs(np.linalg.norm(c2-c4)))/2
    l4 = (abs(np.linalg.norm(c4-c1))+abs(np.linalg.norm(c2-c3)))/2

    dim = np.sort([l1,l2,l3,l4])

    print("In values Range Calculation of Dimensions--------------------------------------------")
    print("------Average of the two lowest dimensions------")
    print("Width :", (dim[0]+dim[1])/2)
    print("Length:",(dim[2] + dim[3]) / 2,"\n")
    print("------considering first two Values as width and length------")
    print("Width:", (dim[0]))
    print("Length:", (dim[1]),"\n")

    return filtered_edge,gray,box

    ''' for i in range(3):
       ver1 = abs(box[int(top_l[0]),int(top_l[1]),i])
       ver2 = abs(box[int(bottom_l[0]),int(bottom_l[1]),i])
       ver3 = abs(box[int(bottom_r[0]),int(bottom_r[1]),i])
    val = np.cumsum(ver1)/3
    val2 = np.cumsum(ver2)/3
    val3 = np.cumsum(ver3)/3
    sub1 = abs(val-val2)*10
    sub2 = abs(val2-val3)*10
    sub3 = abs(val3-val)*10
    dist_2 = np.sort([sub1,sub2,sub3])
    print(dist_2)
    print("Some trial and error method to calculate metric:\n  from pixel values to mm ")
    ratio = (distance[1]/distance[0]/1000)
    print("ratio of leng/wid:", ratio)  # orig length and width = 45,35 - ratio = 1.3

    print('Width of the box : = ', distance[0]*ratio)
    print('Length of the box : = ', distance[1]*ratio)'''




'''   edges = feature.canny(top_mask[:, :, 0], sigma=5)
    filtered_edge = ndimage.morphology.binary_dilation(edges)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.imshow(filtered_edge)
    plt.title('top plane edges')
    plt.show()

    image = 255*(filtered_edge.astype(np.uint8))
    gray = np.float32(filtered_edge)
    corner = cv2.cornerHarris(gray, 17, 5, 0.07)
    dest = cv2.dilate(corner, None)
    filtered_edge[dest > 0.01 * dest.max()] = 1

    filtered_edge1 = scipy.ndimage.morphology.binary_opening(filtered_edge, structure=np.ones((3, 3)))
    filtered_edge = scipy.ndimage.morphology.binary_erosion(filtered_edge1, structure=np.ones((5, 5)))


    indices = filtered_edge == 1
    top_mask[indices, 0] = 1
    ax1 = fig.add_subplot(1, 2, 2)
    plt.imshow(top_mask)
    plt.title('box plane with corners')
    plt.show()
    print(top_mask)
'''
'''    ret, thresh = cv2.threshold(image,127,255,1)
    contours, h = cv2.findContours(thresh,1,2)
    for c in contours:
        approx = cv2.approxPolyDP(contours,0.01*cv2.arcLength(contours,True),True)
        print(len(approx))
'''
'''
    lines = cv2.HoughLines(skeleton,1,np.pi/180,60)
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(top_mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

    plt.imshow(top_mask)
    plt.show()
    gray = np.float32(filtered_edge)
    corner = cv2.cornerHarris(gray, 17, 5, 0.07)
    dest = cv2.dilate(corner, None)


    print(corner.shape)
    filtered_edge[dest > 0.01 * dest.max()] = 1

    filtered_edge1 = ndimage.morphology.binary_opening(filtered_edge, structure=np.ones((3, 3)))
    filtered_edge =ndimage.morphology.binary_erosion(filtered_edge1, structure=np.ones((5, 5)))

    indices = filtered_edge == 1
    top_mask[indices, 0] = 1
    ax1 = fig.add_subplot(1, 2, 2)
    plt.imshow(corner)
    plt.title('box plane with corners')
    plt.show()



'''
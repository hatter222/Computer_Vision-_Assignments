import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage
#import pydicom
from dicom import ReadableDicomDataset
from skimage import color
from skimage import filters
from PIL import  Image
from imageio import imread, imwrite
from PIL import Image
import random

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

obj1 = ReadableDicomDataset("Data\data_2\level-0-frames-0-19278.dcm")
size= obj1.geometry_imsize
print(size)

loc0 = ((int)(size[0][0]/2) , (int)(size[0][1]/2))
#loc1 = ((int)(size[1][0]/2) , (int)(size[1][1]/2))
#loc2 = ((int)(size[2][0]/2) , (int)(size[2][1]/2))
print(loc0)
#loc3 = ((int)(size[3][0]/2) , (int)(size[3][1]/2))
tissue_image0 = obj1.read_region((0,0),0,size[2],1)
tissue_image1 = obj1.read_region((0,0),1,size[2],1)
tissue_image2 = obj1.read_region((0,0),2,size[2],1)
tissue_image3 = obj1.read_region((0,0),2,size[2],1)
plt.imshow(tissue_image0)
plt.show()
plt.imshow(tissue_image1)
plt.show()
plt.imshow(tissue_image2)
plt.show()
plt.imshow(tissue_image3)
plt.show()

# #imagepatches
image_0 = obj1.read_region(loc0,0,(400,300),1)
image_0=cv2.cvtColor(np.array(image_0), cv2.COLOR_RGBA2RGB)
# image_1 = obj1.read_region(loc0,1,(400,300),1)
# image_2 = obj1.read_region(loc0,2,(400,300),1)
# image_3 = obj1.read_region(loc0,3,(400,300),1)
plt.imshow(image_0)
plt.show()
# plt.imshow(image_1)
# plt.show()
# plt.imshow(image_2)
# plt.show()
# plt.imshow(image_3)

# plt.show()
# #plt.imshow(image_1)

# #plt.imshow(image_3)
# Imagetile = obj1.get_tile(1,2)
# plt.imshow(Imagetile)
# plt.show()

# row = obj1.geometry_rows
# cols = obj1.geometry_columns
# print([row,cols])

# #otsu method to find masking
# '''image = Image.fromarray(tissue_image)
# blur = color.rgb2gray(image)
# blur = filters.gaussian(blur,5)
# val = filters.threshold_otsu(blur)
# mask = blur > val
# plt.imshow(mask, cmap='gray',interpolation= 'nearest')
# plt.show()'''

opencvImage0 = cv2.cvtColor(np.array(tissue_image0), cv2.COLOR_RGB2BGR)
opencvImage1 = cv2.cvtColor(np.array(tissue_image1), cv2.COLOR_RGB2BGR)
opencvImage2 = cv2.cvtColor(np.array(tissue_image2), cv2.COLOR_RGB2BGR)
opencvImage3 = cv2.cvtColor(np.array(tissue_image3), cv2.COLOR_RGB2BGR)

gray0 = cv2.cvtColor(opencvImage0, cv2.COLOR_BGR2GRAY)
gray1 = cv2.cvtColor(opencvImage1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(opencvImage2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(opencvImage3, cv2.COLOR_BGR2GRAY)
ret0,mask0 = cv2.threshold(gray0,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret1,mask1 = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret2,mask2 = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3,mask3 = cv2.threshold(gray3,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


# # dilate
# #dil = cv2.dilate(mask, kernel = np.ones((7,7),np.uint8))
# # erode
# #activeMap = cv2.erode(dil, kernel = np.ones((7,7),np.uint8))

mask0 = 255 - mask0
mask1 = 255 - mask1
mask2 = 255 - mask2
mask3 = 255 - mask3



# #imwrite('mask.png', mask)
plt.imshow(mask0, cmap='gray',interpolation= 'nearest')
plt.show() 
plt.imshow(mask1, cmap='gray',interpolation= 'nearest')
plt.show() 
plt.imshow(mask2, cmap='gray',interpolation= 'nearest')
plt.show() 
plt.imshow(mask3, cmap='gray',interpolation= 'nearest')
plt.show() 




#  #10 random patches
img_list0=[]
img_list1=[]
img_list2=[]
img_list3=[]
ds0=4
ds1=8
ds2=16
ds3=32
i=0
j=0
k=0
while (i <10 and j<10):
       x_ds0=random.randint(0, (int)(size[0][0]/ds0))
       y_ds0=random.randint(0, (int)(size[0][1]/ds0))
       
       x_ds1=random.randint(0, (int)(size[0][0]/ds1))
       y_ds1=random.randint(0, (int)(size[0][1]/ds1))
       
       x_ds2=random.randint(0, (int)(size[0][0]/ds2))
       y_ds2=random.randint(0, (int)(size[0][1]/ds2))
       
       x_ds3=random.randint(0, (int)(size[0][0]/ds3))
       y_ds3=random.randint(0, (int)(size[0][1]/ds3))
       
#      loc_0=((random.randint(0, size[0][0]/ds0)),(random.randint(0, size[0][1]/ds0)))
#      loc_1=((random.randint(0, size[0][0]/ds1)),(random.randint(0, size[0][1]/ds1)))
#      loc_2=((random.randint(0, size[0][0]/ds2)),(random.randint(0, size[0][1]/ds2)))
#      loc_3=((random.randint(0, size[0][0]/ds3)),(random.randint(0, size[0][1]/ds3)))
       step_ds0 = int(1024/ds0)
       step_ds1 = int(1024/ds1)
       step_ds2 = int(1024/ds2)
       step_ds3 = int(1024/ds3)
       
       canbeused0 = np.sum((mask0[y_ds0:y_ds0+step_ds0,x_ds0:x_ds0+step_ds0])>1)>0.95*step_ds0*step_ds0
       canbeused1 = np.sum((mask1[y_ds1:y_ds1+step_ds1,x_ds1:x_ds1+step_ds1])>1)>0.95*step_ds1*step_ds1
       canbeused2 = np.sum((mask2[y_ds2:y_ds2+step_ds2,x_ds2:x_ds1+step_ds2])>1)>0.95*step_ds2*step_ds2
       canbeused3 = np.sum((mask3[y_ds3:y_ds3+step_ds3,x_ds3:x_ds3+step_ds3])>1)>0.95*step_ds3*step_ds3
       
       
       if (canbeused2):
           i=i+1
           x=int(x_ds2*ds2)
           y=int(y_ds2*ds2)
           image_2 = obj1.read_region((x,y),0,(400,300),1)
           img_list2.append(image_2)
       if (canbeused2):
           j=j+1
           x=int(x_ds3*ds3)
           y=int(y_ds3*ds3)
           image_3 = obj1.read_region((x,y),0,(400,300),1)
           img_list2.append(image_3)
       
       # if (canbeused1):
       #     j=j+1
       #     x=int(x_ds1*ds1)
       #     y=int(y_ds1*ds1)
       #     image_1 = obj1.read_region((x,y),0,(400,300),1)
       #     img_list1.append(image_1)
#      image_1 = obj1.read_region(loc_i,1,(400,300),1)
#      image_2 = obj1.read_region(loc_i,2,(400,300),1)
#      image_3 = obj1.read_region(loc_i,3,(400,300),1)
#     image=[image_0,image_1, image_2, image_3]
#     img_list.append(image)
#print(i)
def RGB_to_OD(I):
    """
    Convert from RGB to optical density
    :param I:
    :return:
    """
    I = remove_zeros(I)
    return -1 * np.log(I / 255)

def remove_zeros(I):
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    :return:
    """
    mask1 = (I == 0)
    I[mask1] = 1
    return I

def OD_to_RGB(OD):
    """
    Convert from optical density to RGB
    :param OD:
    :return:
    """
    return (255 * np.exp(-1 * OD)).astype(np.uint8)
def normalize_rows(A):
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=1)[:, None]


im=np.stack(img_list2).reshape(-1,3)
OD = RGB_to_OD(im)#.reshape((-1, 3))
OD = (OD[(OD > 0.15).any(axis=1), :])

# # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
_, V = np.linalg.eigh(np.cov(OD, rowvar=False))
# # The two principle eigenvectors
V = V[:, [2, 1]]

# # Make sure vectors are pointing the right way
if V[0, 0] < 0: V[:, 0] *= -1
if V[0, 1] < 0: V[:, 1] *= -1

# Project on this basis.
That = np.dot(OD, V)


##ex5
phi = np.arctan2(That[:, 1], That[:, 0])
minPhi = np.percentile(phi, 1)
maxPhi = np.percentile(phi, 99)
v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
if v1[0] > v2[0]:
    HE = np.array([v1, v2])
else:
    HE = np.array([v2, v1])

# take random sample for plotting
randsamp = np.array(random.sample(OD.tolist(),1000))


# plot points (you can uncomment but then will be some pblem with the next plotting)


fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
# # color needs to be given as RGBA values in range 0..1 for matplotlib
color=np.ones((1000,4),np.float32)
color[:,0:3] = OD_to_RGB(randsamp)/255.
ax.scatter(randsamp[:,0],randsamp[:,1],randsamp[:,2], c=color)
ax.plot([0, HE[0,0]],[0, HE[0,1]],[0, HE[0,2]], linewidth=4, color=(OD_to_RGB(HE[0,:])/255.).tolist()+[1.])
ax.plot([0, HE[1,0]],[0, HE[1,1]],[0, HE[1,2]], linewidth=4, color=(OD_to_RGB(V[1,:])/255.).tolist()+[1.])
ax.set_xlabel('red (OD)')
ax.set_ylabel('green (OD)')
ax.set_zlabel('blue (OD)')
plt.title('stain vector estimation')
plt.show()

#ex5.4
def get_stain_matrix(I):
#     """
#     Get stain matrix (2x3)
#     
     OD = RGB_to_OD(I).reshape((-1, 3))
     OD = (OD[(OD > 0.15).any(axis=1), :])
     _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
     V = V[:, [2, 1]]
     if V[0, 0] < 0: V[:, 0] *= -1
     if V[0, 1] < 0: V[:, 1] *= -1
     That = np.dot(OD, V)
     phi = np.arctan2(That[:, 1], That[:, 0])
     minPhi = np.percentile(phi, 1)
     maxPhi = np.percentile(phi, 99)
     v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
     v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
     if v1[0] > v2[0]:
         HE = np.array([v1, v2])
     else:
         HE = np.array([v2, v1])
     return HE
h,w,c = np.array(image_0).shape
OD = RGB_to_OD(np.array(image_0)).reshape((-1, 3))
D=get_stain_matrix(np.array(image_0))
concentrations= np.linalg.lstsq(D.T,OD.T,rcond=-1)[0]                                           #spams.lasso(OD.T, D=D.T, mode=2, lambda1=0.01, pos=True).toarray().T

print(concentrations.shape)

concentrations = concentrations.T
H = concentrations[:, 0].reshape(h, w)
Eosin=concentrations[:, 1].reshape(h, w)

#H = (255*np.exp(-1 * H)).astype(np.uint8)
#Eosin=(255*np.exp(-1 * Eosin)).astype(np.uint8)
plt.imshow(H,interpolation= 'nearest')
plt.show()

plt.imshow(Eosin,interpolation= 'nearest')
plt.show() 


#ex.5.5
target_stain_matrix=np.array([[0.75, 0.75, 0.5],[0.05, 0.75, 0.5]])
max_C_target=np.array([2.0,1.0]).reshape((1, 2))
maxC_source = np.percentile(concentrations, 99, axis=0).reshape((1, 2))
concentrations *= (max_C_target / maxC_source)
tmp = 255 * np.exp(-1 * np.dot(concentrations, target_stain_matrix))
img=tmp.reshape(np.array(image_0).shape).astype(np.uint8)
# How to plot results?
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2RGBA))
plt.imshow(img, interpolation= 'nearest')
plt.show()


fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
# color needs to be given as RGBA values in range 0..1 for matplotlib
color=np.ones((1000,4),np.float32)
color[:,0:3] = OD_to_RGB(randsamp)/255.
ax.scatter(randsamp[:,0],randsamp[:,1],randsamp[:,2], c=color)
ax.plot([0, HE[0,0]],[0, HE[0,1]],[0, HE[0,2]], linewidth=4, color=(OD_to_RGB(HE[0,:])/255.).tolist()+[1.])
ax.plot([0, HE[1,0]],[0, HE[1,1]],[0, HE[1,2]], linewidth=4, color=(OD_to_RGB(V[1,:])/255.).tolist()+[1.])
ax.set_xlabel('red (OD)')
ax.set_ylabel('green (OD)')
ax.set_zlabel('blue (OD)')
plt.title('stain vector estimation')
#plt.show()
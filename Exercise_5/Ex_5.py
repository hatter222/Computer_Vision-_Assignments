import matplotlib.pyplot as plt
import numpy as np
import cv2
from dicom import ReadableDicomDataset
import skimage
from skimage import color
from skimage import filters
from PIL import  Image

obj1 = ReadableDicomDataset("Data\data_2\level-0-frames-0-19278.dcm")
size= obj1.geometry_imsize
print(size)

loc0 = ((int)(size[0][0]/2) , (int)(size[0][1]/2))
#loc1 = ((int)(size[1][0]/2) , (int)(size[1][1]/2))
#loc2 = ((int)(size[2][0]/2) , (int)(size[2][1]/2))
print(loc0)
#loc3 = ((int)(size[3][0]/2) , (int)(size[3][1]/2))

tissue_image = obj1.read_region((0,0),2,size[2],1)
plt.imshow(tissue_image)
plt.show()

#imagepatches
image_0 = obj1.read_region(loc0,0,(400,300),1)
image_1 = obj1.read_region(loc0,1,(400,300),1)
image_2 = obj1.read_region(loc0,2,(400,300),1)
image_3 = obj1.read_region(loc0,3,(400,300),1)
plt.imshow(image_0)
plt.show()
plt.imshow(image_1)
plt.show()
plt.imshow(image_2)
plt.show()
plt.imshow(image_3)

plt.show()
#plt.imshow(image_1)

#plt.imshow(image_3)
Imagetile = obj1.get_tile(1,2)
plt.imshow(Imagetile)
plt.show()

row = obj1.geometry_rows
cols = obj1.geometry_columns
print([row,cols])

#otsu method to find masking
'''image = Image.fromarray(tissue_image)
blur = color.rgb2gray(image)
blur = filters.gaussian(blur,5)
val = filters.threshold_otsu(blur)
mask = blur > val
plt.imshow(mask, cmap='gray',interpolation= 'nearest')
plt.show()'''

opencvImage = cv2.cvtColor(np.array(tissue_image), cv2.COLOR_RGB2BGR)

gray = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
ret2,mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
mask = 255 - mask
plt.imshow(mask, cmap='gray',interpolation= 'nearest')
plt.show()





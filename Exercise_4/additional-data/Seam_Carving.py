from PIL import Image
from scipy.signal import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import minimum_filter1d
import numpy as np

rgb = np.array(Image.open('kingfishers.jpg'))
gray = np.array(Image.open('kingfishers.jpg').convert('L'))
mask = np.array(Image.open('kingfishers-mask.png').convert('L'))

#4.1 to compute Energy function and Cost
def compute_energy_func(image,mask):
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    output = np.abs(convolve(image, laplacian, 'same'))
    result = np.sqrt(output) # for visualisation Purpose
    energy = result.astype('float')
    #adding mask to cost array
    energy = np.add(energy, mask)
    cost = energy.copy()
    #cumulative cost
    for i in range(1, energy.shape[0]):
        cost[i] = np.add(energy[i], minimum_filter1d(energy[i - 1], 3, mode='reflect'))
    return result , cost

def seam_carving_upsampling(rgb,gray,mask,iterations):
    for i in range(0, iterations):
        indices = compute_path(mask, gray)
        rgb, gray, mask = upsampleImage(rgb, gray, mask,indices)

    Image.fromarray(rgb).save('upsampled_rgb.jpg')
    plt.imshow(gray,'gray')
    plt.show()
    plt.imshow(rgb)
    plt.show()


def compute_path(mask, gray):
    energy, cost = compute_energy_func(gray,mask)
    indices = []
    #to compute from botton to top
    array_temp = cost[-1]
    for i in range(0, cost.shape[0]):
        idx = np.argmin(array_temp)
        if (len(indices) != 0):
            last_idx = indices[-1] # @first last idx =0
            idx = idx - 1
            new_idx = last_idx + idx #backtracking
            if new_idx < 0:
                new_idx = 0
            idx = new_idx
        indices.append(idx)
        if i < cost.shape[0] - 1:
            new_row = cost[-(i + 1)]
            if (idx - 1 > -1):
                start = idx - 1# @ first start 0
            else: start = 0
            if (idx + 2 <= cost.shape[1]):
                end = idx + 2  #@first end 2
            else:
                end = cost.shape[1]
            array_temp = new_row[start:end] #tracking array

    indices = list(reversed(indices))
    return indices


def upsampleImage(rgb, gray, mask, indices):
    #adding additional colm
    # updated image with additional pixels
    add_colm = np.zeros((mask.shape[0], 1)).astype('float')
    mask = np.append(mask, add_colm, 1)
    gray = np.append(gray, add_colm, 1)
    add_colm = np.zeros((mask.shape[0], 1, 3))
    rgb = np.append(rgb, add_colm, 1).astype(np.uint8)
    for i in range(0, mask.shape[0]): #1024 row
        idx = indices[i]
        end = mask.shape[1] #1537 cols
        mask[i, idx + 1:end] = mask[i, idx:end - 1]
        mask[i, idx] = 255. #to avoid distortion
        mask[i, idx + 1] = 255
        # insert the seam
        gray[i, idx + 1:end] = gray[i, idx:end - 1]
        rgb[i, idx + 1:end] = rgb[i, idx:end - 1]

        #plt.imshow(gray, 'gray')
        #plt.show()
        #plt.imshow(rgb)
        #plt.show()

    return rgb, gray, mask

def delete_path(rgb, gray, mask, indices):

    for i in range(0, gray.shape[0]):
        idx = indices[i]
        if idx != gray.shape[1]:
            end = gray.shape[1]
            mask[i, idx:end-1] = mask[i, idx+1:end]
            gray[i, idx:end-1] = gray[i, idx+1:end]
            rgb[i, idx:end-1] = rgb[i, idx+1:end]

    mask = np.delete(mask, -1 , 1)
    gray = np.delete(gray, -1 , 1)
    rgb = np.delete(rgb, -1 , 1)
    return rgb, gray, mask

def seam_carving_downsampling(rgb,gray,mask,iterations):
    for i in range (0,iterations):
        indices = compute_path(mask, gray)
        rgb, gray, mask = delete_path(rgb, gray, mask, indices)

    Image.fromarray(rgb).save('downsampled.jpg')
    plt.imshow(gray,'gray')
    plt.show()
    plt.imshow(rgb)
    plt.show()

seam_carving_downsampling(rgb,gray,mask,100)
seam_carving_upsampling(rgb,gray,mask,500)

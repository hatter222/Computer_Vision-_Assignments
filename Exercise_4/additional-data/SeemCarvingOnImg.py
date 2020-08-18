import numpy as np
import cv2
from scipy import signal as sg
from scipy.ndimage import minimum_filter1d
from PIL import Image
from tqdm import trange
from imageio import imread, imwrite


def calc_energy(img):
    filter_vectors= np.array([[0, 1, 0],

                            [1, -4,  1],

                            [0, 1, 0]])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_vectors = np.stack([filter_vectors] * 3, axis=2)
    img = img.astype('float32')
    
    convolved=np.abs(sg.convolve(img,filter_vectors,'same'))
    # We sum the energies in the red, green, and blue channels
    energy_map=np.sum(convolved, axis=2)
    # We sum the energies in the red, green, and blue channels
    #energy_map = convolved.sum(axis=2)

    return energy_map

def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    for i in trange(c - new_c): # use range if you don't want to use tqdm
        img = carve_column(img)

    return img

def crop_r(img, scale_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=np.bool)
    
    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i, j]
    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)
    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

img = Image.open('common-kestrel.jpg')  #np.array(Image.open('common_kestrel.jpg'))
out = crop_r(img, 0.8)
imwrite('output.png', out)

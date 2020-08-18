from PIL import Image
from scipy.signal import convolve
import matplotlib.pyplot as plt
import numpy as np

def compute_energy_func(image):
    laplacian = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    res = np.abs(convolve(image, laplacian, 'same'))
    res = np.sqrt(res)
    return res

rgb = np.array(Image.open('common-kestrel.jpg'))
gray = np.array(Image.open('common-kestrel.jpg').convert('L'))
mask = compute_energy_func(gray)
for i in range(0, 250):
    energy = compute_energy_func(gray)
    energy = np.add(energy, mask)
    sum = np.argmin(np.sum(energy, axis=1))
    energy = np.delete(energy, sum, 0)
    gray = np.delete(gray, sum, 0)
    rgb = np.delete(rgb, sum, 0)
    mask = np.delete(mask, sum, 0)
energy = np.rot90(energy)
gray = np.rot90(gray)
rgb = np.rot90(rgb)
mask = np.rot90(mask)
for i in range(0, 500):
    energy = compute_energy_func(gray)
    energy = np.add(energy, mask)
    sum = np.argmin(np.sum(energy, axis=1))
    energy = np.delete(energy, sum, 0)
    gray = np.delete(gray, sum, 0)
    rgb = np.delete(rgb, sum, 0)
    mask = np.delete(mask, sum, 0)
gray = np.rot90(gray, 3)
rgb = np.rot90(rgb, 3)

Image.fromarray(rgb).save('delete_row.jpg')
plt.imshow(gray,'gray')
plt.show()
plt.imshow(rgb)
plt.show()


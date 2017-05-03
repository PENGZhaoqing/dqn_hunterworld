import matplotlib.pyplot as plt
import cv2
from skimage.feature import hog
from skimage import data, color, exposure
import numpy as np

# image = color.rgb2gray(data.astronaut())

# plt.imshow(image,cmap='gray')
# plt.show()

# image = cv2.imread('crop001501.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('crop001501.png', cv2.IMREAD_COLOR)
# image = color.rgb2gray(image)

image = cv2.imread('crop001501.png', cv2.IMREAD_GRAYSCALE) / 255.0

fd, hog_image = hog(image, orientations=10, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()

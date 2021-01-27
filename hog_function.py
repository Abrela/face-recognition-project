import numpy as np
import cv2
from skimage.feature import hog

from skimage import exposure
import matplotlib.pyplot


def get_hog_vector(image):
    img = cv2.imread(image)
    img = np.float32(img) / 255.0

    fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

    # Rescale histogram for better display
    return fd


get_hog_vector("C:/Users/pawel/Desktop/zdjecie.pgm")

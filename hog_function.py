import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot


def get_hog_vector(image):
    img = cv2.imread(image)
    img = np.float32(img) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    fig, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)
    ax1.imshow(img, cmap=matplotlib.pyplot.cm.gray)
    ax1.set_title('Oryginal image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.imshow(hog_image_rescaled, cmap=matplotlib.pyplot.cm.gray)
    ax2.set_title('HOG of image')

    matplotlib.pyplot.show()
    return fd



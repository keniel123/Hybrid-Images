import math
import numpy as np

from MyConvolution import convolve


def myHybridImages(lowImage: np.ndarray, lowSigma: float, highImage: np.ndarray, highSigma: float) -> np.ndarray:
    """
    Create hybrid images by combining a low-pass and high-pass filtered pair.
        :param lowImage: the image to low-pass filter (either greyscale shape=(rows,cols) or colour
    shape=(rows,cols,channels))
        :type numpy.ndarray
    :param lowSigma: the standard deviation of the Gaussian used for low-pass filtering lowImage :type float
    :param highImage: the image to high-pass filter (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
        :type numpy.ndarray
        :param highSigma: the standard deviation of the Gaussian used for low-pass filtering highImage
    before subtraction to create the high-pass filtered image
    :type float
    :returns returns the hybrid image created
    by low-pass filtering lowImage with a Gaussian of s.d. lowSigma and combining it with
    a high-pass image created by subtracting highImage from highImage convolved with
    a Gaussian of s.d. highSigma. The resultant image has the same size as the input images.
        :rtype numpy.ndarray
    """
    print(lowImage.shape)
    print(highImage.shape)

    if lowImage.shape[:2] != highImage.shape[:2]:
        new_height = min(lowImage.shape[0], highImage.shape[0])
        new_width = min(lowImage.shape[1], highImage.shape[1])
        lowImage = center_crop(lowImage, new_width,new_height )
        highImage = center_crop(highImage, new_width, new_height)

    # All images are normalised
    lowfImage = convolve(lowImage / 255, makeGaussianKernel(lowSigma))
    highfImage = (highImage / 255) - convolve(highImage / 255, makeGaussianKernel(highSigma))

    return (lowfImage + highfImage) * 255  # rescale image to range 0-255


def gaussian_kernel(sigma):
    size = int(8.0 * sigma + 1.0)
    if size % 2 == 0:
        size += 1
    mu = np.floor([size / 2, size / 2])
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-(0.5 / (sigma * sigma)) * (np.square(i - mu[0]) +
                                                              np.square(j - mu[0]))) / np.sqrt(
                2 * math.pi * sigma * sigma)

    kernel = kernel / np.sum(kernel)
    return kernel


def makeGaussianKernel(sigma: float) -> np.ndarray:
    """
    Use this function to create a 2D gaussian kernel with standard deviation sigma.
    The kernel values should sum to 1.0,
    and the size should be floor(8*sigma+1)
    or floor(8*sigma+1)+1 (whichever is odd) as per the assignment specification.
    """
    # Your code here.
    # size 对应卷积kernel大小

    size = int(8.0 * sigma + 1.0)  # (this implies the window is +/- 4 sigmas from the centre of the Gaussian)

    if (size % 2 == 0):
        # size must be odd
        size = size + 1
    kernel_size = size

    kernel = np.zeros([kernel_size, kernel_size])
    center_kernel = kernel_size // 2
    s = 2 * (sigma ** 2)

    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center_kernel
            y = j - center_kernel
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]

    sum_val = 1 / sum_val
    kernel_res = kernel * sum_val

    return kernel_res


def center_crop(img, new_width=None, new_height=None):
    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]


    return center_cropped_img

import numpy as np


def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve an image with a kernel assuming zero-padding of the image to handle the borders
    :param image: the image (either greyscale shape=(rows,cols) or colour shape=(rows,cols,channels))
    :type numpy.ndarray
    :param kernel: the kernel (shape=(kheight,kwidth); both dimensions odd)
    :type numpy.ndarray
    :returns the convolved image (of the same shape as the input image)
    :rtype numpy.ndarray
    """
    # Your code here. You'll need to vectorise your implementation to ensure it runs
    # at a reasonable speed.

    # convoluted image
    convolved = np.zeros(image.shape)
    kernel = np.flip(kernel)  # flip kernel
    # image dimension
    row, col = image.shape[:2]

    # template dimension
    trow, tcol = kernel.shape

    # half of template row and col
    trhalf, tchalf = trow // 2, tcol // 2

    # Test if kernel of odd shape
    if (trow % 2) == 0 or (tcol % 2) == 0:
        print(trow, tcol)
        print("The kernel is not of odd shape")

    if len(image.shape) == 2:
        channels = 1  # grayscale
        # tc_image = np.pad(image, ((trhalf, trhalf), (tchalf, tchalf)))
        padding_im = np.zeros([row + 2 * trhalf, col + 2 * tchalf])
        padding_im[trhalf:-trhalf, tchalf:-tchalf] = image

    else:
        channels = 3  # rgb
        # tc_image = np.pad(image, ((trhalf, trhalf), (tchalf, tchalf), (0, 0)))
        padding_im = np.zeros([row + 2 * trhalf, col + 2 * tchalf, 3])
        padding_im[trhalf:-trhalf, tchalf:-tchalf, 0:] = image

    for i in range(0, channels):
        for c in range(0, col):
            for r in range(0, row):
                if channels == 1:
                    convolved[r, c] = (kernel * padding_im[r:r + trow, c:c + tcol]).sum()
                else:
                    convolved[r, c, i] = (kernel * padding_im[r:r + trow, c:c + tcol, i]).sum()

    return convolved



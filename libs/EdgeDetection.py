import cv2
import numpy as np
from scipy.signal import convolve2d

def apply_kernel(image: np.ndarray, horizontal_kernel: np.ndarray, vertical_kernel: np.ndarray):
    """
        Convert image to gray scale and convolve with kernels
        :param image: Image to apply kernel to
        :param horizontal_kernel: The horizontal array of the kernel
        :param vertical_kernel: The vertical array of the kernel
        :return: The result of convolution
    """
    # convert to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # convolution
    horizontal_edge = convolve2d(gray, horizontal_kernel)
    vertical_edge = convolve2d(gray, vertical_kernel)

    mag = np.sqrt(pow(horizontal_edge, 2.0) + pow(vertical_edge, 2.0))

    return mag



def prewitt_edge(image: np.ndarray):
    """
        Apply Prewitt Operator to detect edges
        :param image: Image to detect edges in
        :return: edges image
    """
    # define filters
    vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    horizontal = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    mag = apply_kernel(image, horizontal, vertical)

    return mag

def sobel_edge(image: np.ndarray):
    """
        Apply Sobel Operator to detect edges
        :param image: Image to detect edges in
        :return: edges image
    """
    # define filters
    vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    mag = apply_kernel(image, horizontal, vertical)

    return mag

def roberts_edge(image: np.ndarray):
    """
        Apply Roberts Operator to detect edges
        :param image: Image to detect edges in
        :return: edges image
    """
    # define filters
    vertical = np.array([[0, 1], [-1, 0]])
    horizontal = np.array([[1, 0], [0, -1]])

    mag = apply_kernel(image, horizontal, vertical)

    return mag
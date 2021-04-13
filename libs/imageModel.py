import cv2
import numpy as np
from libs import Noise, LowPass
from libs import Histogram
from libs import FFilters
from libs import EdgeDetection

def add_noise(data: np.array, type: str, snr: float = 0.5, sigma: int = 64) -> np.ndarray:
    """
    This function adds different types of noises to the given image
    :param type: Specify the type of noise to be added
    :return: numpy array of the new noisy image
    """

    noisy_image = None

    if type == "uniform":
        noisy_image = Noise.UniformNoise(source=data, snr=snr)

    elif type == "gaussian":
        noisy_image = Noise.GaussianNoise(source=data, sigma=sigma, snr=snr)

    elif type == "salt & pepper":
        noisy_image = Noise.SaltPepperNoise(source=data, snr=snr)

    return noisy_image


def apply_filter(data: np.array, type: str, shape: int, sigma: [int, float] = 0) -> np.ndarray:
    """
    This function adds different types of filters to the given image
    :param data: The given image numpy array
    :param type: The type of filter to be applied on the given image
    :return: numpy array of the new filtered image
    """

    filtered_image = None

    if type == "average":
        filtered_image = LowPass.AverageFilter(source=data, shape=shape)

    elif type == "gaussian":
        filtered_image = LowPass.GaussianFilter(source=data, shape=shape, sigma=sigma)

    elif type == "median":
        filtered_image = LowPass.MedianFilter(source=data, shape=shape)

    return filtered_image


def apply_edge_mask(data: np.array, type: str, shape: int = 3):
    """

    :param type:
    :return:
    """
    edged_image = None

    if type == "sobel":
        edged_image = EdgeDetection.sobel_edge(data)

    elif type == "roberts":
        edged_image = EdgeDetection.roberts_edge(data)

    elif type == "prewitt":
        edged_image = EdgeDetection.prewitt_edge(data)

    elif type == "canny":
        # TODO: Add Canny Mask Algorithm on self.imgByte
        pass

    return edged_image


def get_histogram(data: np.array, type: str, bins_num: int = 255):
    """

    :param type:
    :return:
    """

    if type == "original":
        hist, bins = Histogram.histogram(data=data, bins_num=bins_num)
        return hist, bins

    if type == "equalized":
        hist, bins = Histogram.equalize_histogram(data=data, bins_num=bins_num)
        return hist, bins

    elif type == "normalized":
        norm, histo, bins = Histogram.normalize_histogram(data=data, bins_num=bins_num)
        return norm, histo, bins


def thresholding(data: np.array, type: str, threshold: int = 128, divs: int = 4):
    """

    :param data:
    :param type:
    :param threshold:
    :param divs:
    :return:
    """

    threshold_image = None

    if type == "local":
        threshold_image = Histogram.local_threshold(data=data, divs=divs)

    elif type == "global":
        threshold_image = Histogram.global_threshold(data=data, threshold=threshold)

    return threshold_image


def rgb_to_gray(data: np.array):
    """

    :return:
    """
    return np.dot(data[..., :3], [0.299, 0.587, 0.114])


def mix_images(data1: np.ndarray, data2: np.ndarray, hpf_size: int = 15, lpf_size: int = 15):
    """

    :param data1:
    :param data2:
    :param hpf_size:
    :param lpf_size:
    :return:
    """

    image1_dft = FFilters.HighPass(data1, hpf_size)
    image2_dft = FFilters.LowPass(data2, lpf_size)

    return image1_dft + image2_dft
#
# Histograms Implementations
#

import numpy as np


def histogram(source: np.array, bins_num: int = 255):
    """

    :param source:
    :param bins_num:
    :return:
    """
    if bins_num == 2:
        new_data = source
    else:
        new_data = np.round(np.interp(source, (source.min(), source.max()), (0, bins_num))).astype('uint8')
    bins = np.arange(0, bins_num)
    hist = np.bincount(new_data.ravel(), minlength=bins_num)
    return hist, bins


def equalize_histogram(source: np.ndarray, bins_num: int = 255):
    """
        Histogram Equalization Implementation
    :param source: Input Source Image
    :param bins_num: number of bins
    :return: Equalized Image
    """

    bins = np.arange(0, bins_num)

    # Calculate the Occurrences of each pixel in the input
    hist_array = np.bincount(source.flatten(), minlength=bins_num)

    # Normalize Resulted array
    px_count = np.sum(hist_array)
    hist_array = hist_array/px_count

    # Calculate the Cumulative Sum
    hist_array = np.cumsum(hist_array)

    # Pixel Mapping
    trans_map = np.floor(255 * hist_array)

    # Transform Mapping to Image
    img1d = list(source.flatten())
    map_img1d = [trans_map[px] for px in img1d]

    # Reshape Image
    map_img2d = np.reshape(np.asarray(map_img1d), source.shape)

    return map_img2d, bins


def normalize_histogram(source: np.array, bins_num: int = 255):
    mn = np.min(source)
    mx = np.max(source)
    norm = (source - mn) * (1.0 / (mx - mn))
    histo, bins = histogram(norm, bins_num=bins_num)
    return norm, histo, bins


def global_threshold(source: np.ndarray, threshold: int):
    src = np.copy(source)
    if len(src.shape) > 2:
        src = rgb_to_gray(source)
    return (src > threshold).astype(int)


def rgb_to_gray(source: np.ndarray):
    return np.dot(source[..., :3], [0.299, 0.587, 0.114]).astype('uint8')


def local_threshold(source: np.ndarray, divs: int) -> np.ndarray:
    """
        Global Thresholding Implementation using mean
    :param source: Input Source Image
    :param divs: Number of Regions
    :return: Threshold-ed image
    """
    # Vertical Split of the Image
    src = np.copy(source)
    if len(src.shape) > 2:
        src = rgb_to_gray(source)

    s = np.sqrt(divs)
    v_splits = np.split(src, s)

    splits = []
    for sp in v_splits:
        splits.append(np.split(sp, s, -1))

    c1 = []
    # Calculate the mean and threshold for each split
    for ix, x in enumerate(splits):
        for iy, y in enumerate(x):
            threshold = int(np.mean(y))
            splits[ix][iy] = global_threshold(splits[ix][iy], threshold)
        c1.append(np.concatenate(splits[ix], -1))

    out = np.concatenate(c1)
    return out

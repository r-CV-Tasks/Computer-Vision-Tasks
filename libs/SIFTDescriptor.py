import cv2
import numpy as np
from cv2 import KeyPoint

import Harris

#########################
#    Gaussian Filter    #
#########################
float_tolerance = 1e-7


def gaussian_filter(sigma: float) -> np.ndarray:
    """
    Gaussian Filter 2D Kernel Generator
    :param sigma: Standard Deviation
    :return: 2d Array
    """
    size = 4
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
    return g / g.sum()


#########################
# Keypoint orientations #
#########################
def calculateOrientation(keypoint: KeyPoint, source: np.ndarray) -> list:
    """
    Calculates the Orientation of a given keypoint
    :param keypoint: Keypoint class
    :param source: Source Image
    :return: list of all Keypoints after orientation addition
    """
    # Taking only a (9, 9) Window arround each Keypoint
    radius = 9
    bins = 36
    raw_histogram = np.zeros(bins)
    kp_oriented = []
    src = np.copy(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
    smooth_histogram = np.zeros(bins)

    # Looping each pixel in the selected window around the Keypoint
    for i in range(-radius, radius + 1):
        y = int(keypoint.pt[1] + i)
        for j in range(-radius, radius + 1):
            # Calculate Magnitude and Theta
            x = int(keypoint.pt[0] + j)
            if 0 < x < src.shape[1] - 1:
                xx = int(src[y, x + 1]) - int(src[y, x - 1])
                yy = int(src[y + 1, x]) - int(src[y - 1, x])
                mag = np.sqrt(xx * xx + yy * yy)
                theta = np.rad2deg(np.arctan2(yy, xx))
                weight = np.exp(-0.5 * (i ** 2 + j ** 2))

                # Add the Magnitude to the right bin in histogram
                hist_index = abs(int((theta * bins) / 360.0))
                raw_histogram[hist_index % bins] += mag * weight

    for n in range(bins):
        smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % bins]) +
                               raw_histogram[n - 2] + raw_histogram[(n + 2) % bins]) / 16.

    # Finding New points with Orientation higher
    # than 80% of the maximum peak in histogram
    max_orientation = max(smooth_histogram)
    keypoint.angle = max_orientation
    kp_oriented.append(keypoint)

    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1),
                                                smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= 0.8 * max_orientation:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30)
            # in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % bins]
            right_value = smooth_histogram[(peak_index + 1) % bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (
                        left_value - 2 * peak_value + right_value)) % bins
            orientation = 360. - interpolated_peak_index * 360. / bins
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            new_keypoint = KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            kp_oriented.append(new_keypoint)
    return kp_oriented

    # for __bin in smooth_histogram:
    #     if __bin >= 0.8 * max_orientation:
    #         new_kp = KeyPoint(*keypoint.pt, keypoint.size, __bin, keypoint.response, keypoint.octave)
    #         kp_oriented.append(new_kp)
    # return kp_oriented


#########################
# Generating Descriptors#
#########################
def generateDescriptors(keypoint: KeyPoint, source: np.ndarray):
    """
    Calculating the Sift Descriptor
    :param keypoint: Image Features
    :param source: Image
    :return: List(128) a unique fingerprint of the Keypoint
    """
    feature = []
    radius = 7
    bins = 8
    sigma = 1.6
    # window = source[int(keypoint.pt[1]-6):int(keypoint.pt[1]+10), int(keypoint.pt[0]-6):int(keypoint.pt[0]+10)]
    # splits = [np.vsplit(i, 4)*gaussian_filter(1.6) for i in np.hsplit(window, 4)]
    src = np.copy(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
    kernel = gaussian_filter(sigma)

    # # Taking a 16 by 16 window around the keypoint
    for i in range(-radius, radius + 2, 4):
        y = int(keypoint.pt[1] + i)
        for j in range(-radius, radius + 2, 4):
            hist = np.zeros(bins)
            x = int(keypoint.pt[0] + j)
            # Looping each 4 by 4 window inside the larger window
            for window_i in range(y, y + 4):
                for window_j in range(x, x + 4):
                    # Calculate the Weight from the Gaussian Kernel Created
                    weight = kernel[window_j - x, window_i - y]
                    xx = int(src[window_j, window_i + 1] * weight) - int(src[window_j, window_i - 1] * weight)
                    yy = int(src[window_j + 1, window_i] * weight) - int(src[window_j - 1, window_j] * weight)
                    mag = np.sqrt(xx * xx + yy * yy)
                    theta = np.rad2deg(np.arctan2(yy, xx)) - keypoint.angle
                    hist_indx = abs(int((theta * bins) / 360.0))
                    hist[hist_indx % bins] += mag
            feature.extend(hist)
    return np.array(feature)


#########################
# SIFT+Harris Algorithm #
#########################
def siftHarris(source: np.ndarray, n_feats: int = 100, threshold: float = 0.1):
    """
    Using Harris Operator to Calculate Image Features and using the Sift
    Algorithm to generate a Descriptor

    :param source: Input Image
    :param n_feats: Down sampling factor of the Number of
                    Features Selected from Harris
    :param threshold: Harris Operator Threshold
    :return: (n_feats, 128) Unique Features Descriptors
    """
    harris = Harris.apply_harris_operator(source)
    indices = Harris.get_harris_indices(harris, threshold)[0]  # Get only Corners
    indices = np.transpose(np.nonzero(indices))
    kps = []
    for idx in indices:
        k = KeyPoint()
        k.pt = (idx[0], idx[1])
        kps.append(k)

    kps_orientation = []
    dsc = []
    for kp in kps:
        kps_orientation.extend(calculateOrientation(kp, source))
    print(len(kps_orientation))
    for kp in kps_orientation[::n_feats]:
        dsc.append(generateDescriptors(kp, source))

    return kps_orientation, np.array(dsc)


if __name__ == '__main__':
    img = cv2.imread("../resources/Images/cat256_edited_v2.png")
    imgx = cv2.imread("../resources/Images/cat256.jpg")
    _, dscs = siftHarris(img, 1, 0.4)
    _, dscsx = siftHarris(imgx, 1, 0.4)
    print(dscs.shape)
    print(dscsx.shape)

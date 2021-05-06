import numpy as np
from cv2 import KeyPoint
import cv2
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
                weight = np.exp(-0.5 * (i ** 2 + j**2))

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

    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= 0.8 * max_orientation:
            # Quadratic peak interpolation
            # The interpolation update is given by equation (6.30)
            # in https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
            left_value = smooth_histogram[(peak_index - 1) % bins]
            right_value = smooth_histogram[(peak_index + 1) % bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % bins
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
    # for i in range(-radius, radius + 2, 4):
    #     y = int(keypoint.pt[1] + i)
    #     for j in range(-radius, radius + 2, 4):
    #         hist = np.zeros(bins)
    #         x = int(keypoint.pt[0] + j)
    #         # Looping each 4 by 4 window inside the larger window
    #         for window_i in range(y, y + 4):
    #             for window_j in range(x, x + 4):
    #                 # Calculate the Weight from the Gaussian Kernel Created
    #                 weight = kernel[window_j - x, window_i - y]
    #                 xx = int(src[window_j, window_i + 1] * weight) - int(src[window_j, window_i - 1] * weight)
    #                 yy = int(src[window_j + 1, window_i] * weight) - int(src[window_j - 1, window_j] * weight)
    #                 mag = np.sqrt(xx * xx + yy * yy)
    #                 theta = np.rad2deg(np.arctan2(yy, xx)) - keypoint.angle
    #                 hist_indx = abs(int((theta * bins) / 360.0))
    #                 hist[hist_indx % bins] += mag
    #         feature.extend(hist)
    # return np.array(feature)
    window_width = 4
    descriptors = []

    num_rows, num_cols = src.shape
    point = np.round(np.array(keypoint.pt)).astype('int')
    descriptor_max_value = 0.2

    bins_per_degree = bins / 360.
    angle = 360. - keypoint.angle
    cos_angle = np.cos(np.deg2rad(angle))
    sin_angle = np.sin(np.deg2rad(angle))
    weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
    row_bin_list = []
    col_bin_list = []
    magnitude_list = []
    orientation_bin_list = []
    # first two dimensions are increased by 2 to account for border effects
    histogram_tensor = np.zeros((window_width + 2, window_width + 2, bins))

    # Descriptor window size (described by half_width) follows OpenCV convention
    hist_width = 8
    # sqrt(2) corresponds to diagonal length of a pixel
    half_width = int(round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))
    # ensure half_width lies within image
    half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))

    for row in range(-half_width, half_width + 1):
        for col in range(-half_width, half_width + 1):
            row_rot = col * sin_angle + row * cos_angle
            col_rot = col * cos_angle - row * sin_angle
            row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
            col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
            if -1 < row_bin < window_width and -1 < col_bin < window_width:
                window_row = int(round(point[1] + row))
                window_col = int(round(point[0] + col))
                if 0 < window_row < num_rows - 1 and 0 < window_col < num_cols - 1:
                    dx = int(src[window_row, window_col + 1]) - int(src[window_row, window_col - 1])
                    dy = int(src[window_row - 1, window_col]) - int(src[window_row + 1, window_col])
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                    weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                    row_bin_list.append(row_bin)
                    col_bin_list.append(col_bin)
                    magnitude_list.append(weight * gradient_magnitude)
                    orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

    for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
        # Smoothing via trilinear interpolation
        # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
        # Note that we are really doing the inverse of trilinear interpolation here
        # (we take the center value of the cube and distribute it among its eight neighbors)
        row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
        row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
        if orientation_bin_floor < 0:
            orientation_bin_floor += bins
        if orientation_bin_floor >= bins:
            orientation_bin_floor -= bins

        c1 = magnitude * row_fraction
        c0 = magnitude * (1 - row_fraction)
        c11 = c1 * col_fraction
        c10 = c1 * (1 - col_fraction)
        c01 = c0 * col_fraction
        c00 = c0 * (1 - col_fraction)
        c111 = c11 * orientation_fraction
        c110 = c11 * (1 - orientation_fraction)
        c101 = c10 * orientation_fraction
        c100 = c10 * (1 - orientation_fraction)
        c011 = c01 * orientation_fraction
        c010 = c01 * (1 - orientation_fraction)
        c001 = c00 * orientation_fraction
        c000 = c00 * (1 - orientation_fraction)

        histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
        histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % bins] += c001
        histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
        histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % bins] += c011
        histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
        histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % bins] += c101
        histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
        histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % bins] += c111

    descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
    # Threshold and normalize descriptor_vector
    threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
    descriptor_vector[descriptor_vector > threshold] = threshold
    descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)
    # Multiply by 512, round, and saturate between 0 and 255
    # to convert from float32 to unsigned char (OpenCV convention)
    descriptor_vector = np.round(512 * descriptor_vector)
    descriptor_vector[descriptor_vector < 0] = 0
    descriptor_vector[descriptor_vector > 255] = 255
    descriptors.append(descriptor_vector)
    return np.array(descriptors, dtype='float32')


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
        dsc.extend(generateDescriptors(kp, source))

    return kps_orientation, np.array(dsc)


if __name__ == '__main__':
    img = cv2.imread("../resources/Images/cat256_edited_v2.png")
    imgx = cv2.imread("../resources/Images/cat256.jpg")
    _, dscs = siftHarris(img, 1, 0.4)
    _, dscsx = siftHarris(imgx, 1, 0.4)
    print(dscs.shape)
    print(dscsx.shape)

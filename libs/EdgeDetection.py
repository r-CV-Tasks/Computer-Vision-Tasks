import cv2
import numpy as np
from scipy.signal import convolve2d
from LowPass import gaussian_filter


def apply_kernel(source: np.ndarray, horizontal_kernel: np.ndarray, vertical_kernel: np.ndarray,
                 ReturnEdge: bool = False):
    """
        Convert image to gray scale and convolve with kernels
        :param source: Image to apply kernel to
        :param horizontal_kernel: The horizontal array of the kernel
        :param vertical_kernel: The vertical array of the kernel
        :param ReturnEdge: Return Horizontal & Vertical Edges
        :return: The result of convolution
    """
    # convert to gray scale if not already
    if len(source.shape) > 2:
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    else:
        gray = source

    # convolution
    horizontal_edge = convolve2d(gray, horizontal_kernel)
    vertical_edge = convolve2d(gray, vertical_kernel)

    mag = np.sqrt(pow(horizontal_edge, 2.0) + pow(vertical_edge, 2.0))
    if ReturnEdge:
        return mag, horizontal_edge, vertical_edge
    return mag


def prewitt_edge(source: np.ndarray):
    """
        Apply Prewitt Operator to detect edges
        :param source: Image to detect edges in
        :return: edges image
    """
    # define filters
    vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    horizontal = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    mag = apply_kernel(source, horizontal, vertical)

    return mag


def sobel_edge(source: np.ndarray, GetDirection: bool = False):
    """
        Apply Sobel Operator to detect edges
        :param source: Image to detect edges in
        :param GetDirection: Get Gradient direction in Pi Terms
        :return: edges image
    """
    # define filters
    # vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = np.flip(horizontal.T)
    mag, horizontal_edge, vertical_edge = apply_kernel(source, horizontal, vertical, True)

    if GetDirection:
        direction = np.arctan2(vertical_edge, horizontal_edge)
        return mag, direction
    return mag


def roberts_edge(source: np.ndarray):
    """
        Apply Roberts Operator to detect edges
        :param source: Image to detect edges in
        :return: edges image
    """
    # define filters
    vertical = np.array([[0, 1], [-1, 0]])
    horizontal = np.array([[1, 0], [0, -1]])

    mag = apply_kernel(source, horizontal, vertical)

    return mag


def canny_edge(source: np.ndarray):
    """
    Apply Canny Operator to detect edges
    :param source: Image to detect edges in
    :return: edges image
    """
    # Convert to gray Scale
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Filter
    filtered_image = gaussian_filter(gray, 3, 9)

    # Get Gradient Magnitude & Direction
    gradient_magnitude, gradient_direction = sobel_edge(filtered_image, True)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    # Apply Non-Maximum Suppression
    suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Apply Double Thresholding
    thresholded_image = double_threshold(suppressed_image, 0.05, 0.09, 70)

    # Apply Hysteresis
    canny_edges = hysteresis(thresholded_image, 70, 255)

    return canny_edges


def non_maximum_suppression(gradient_magnitude: np.ndarray, gradient_direction: np.ndarray):
    """
    Applies Non-Maximum Suppressed Gradient Image To Thin Out The Edges
    :param gradient_magnitude: Gradient Image To A Thin Out It's Edges
    :param gradient_direction: Direction of The Image's Edges
    :return Non-Maximum Suppressed Image:
    """
    m, n = gradient_magnitude.shape
    suppressed_image = np.zeros(gradient_magnitude.shape)

    # Convert Rad Directions To Degree
    gradient_direction = np.rad2deg(gradient_direction)
    gradient_direction += 180
    pi = 180

    for row in range(1, m - 1):
        for col in range(1, n - 1):
            try:
                direction = gradient_direction[row, col]
                # 0°
                if (0 <= direction < pi / 8) or (15 * pi / 8 <= direction <= 2 * pi):
                    before_pixel = gradient_magnitude[row, col - 1]
                    after_pixel = gradient_magnitude[row, col + 1]
                # 45°
                elif (pi / 8 <= direction < 3 * pi / 8) or (9 * pi / 8 <= direction < 11 * pi / 8):
                    before_pixel = gradient_magnitude[row + 1, col - 1]
                    after_pixel = gradient_magnitude[row - 1, col + 1]
                # 90°
                elif (3 * pi / 8 <= direction < 5 * pi / 8) or (11 * pi / 8 <= direction < 13 * pi / 8):
                    before_pixel = gradient_magnitude[row - 1, col]
                    after_pixel = gradient_magnitude[row + 1, col]
                # 135°
                else:
                    before_pixel = gradient_magnitude[row - 1, col - 1]
                    after_pixel = gradient_magnitude[row + 1, col + 1]

                if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                    suppressed_image[row, col] = gradient_magnitude[row, col]
            except IndexError as e:
                raise e

    return suppressed_image


def double_threshold(source, low_ratio, high_ratio, weak):
    """
       Apply Double Thresholding To Image
       :param source: Image to Threshold
       :param low_ratio: low Threshold Ratio, Used to Get Lowest Allowed Value
       :param high_ratio: high Threshold Ratio, Used to Get Minimum Value To Be Boosted
       :param weak: Pixel Value For Pixels Between The Two Thresholds
       :return: Thresholded Image
       """

    # Get Threshold Values
    high = source.max() * high_ratio
    low = source.max() * low_ratio

    # Create Empty Array
    thresholded_image = np.zeros(source.shape)

    strong = 255
    # Find Position of strong & Weak Pixels
    strong_row, strong_col = np.where(source >= high)
    weak_row, weak_col = np.where((source <= high) & (source >= low))

    # Apply Thresholding
    thresholded_image[strong_row, strong_col] = strong
    thresholded_image[weak_row, weak_col] = weak

    return thresholded_image


def hysteresis(source, weak=70, strong=255):
    """

    :param source:
    :param weak:
    :param strong:
    :return:
    """
    m, n = source.shape
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            if source[i, j] == weak:
                try:
                    if ((source[i + 1, j - 1] == strong) or (source[i + 1, j] == strong) or (
                            source[i + 1, j + 1] == strong)
                            or (source[i, j - 1] == strong) or (source[i, j + 1] == strong)
                            or (source[i - 1, j - 1] == strong) or (source[i - 1, j] == strong) or (
                                    source[i - 1, j + 1] == strong)):
                        source[i, j] = strong
                    else:
                        source[i, j] = 0
                except IndexError as e:
                    raise e
    return source

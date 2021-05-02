import cv2
import numpy as np
from scipy.signal import convolve2d

from libs.LowPass import gaussian_filter


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


def sobel_edge(source: np.ndarray, GetMagnitude: bool = True, GetDirection: bool = False):
    """
        Apply Sobel Operator to detect edges
        :param source: Image to detect edges in
        :param GetMagnitude: Get Magnitude of horizontal and vertical edges
        :param GetDirection: Get Gradient direction in Pi Terms
        :return: edges image
    """
    # define filters
    # vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # horizontal = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical = np.flip(horizontal.T)
    mag, HorizontalEdge, VerticalEdge = apply_kernel(source, horizontal, vertical, True)

    HorizontalEdge = HorizontalEdge[:-2, :-2]
    VerticalEdge = VerticalEdge[:-2, :-2]

    if GetMagnitude == False:
        return HorizontalEdge, VerticalEdge

    if GetDirection:
        Direction = np.arctan2(VerticalEdge, HorizontalEdge)
        return mag, Direction

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
    # Convert to Gray Scale
    Gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian Filter
    FilteredImage = gaussian_filter(Gray, 3, 9)

    # Get Gradient Magnitude & Direction
    GradientMagnitude, GradientDirection = sobel_edge(FilteredImage, True)
    GradientMagnitude *= 255.0 / GradientMagnitude.max()

    # Apply Non-Maximum Suppression
    SuppressedImage = NonMaximumSuppression(GradientMagnitude, GradientDirection)

    # Apply Double Thresholding
    ThresholdedImage = DoubleThreshold(SuppressedImage, 0.05, 0.09, 70)

    # Apply Hysteresis
    CannyEdges = Hysteresis(ThresholdedImage, 70, 255)

    return CannyEdges


def NonMaximumSuppression(GradientMagnitude: np.ndarray, GradientDirection: np.ndarray):
    """
    Applies Non-Maximum Suppressed Gradient Image To Thin Out The Edges
    :param GradientMagnitude: Gradient Image To Thin Out It's Edges
    :param GradientDirection: Direction of The Image's Edges
    :return Non-Maximum Suppressed Image:
    """
    M, N = GradientMagnitude.shape
    SuppressedImage = np.zeros(GradientMagnitude.shape)

    # Convert Rad Directions To Degree
    GradientDirection = np.rad2deg(GradientDirection)
    GradientDirection += 180
    PI = 180

    for row in range(1, M - 1):
        for col in range(1, N - 1):
            try:
                direction = GradientDirection[row, col]
                # 0°
                if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                    before_pixel = GradientMagnitude[row, col - 1]
                    after_pixel = GradientMagnitude[row, col + 1]
                # 45°
                elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                    before_pixel = GradientMagnitude[row + 1, col - 1]
                    after_pixel = GradientMagnitude[row - 1, col + 1]
                # 90°
                elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                    before_pixel = GradientMagnitude[row - 1, col]
                    after_pixel = GradientMagnitude[row + 1, col]
                # 135°
                else:
                    before_pixel = GradientMagnitude[row - 1, col - 1]
                    after_pixel = GradientMagnitude[row + 1, col + 1]

                if GradientMagnitude[row, col] >= before_pixel and GradientMagnitude[row, col] >= after_pixel:
                    SuppressedImage[row, col] = GradientMagnitude[row, col]
            except IndexError as e:
                pass

    return SuppressedImage


def DoubleThreshold(Image, LowRatio, HighRatio, Weak):
    """
       Apply Double Thresholding To Image
       :param Image: Image to Threshold
       :param LowRatio: Low Threshold Ratio, Used to Get Lowest Allowed Value
       :param HighRatio: High Threshold Ratio, Used to Get Minimum Value To Be Boosted
       :param Weak: Pixel Value For Pixels Between The Two Thresholds
       :return: Thresholded Image
       """

    # Get Threshold Values
    High = Image.max() * HighRatio
    Low = Image.max() * LowRatio

    # Create Empty Array
    ThresholdedImage = np.zeros(Image.shape)

    Strong = 255
    # Find Position of Strong & Weak Pixels
    StrongRow, StrongCol = np.where(Image >= High)
    WeakRow, WeakCol = np.where((Image <= High) & (Image >= Low))

    # Apply Thresholding
    ThresholdedImage[StrongRow, StrongCol] = Strong
    ThresholdedImage[WeakRow, WeakCol] = Weak

    return ThresholdedImage


def Hysteresis(Image, Weak=70, Strong=255):
    M, N = Image.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if Image[i, j] == Weak:
                try:
                    if ((Image[i + 1, j - 1] == Strong) or (Image[i + 1, j] == Strong) or (
                            Image[i + 1, j + 1] == Strong)
                            or (Image[i, j - 1] == Strong) or (Image[i, j + 1] == Strong)
                            or (Image[i - 1, j - 1] == Strong) or (Image[i - 1, j] == Strong) or (
                                    Image[i - 1, j + 1] == Strong)):
                        Image[i, j] = Strong
                    else:
                        Image[i, j] = 0
                except IndexError as e:
                    pass
    return Image

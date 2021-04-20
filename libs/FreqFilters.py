
#
# Frequency Domain Filters
#

import numpy as np
from imageModel import rgb_to_gray


def SquareZeroPad(source: np.ndarray, size_x: int, size_y: int) -> np.ndarray:
    """
        Pad Image/Array to Desired Output shape
    :param source: Input Array/Image
    :param size_x: Desired width size
    :param size_y: Desired height
    :return: Padded Square Array
    """
    src = np.copy(source)
    x, y = src.shape

    out_x = (size_x-x) // 2
    out_xx = size_x - out_x - x

    out_y = (size_y-y) // 2
    out_yy = size_y - out_y - y

    return np.pad(src, ((out_x, out_xx), (out_y, out_yy)), constant_values=0)


def __FFilter(source: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
        Application of a Filter in frequency domain
    :param source: Source Image
    :param kernel: Kernel applied on image
    :return: Filtered Image
    """
    src = np.copy(source)

    # Covert Image to gray scale
    src = rgb_to_gray(src)

    # Convert Image to Frequency Domain
    # and decentralize the output
    ft_src = np.fft.fft2(src)
    ft_src_shifted = np.fft.fftshift(ft_src)

    # apply Kernel
    out = np.fft.ifftshift(ft_src_shifted * kernel)
    out = np.fft.ifft2(out)

    return np.abs(out)


def HighPass(source: np.ndarray, size: int) -> np.ndarray:
    """
        Frequency Domain High Pass Filter
    :param source: Source Image
    :param size: An Integer that denotes the Kernel size if 3
                   then the kernel is (3, 3)
    :return: Filtered Image
    """
    # Create a kernel with ones in the middle for high frequencies
    kernel = SquareZeroPad(np.ones((size, size)), source.shape[0], source.shape[1])

    # Apply Kernel
    out = __FFilter(source, kernel)

    return out


def LowPass(source: np.ndarray, size: int) -> np.ndarray:
    """
            Frequency Domain Low Pass Filter
    :param source: Source Image
    :param size: An Integer that denotes the Kernel size if 3
                   then the kernel is (3, 3)
    :return: Filtered Image
    """
    # Create Kernel with ones on the edges for low frequencies
    kernel = 1 - SquareZeroPad(np.ones((size, size)), source.shape[0], source.shape[1])

    # Apply Kernel
    out = __FFilter(source, kernel)

    return out

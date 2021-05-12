import cv2
import numpy as np
import matplotlib.pyplot as plt
from Histogram import global_threshold


def apply_optimal_threshold(source: np.ndarray):
    """

    :param source:
    :return:
    """

    src = np.copy(source)
    OldThreshold = GetInitialThreshold(src)
    NewThreshold = GetOptimalThreshold(src, OldThreshold)
    iteration = 0
    while OldThreshold != NewThreshold:
        OldThreshold = NewThreshold
        NewThreshold = GetOptimalThreshold(src, OldThreshold)
        iteration += 1
    # src[src >= 25] = 0
    return global_threshold(src, NewThreshold)


def GetInitialThreshold(source: np.ndarray):
    MaxX = source.shape[1] - 1
    MaxY = source.shape[0] - 1
    BackMean = (source[0, 0] + source[0, MaxX] + source[MaxY, 0] + source[MaxY, MaxX]) / 4
    Sum = 0
    Length = 0
    for i in range(0, source.shape[1]):
        for j in range(0, source.shape[0]):
            if not ((i == 0 and j == 0) or (i == MaxX and j == 0) or (i == 0 and j == MaxY) or (
                    i == MaxX and j == MaxY)):
                Sum += source[j, i]
                Length += 1
    ForeMean = Sum / Length
    Threshold = (BackMean + ForeMean) / 2
    return Threshold


def GetOptimalThreshold(source: np.ndarray, Threshold):
    Back = source[np.where(source < Threshold)]
    Fore = source[np.where(source > Threshold)]
    BackMean = np.mean(Back)
    ForeMean = np.mean(Fore)
    Threshold = (BackMean + ForeMean) / 2
    return Threshold


def apply_otsu_threshold(source: np.ndarray):
    """

    :param source:
    :return:
    """

    src = np.copy(source)

    return src


def apply_spectral_threshold(source: np.ndarray):
    """

    :param source:
    :return:
    """

    src = np.copy(source)

    return src


def IsolatedTests():
    """
    Tests The Thresholding Techniques Without Need For GUI, DEV-ONLY
    :return:
    """
    source = cv2.imread("../resources/Images/hand_512.jpg", 0)
    if len(source.shape) > 2:
        img = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    else:
        img = source
    x = img.shape[1]
    y = img.shape[0]
    OptImg = apply_optimal_threshold(img)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(source, cmap='gray')
    ax[1].imshow(OptImg, cmap='gray')
    print()
    plt.show()


if __name__ == "__main__":
    IsolatedTests()

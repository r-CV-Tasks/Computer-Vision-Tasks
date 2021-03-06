import cv2
import matplotlib.pyplot as plt
import numpy as np

from Histogram import global_threshold, normalize_histogram
from EdgeDetection import DoubleThreshold


def apply_optimal_threshold(source: np.ndarray):
    """
    Applies Thresholding To The Given Grayscale Image Using The Optimal Thresholding Method
    :param source: NumPy Array of The Source Grayscale Image
    :return: Thresholded Image
    """

    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    # Calculate Initial Thresholds Used in Iteration
    print(f"src in optimal: {src}")
    print(f"src shape: {src.shape}")
    OldThreshold = GetInitialThreshold(src)
    NewThreshold = GetOptimalThreshold(src, OldThreshold)
    iteration = 0
    # Iterate Till The Threshold Value is Constant Across Two Iterations
    while OldThreshold != NewThreshold:
        OldThreshold = NewThreshold
        NewThreshold = GetOptimalThreshold(src, OldThreshold)
        iteration += 1
    # src[src >= 25] = 0
    # Return Thresholded Image Using Global Thresholding
    return global_threshold(src, NewThreshold)


def GetInitialThreshold(source: np.ndarray):
    """
    Gets The Initial Threshold Used in The Optimal Threshold Method
    :param source: NumPy Array of The Source Grayscale Image
    :return Threshold: Initial Threshold Value
    """
    # Maximum X & Y Values For The Image
    MaxX = source.shape[1] - 1
    MaxY = source.shape[0] - 1
    # Mean Value of Background Intensity, Calculated From The Four Corner Pixels
    BackMean = (int(source[0, 0]) + int(source[0, MaxX]) + int(source[MaxY, 0]) + int(source[MaxY, MaxX])) / 4
    Sum = 0
    Length = 0
    # Loop To Calculate Mean Value of Foreground Intensity
    for i in range(0, source.shape[1]):
        for j in range(0, source.shape[0]):
            # Skip The Four Corner Pixels
            if not ((i == 0 and j == 0) or (i == MaxX and j == 0) or (i == 0 and j == MaxY) or (
                    i == MaxX and j == MaxY)):
                Sum += source[j, i]
                Length += 1
    ForeMean = Sum / Length
    # Get The Threshold, The Average of The Mean Background & Foreground Intensities
    Threshold = (BackMean + ForeMean) / 2
    return Threshold


def GetOptimalThreshold(source: np.ndarray, Threshold):
    """
    Calculates Optimal Threshold Based on Given Initial Threshold
    :param source: NumPy Array of The Source Grayscale Image
    :param Threshold: Initial Threshold
    :return OptimalThreshold: Optimal Threshold Based on Given Initial Threshold
    """
    # Get Background Array, Consisting of All Pixels With Intensity Lower Than The Given Threshold
    Back = source[np.where(source < Threshold)]
    # Get Foreground Array, Consisting of All Pixels With Intensity Higher Than The Given Threshold
    Fore = source[np.where(source > Threshold)]
    # Mean of Background & Foreground Intensities
    BackMean = np.mean(Back)
    ForeMean = np.mean(Fore)
    # Calculate Optimal Threshold
    OptimalThreshold = (BackMean + ForeMean) / 2
    return OptimalThreshold


def apply_otsu_threshold(source: np.ndarray):
    """
     Applies Thresholding To The Given Grayscale Image Using Otsu's Thresholding Method
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    src = np.copy(source)

    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass


    # Get Image Dimensions
    YRange, XRange = src.shape
    # Get The Values of The Histogram Bins
    HistValues = plt.hist(src.ravel(), 256)[0]
    # Calculate The Probability Density Function
    PDF = HistValues / (YRange * XRange)
    # Calculate The Cumulative Density Function
    CDF = np.cumsum(PDF)
    OptimalThreshold = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    for t in range(1, 255):
        # Background Intensities Array
        Back = np.arange(0, t)
        # Object/Foreground Intensities Array
        Fore = np.arange(t, 256)
        # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
        CDF2 = np.sum(PDF[t + 1:256])
        BackMean = sum(Back * PDF[0:t]) / CDF[t]
        ForeMean = sum(Fore * PDF[t:256]) / CDF2
        # Calculate Cross-Class Variance
        Variance = CDF[t] * CDF2 * (ForeMean - BackMean) ** 2
        # Filter Out Max Variance & It's Threshold
        if Variance > MaxVariance:
            MaxVariance = Variance
            OptimalThreshold = t
    return global_threshold(src, OptimalThreshold)


def apply_spectral_threshold(source: np.ndarray):
    """
     Applies Thresholding To The Given Grayscale Image Using Spectral Thresholding Method
     :param source: NumPy Array of The Source Grayscale Image
     :return: Thresholded Image
     """
    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    # Get Image Dimensions
    YRange, XRange = src.shape
    # Get The Values of The Histogram Bins
    HistValues = plt.hist(src.ravel(), 256)[0]
    # Calculate The Probability Density Function
    PDF = HistValues / (YRange * XRange)
    # Calculate The Cumulative Density Function
    CDF = np.cumsum(PDF)
    OptimalLow = 1
    OptimalHigh = 1
    MaxVariance = 0
    # Loop Over All Possible Thresholds, Select One With Maximum Variance Between Background & The Object (Foreground)
    Global = np.arange(0, 256)
    GMean = sum(Global * PDF) / CDF[-1]
    for LowT in range(1, 254):
        for HighT in range(LowT + 1, 255):
            try:
                # Background Intensities Array
                Back = np.arange(0, LowT)
                # Low Intensities Array
                Low = np.arange(LowT, HighT)
                # High Intensities Array
                High = np.arange(HighT, 256)
                # Get Low Intensities CDF
                CDFL = np.sum(PDF[LowT:HighT])
                # Get Low Intensities CDF
                CDFH = np.sum(PDF[HighT:256])
                # Calculation Mean of Background & The Object (Foreground), Based on CDF & PDF
                BackMean = sum(Back * PDF[0:LowT]) / CDF[LowT]
                LowMean = sum(Low * PDF[LowT:HighT]) / CDFL
                HighMean = sum(High * PDF[HighT:256]) / CDFH
                # Calculate Cross-Class Variance
                Variance = (CDF[LowT] * (BackMean - GMean) ** 2 + (CDFL * (LowMean - GMean) ** 2) + (
                        CDFH * (HighMean - GMean) ** 2))
                # Filter Out Max Variance & It's Threshold
                if Variance > MaxVariance:
                    MaxVariance = Variance
                    OptimalLow = LowT
                    OptimalHigh = HighT
            except RuntimeWarning:
                pass
    return DoubleThreshold(src, OptimalLow, OptimalHigh, 128, False)


def LocalThresholding(source: np.ndarray, RegionsX: int, RegionsY: int, ThresholdingFunction):
    """
       Applies Local Thresholding To The Given Grayscale Image Using The Given Thresholding Callback Function
       :param source: NumPy Array of The Source Grayscale Image
       :param Regions: Number of Regions To Divide The Image To
       :param ThresholdingFunction: Function That Does The Thresholding
       :return: Thresholded Image
       """
    src = np.copy(source)
    if len(src.shape) > 2:
        src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        pass

    YMax, XMax = src.shape
    Result = np.zeros((YMax, XMax))
    YStep = YMax // RegionsY
    XStep = XMax // RegionsX
    XRange = []
    YRange = []
    for i in range(0, RegionsX):
        XRange.append(XStep * i)

    for i in range(0, RegionsY):
        YRange.append(YStep * i)

    XRange.append(XMax)
    YRange.append(YMax)
    for x in range(0, RegionsX):
        for y in range(0, RegionsY):
            Result[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]] = ThresholdingFunction(src[YRange[y]:YRange[y + 1], XRange[x]:XRange[x + 1]])
    return Result


def IsolatedTests():
    """
    Tests The Thresholding Techniques Without Need For GUI, DEV-ONLY
    :return:
    """
    # source = cv2.imread("../resources/Images/hand_512.jpg")
    source = cv2.imread("../resources/Images/hand_512.jpg")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    img = np.copy(source)
    img_copy = np.copy(source)
    # if len(source.shape) > 2:
    #     img = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    # else:
    #     img = source
    x = img.shape[1]
    y = img.shape[0]
    LocOpt = LocalThresholding(img, 10, apply_otsu_threshold)
    # OptImg = apply_optimal_threshold(img)
    # OtsuImg = apply_otsu_threshold(img)
    # SpecImg = apply_spectral_threshold(img)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(source, cmap='gray')
    ax[0][0].title.set_text('Source Image')
    ax[0][0].set_axis_off()
    ax[0][1].imshow(LocOpt, cmap='gray')
    ax[0][1].title.set_text('Optimal Local Threshold')
    ax[0][1].set_axis_off()
    # ax[1][0].imshow(OptImg, cmap='gray')
    # ax[1][0].title.set_text('Optimal Global Threshold')
    # ax[1][0].set_axis_off()
    # ax[1][1].imshow(SpecImg, cmap='gray')
    # ax[1][1].title.set_text('Spectral Threshold')
    # ax[1][1].set_axis_off()
    plt.show()


if __name__ == "__main__":
    IsolatedTests()

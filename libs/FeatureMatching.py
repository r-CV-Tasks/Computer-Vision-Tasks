import cv2
import numpy as np
from libs.Harris import apply_harris_operator2
import matplotlib.pyplot as plt


def apply_feature_matching(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    """

    :param desc1:
    :param desc2:
    :return:
    """

    out = desc1

    return out

def match_features(desc1: np.ndarray, desc2: np.ndarray, calculator) -> list:
    """

    Perform feature matching between 2 feature descriptors

    :param desc1: The feature descriptors of image 1.
                  Dimensions: rows (number of key points) x columns (dimension of the feature descriptor i.e: 128)
    :param desc2: The feature descriptors of image 2.
                  Dimensions: rows (number of key points) x columns (dimension of the feature descriptor i.e: 128)
    :param calculator: A function to use in matching features

    :return:
        features matches, a list of cv2.DMatch objects:
            How to set attributes:
                - queryIdx: The index of the feature in the first image
                - trainIdx: The index of the feature in the second image
                - distance: The distance between the two features
    """

    # Check descriptors dimensions are 2
    assert desc1.ndim == 2, "Descriptor 1 shape is not 2"
    assert desc2.ndim == 2, "Descriptor 2 shape is not 2"

    # Check that the two features have the same descriptor type
    assert desc1.shape[1] == desc2.shape[1], "Descriptors shapes are not equal"

    # If there is not key points in any of the images
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    # number of key points in each image
    numKeyPoints1 = desc1.shape[0]
    numKeyPoints2 = desc2.shape[0]

    # List to store the matches scores
    matches = []

    # Loop over each key point in image1
    # We need to calculate similarity with each key point in image2
    for kp1 in range(numKeyPoints1):
        # Initial variables which will be updated in the loop
        distance = -np.inf
        y_index = -1

        # Loop over each key point in image2
        for kp2 in range(numKeyPoints2):

            # Match features between the 2 vectors
            value = calculator(desc1[kp1], desc2[kp2])

            if value > distance:
                distance = value
                y_index = kp2

        cur = cv2.DMatch()
        cur.queryIdx = kp1
        cur.trainIdx = y_index
        cur.distance = distance
        matches.append(cur)

    return matches

def calculate_ssd(desc1: list, desc2: list) -> float:
    """
    This function is responsible of:
        - Calculating the Sum Square Distance between two feature vectors.
        - Matching a feature in the first image with the closest feature in the second image.

    Note:
        - Multiple features from the first image may match the same feature in the second image.
        - We need to minimize the SSD value.

    :param desc1: The feature descriptor vector of one key point in image1.
                  Dimensions: rows (1) x columns (dimension of the feature descriptor i.e: 128)
    :param desc2: The feature descriptor vector of one key point in image2.
                  Dimensions: rows (1) x columns (dimension of the feature descriptor i.e: 128)
    :return: A float number represent the SSD between two features vectors
    """

    sumSquare = 0

    # Get SSD between the 2 vectors
    for m in range(len(desc1)):
        sumSquare += (desc1[m] - desc2[m]) ** 2

    # The (-) sign here because the condition we applied after this function call is reversed
    sumSquare = - (np.sqrt(sumSquare))

    return sumSquare


def calculate_ncc(desc1: list, desc2: list) -> float:
    """
    This function is responsible of:
        - Calculating the Normalized Cross Correlation between two feature vectors.
        - Matching a feature in the first image with the closest feature in the second image.

    Note:
        - Multiple features from the first image may match the same feature in the second image.
        - We need to maximize the correlation value.

    :param desc1: The feature descriptor vector of one key point in image1.
                  Dimensions: rows (1) x columns (dimension of the feature descriptor i.e: 128)
    :param desc2: The feature descriptor vector of one key point in image2.
                  Dimensions: rows (1) x columns (dimension of the feature descriptor i.e: 128)
    :return: A float number represent the correlation between two features vectors
    """

    # Normalize the 2 vectors
    out1_norm = (desc1 - np.mean(desc1)) / (np.std(desc1))
    out2_norm = (desc2 - np.mean(desc2)) / (np.std(desc2))

    # Apply similarity product between the 2 normalized vectors
    corr_vector = np.multiply(out1_norm, out2_norm)

    # Get mean of the result vector
    corr = float(np.mean(corr_vector))

    return corr


# TODO # Unfinished function
def calculate_rssd(desc1, desc2):
    sumSquare = 0

    for m in range(len(desc1)):
        sumSquare += (desc1[m] - desc2[m]) ** 2
    sumSquare = np.sqrt(sumSquare)



def matchFeatures_SSD(desc1, desc2):
    """'''
    Input:
        desc1 -- the feature descriptors of image 1 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
        desc2 -- the feature descriptors of image 2 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
    Output:
        features matches: a list of cv2.DMatch objects
            How to set attributes:
                queryIdx: The index of the feature in the first image
                trainIdx: The index of the feature in the second image
                distance: The distance between the two features
    """

    # feature count = n
    assert desc1.ndim == 2, "dimensions are not equal"
    # feature count = m
    assert desc2.ndim == 2
    # the two features should have the type
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    # TODO 7: Perform simple feature matching.  This uses the SSD
    # distance between two feature vectors, and matches a feature in
    # the first image with the closest feature in the second image.
    # Note: multiple features from the first image may match the same
    # feature in the second image.
    # TODO-BLOCK-BEGIN
    numKeyPoints1 = desc1.shape[0]
    numKeyPoints2 = desc2.shape[0]
    matches = []

    for x in range(numKeyPoints1):
        # distance = -1
        distance = np.inf
        y_ind = -1
        for y in range(numKeyPoints2):
            sumSquare = 0
            for m in range(desc1.shape[1]):
                sumSquare += (desc1[x][m] - desc2[y][m]) ** 2

            sumSquare = np.sqrt(sumSquare)

            if sumSquare < distance:
                distance = sumSquare
                y_ind = y

        cur = cv2.DMatch()
        cur.queryIdx = x
        cur.trainIdx = y_ind
        cur.distance = distance
        matches.append(cur)
    # TODO-BLOCK-END

    return matches


def matchFeatures_Ratio_Test(desc1, desc2):
    """"
    Input:
        desc1 -- the feature descriptors of image 1 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
        desc2 -- the feature descriptors of image 2 stored in a numpy array,
            dimensions: rows (number of key points) x
            columns (dimension of the feature descriptor)
    Output:
        features matches: a list of cv2.DMatch objects
            How to set attributes:
                queryIdx: The index of the feature in the first image
                trainIdx: The index of the feature in the second image
                distance: The ratio test score
    """

    matches = []
    # feature count = n
    assert desc1.ndim == 2
    # feature count = m
    assert desc2.ndim == 2
    # the two features should have the type
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    # TODO 8: Perform ratio feature matching.
    # This uses the ratio of the SSD distance of the two best matches
    # and matches a feature in the first image with the closest feature in the
    # second image.
    # Note: multiple features from the first image may match the same
    # feature in the second image.
    # You don't need to threshold matches in this function
    # TODO-BLOCK-BEGIN
    numKeyPoints1 = desc1.shape[0]
    numKeyPoints2 = desc2.shape[0]
    matches = []

    for x in range(numKeyPoints1):
        distance1 = -1
        distance2 = -1
        y_ind = -1
        for y in range(numKeyPoints2):
            sumSquare = 0
            for m in range(desc1.shape[1]):
                sumSquare += (desc1[x][m] - desc2[y][m]) ** 2
            sumSquare = np.sqrt(sumSquare)
            if distance1 < 0 or (sumSquare < distance1 and distance1 >= 0):
                distance2 = distance1
                distance1 = sumSquare
                y_ind = y
            elif distance2 < 0 or (sumSquare < distance2 and distance2 >= 0):
                distance2 = sumSquare
        cur = cv2.DMatch()
        cur.queryIdx = x
        cur.trainIdx = y_ind
        cur.distance = distance1 / distance2
        matches.append(cur)
    # TODO-BLOCK-END

    return matches


def normalized_cross_correlation(desc1, desc2):
    # feature count = n
    assert desc1.ndim == 2
    # feature count = m
    assert desc2.ndim == 2
    # the two features should have the type
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    # TODO 7: Perform simple feature matching.  This uses the SSD
    # distance between two feature vectors, and matches a feature in
    # the first image with the closest feature in the second image.
    # Note: multiple features from the first image may match the same
    # feature in the second image.
    # TODO-BLOCK-BEGIN
    numKeyPoints1 = desc1.shape[0]
    numKeyPoints2 = desc2.shape[0]

    print(f"kp1: {numKeyPoints1}")
    print(f"kp2: {numKeyPoints2}")

    matches = []

    for x in range(numKeyPoints1):
        distance = -1
        y_ind = -1
        for y in range(numKeyPoints2):

            out1_norm = (desc1[x] - np.mean(desc1[x])) / (np.std(desc1[x]))
            out2_norm = (desc2[y] - np.mean(desc2[y])) / (np.std(desc2[y]))

            corr = np.mean(np.multiply(out1_norm, out2_norm))

            # corr = np.mean(np.multiply(out1_zero_mean, out2_zero_mean)) / (out1_std * out2_std)
            if corr > distance:
                distance = corr
                y_ind = y

        cur = cv2.DMatch()
        cur.queryIdx = x
        cur.trainIdx = y_ind
        print(f"distance: {distance}")
        cur.distance = distance
        matches.append(cur)
    # TODO-BLOCK-END

    return matches


def main():
    # read images
    img1 = cv2.imread("../resources/Images/cat512.jpg")
    img2 = cv2.imread("../resources/Images/cat512_edited_v2.png")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # pip install opencv-contrib-python

    # sift
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # feature matching
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # matches = bf.match(descriptors_1, descriptors_2)
    # matches = sorted(matches, key=lambda x: x.distance)

    # matches_2 = matchFeatures_SSD(descriptors_1, descriptors_2)
    # matches_2 = sorted(matches_2, key=lambda x: x.distance)

    # matches_3 = matchFeatures_Ratio_Test(descriptors_1, descriptors_2)
    # matches_3 = sorted(matches_3, key=lambda x: x.distance)

    # matches_4 = normalized_cross_correlation(descriptors_1, descriptors_2)

    matches_4 = match_features(descriptors_1, descriptors_2, calculate_ncc)
    matches_4 = sorted(matches_4, key=lambda x: x.distance, reverse=True)

    # print(f"matches_4 {matches_4}")
    # print(f"matches_4.shape {len(matches_4)}")

    # img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
    #                        matches[:30], img2, flags=2)
    # img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
    #                        matches_2[:30], img2, flags=2)
    # img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
    #                        matches_3[:30], img2, flags=2)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                           matches_4[:30], img2, flags=2)

    plt.imshow(img3), plt.show()

    # matched_image = apply_feature_matching()


if __name__ == "__main__":
    main()

from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np


def apply_feature_matching(desc1: np.ndarray, desc2: np.ndarray,
                           match_calculator: Callable[[list, list], float]) -> list:
    """

    Perform feature matching between 2 feature descriptors

    :param desc1: The feature descriptors of image 1.
                  Dimensions: rows (number of key points) x columns (dimension of the feature descriptor i.e: 128)
    :param desc2: The feature descriptors of image 2.
                  Dimensions: rows (number of key points) x columns (dimension of the feature descriptor i.e: 128)
    :param match_calculator: A Callable function to use in matching features:
                        - calculate_ssd (Sum Square Distance)
                        - calculate_ncc (Normalized Cross Correlation)

    :return: features matches, a list of cv2.DMatch objects
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
    num_key_points1 = desc1.shape[0]
    num_key_points2 = desc2.shape[0]

    # List to store the matches scores
    matches = []

    # Loop over each key point in image1
    # We need to calculate similarity with each key point in image2
    for kp1 in range(num_key_points1):
        # Initial variables which will be updated in the loop
        distance = -np.inf
        y_index = -1

        # Loop over each key point in image2
        for kp2 in range(num_key_points2):

            # Match features between the 2 vectors
            value = match_calculator(desc1[kp1], desc2[kp2])

            # SSD values examples: (50, 200, 70), we need to minimize SSD (Choose 50)
            # In case of SSD matching: (value is returned as a "negative" number) (-50, -200, -70)
            # So we compare it with -np.inf. (The sorting will be reversed later)

            # NCC values examples: (0.58, 0.87, 0.42), we need to maximize NCC (Choose 0.87)
            # In case of NCC matching: (value is returned as a "positive" number) (-58, -0.87, -0.42)
            # So we compare it with -np.inf. (The sorting will be reversed later)

            if value > distance:
                distance = value
                y_index = kp2

        # Create a cv2.DMatch object for each match and set attributes:
        # queryIdx: The index of the feature in the first image
        # trainIdx: The index of the feature in the second image
        # distance: The distance between the two features
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

    sum_square = 0

    # Get SSD between the 2 vectors
    for m in range(len(desc1)):
        sum_square += (desc1[m] - desc2[m]) ** 2

    # The (-) sign here because the condition we applied after this function call is reversed
    sum_square = - (np.sqrt(sum_square))

    return sum_square


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
    """
    This function is responsible for:
        - Calculating the ratio of the SSD distance of the two best matches
        - Matching a feature in the first image with the closest feature in the second image.

    Note:
        - multiple features from the first image may match the same feature in the second image.
        - You don't need to threshold matches in this function

    :param desc1: The feature descriptor vector of one key point in image1.
                  Dimensions: rows (1) x columns (dimension of the feature descriptor i.e: 128)
    :param desc2: The feature descriptor vector of one key point in image2.
                  Dimensions: rows (1) x columns (dimension of the feature descriptor i.e: 128)
    :return: A float number represent the SSD between two features vectors
    """

    raise NotImplementedError


def match_features_ratio_test(desc1, desc2):
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

    # feature count = n
    assert desc1.ndim == 2
    # feature count = m
    assert desc2.ndim == 2
    # the two features should have the type
    assert desc1.shape[1] == desc2.shape[1]

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return []

    num_key_points1 = desc1.shape[0]
    num_key_points2 = desc2.shape[0]
    matches = []

    for x in range(num_key_points1):
        distance1 = -1
        distance2 = -1
        y_ind = -1
        for y in range(num_key_points2):
            sum_square = 0
            for m in range(desc1.shape[1]):
                sum_square += (desc1[x][m] - desc2[y][m]) ** 2
            sum_square = np.sqrt(sum_square)
            if distance1 < 0 or (sum_square < distance1 and distance1 >= 0):
                distance2 = distance1
                distance1 = sum_square
                y_ind = y
            elif distance2 < 0 or (sum_square < distance2 and distance2 >= 0):
                distance2 = sum_square
        cur = cv2.DMatch()
        cur.queryIdx = x
        cur.trainIdx = y_ind
        cur.distance = distance1 / distance2
        matches.append(cur)

    return matches


def main():
    # read images
    img1 = cv2.imread("../resources/Images/cat256.jpg")
    img2 = cv2.imread("../resources/Images/cat512_edited_v3.png")
    # img2 = cv2.imread("../resources/Images/cat256_edited_v2.png")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # sift
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    # matches = apply_feature_matching(descriptors_1, descriptors_2, calculate_ncc)
    # matches = sorted(matches, key=lambda x: x.distance, reverse=True)

    # matches = sorted(matches, key=lambda x: x.distance, reverse=True)

    matched_image = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                                    matches[:30], img2, flags=2)

    plt.imshow(matched_image)
    plt.show()


if __name__ == "__main__":
    main()

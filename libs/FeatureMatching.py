import cv2
import numpy as np
from libs.Harris import apply_harris_operator2
import matplotlib.pyplot as plt

def apply_feature_matching(desc1: np.ndarray, desc2: np.ndarray) -> np.ndarray:
    """

    :param source:
    :param template:
    :return:
    """

    out = desc1


    return out


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

    matches = []
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
    matches = []

    for x in range(numKeyPoints1):
        distance = -1
        y_ind = -1
        for y in range(numKeyPoints2):
            sumSquare = 0
            for m in range(desc1.shape[1]):
                sumSquare += (desc1[x][m] - desc2[y][m]) ** 2
            sumSquare = np.sqrt(sumSquare)
            if distance < 0 or (sumSquare < distance and distance >= 0):
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
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # matches = bf.match(descriptors_1, descriptors_2)
    # matches = sorted(matches, key=lambda x: x.distance)

    # matches_2 = matchFeatures_SSD(descriptors_1, descriptors_2)
    # matches_2 = sorted(matches_2, key=lambda x: x.distance)

    matches_3 = matchFeatures_Ratio_Test(descriptors_1, descriptors_2)
    matches_3 = sorted(matches_3, key=lambda x: x.distance)

    # print(f"matches_2 {matches_2}")
    # print(f"matches_2.shape {len(matches_2)}")

    # img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
    #                        matches[:30], img2, flags=2)
    # img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
    #                        matches_2[:30], img2, flags=2)
    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                           matches_3[:30], img2, flags=2)
    plt.imshow(img3), plt.show()

    # matched_image = apply_feature_matching()


if __name__ == "__main__":
    main()
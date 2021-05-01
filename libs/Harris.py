import cv2
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from EdgeDetection import sobel_edge
from LowPass import gaussian_filter
# from scipy.ndimage import gaussian_filter

from Contour import GenerateWindowCoordinates

def apply_harris_operator(source: np.ndarray) -> (np.ndarray, np.ndarray):
    """

    :param source:
    :return:
    """

    src = np.copy(source)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Get Gradient of image
    I_x, I_y = sobel_edge(source=src, GetMagnitude=False)

    Ixx = gaussian_filter(source=I_x ** 2, sigma=255)
    Ixy = gaussian_filter(source=I_y * I_x, sigma=255)
    Iyy = gaussian_filter(source=I_y ** 2, sigma=255)

    # Ixx = cv2.GaussianBlur(src=I_x **2, ksize=(5, 5), sigmaX=0)
    # Ixy = cv2.GaussianBlur(src=I_y * I_x, ksize=(5, 5), sigmaX=0)
    # Iyy = cv2.GaussianBlur(src=I_y ** 2, ksize=(5, 5), sigmaX=0)

    # This is H Matrix
    # [ Ix^2        Ix * Iy ]
    # [ Ix * Iy     Iy^2    ]

    # Harris Response R
    # R = det(H) - k(trace(H))^2
    # The response R is an array of peak values of each row in the image.
    # We can use these peak values to isolate corners and edges
    # Edge   : R < 0
    # Corner : R > 0
    # Flat   : R = 0

    # k is a sensitivity factor to separate corners from edges
    # Small values of k result in detection of sharp corners
    k = 0.05

    # where A, B and C are shifts of window defined by w.
    # The lambdas are the Eigen values of H

    det_H   = Ixx * Iyy - Ixy ** 2
    trace_H = Ixx + Iyy

    harris_response = det_H - k * (trace_H ** 2)

    img_corners = np.copy(source)
    img_edges = np.copy(source)

    for rowindex, response in enumerate(harris_response):
        for colindex, r in enumerate(response):
            if r > 0:
                # this is a corner
                img_corners[rowindex, colindex] = [255, 0, 0]
            elif r < 0:
                # this is an edge
                img_edges[rowindex, colindex] = [0, 255, 0]

    return img_corners, img_edges

def get_harris_corners(source: np.ndarray):
    """

    :param source:
    :return:
    """
    return source


def display_image(src):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(src)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, src.shape[1])
    ax.set_ylim(src.shape[0], 0)
    plt.show()


def main():

    img = cv2.imread("../resources/Images/harris_image_400.jpg")

    img_corners, img_edges = apply_harris_operator(img)

    display_image(img_corners)
    display_image(img_edges)

    # Create neighborhood window
    WindowCoordinates = GenerateWindowCoordinates(5)

    # h_mat = np.array([[10, 20], [50, 60]])
    # print(np.trace(h_mat))
    # print(linalg.det(h_mat))
    # eigen_values, eigen_vectors = np.linalg.eig(h_mat)
    # print(eigen_values)
    # print(eigen_vectors)

if __name__ == "__main__":
    main()

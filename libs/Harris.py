import cv2
import numpy as np
import matplotlib.pyplot as plt

from libs.EdgeDetection import sobel_edge
from libs.LowPass import gaussian_filter


def apply_harris_operator(source: np.ndarray, k: float = 0.05) -> np.ndarray:
    """

    :param source:
    :param k: sensitivity factor to separate corners from edges.
              Small values of k result in detection of sharp corners
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

    # where A, B and C are shifts of window defined by w.
    # The lambdas are the Eigen values of H

    det_H   = Ixx * Iyy - Ixy ** 2
    trace_H = Ixx + Iyy

    harris_response = det_H - k * (trace_H ** 2)

    return harris_response

def apply_harris_operator2(source: np.ndarray, k: float = 0.03, window_size: int = 3) -> np.ndarray:
    """

    :param source: image source
    :param k: sensitivity factor to separate corners from edges
    :return:
    """

    src = np.copy(source)
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # I_x, I_y = sobel_edge(source=src, GetMagnitude=False)

    I_x = cv2.Sobel(src, cv2.CV_64F, 1, 0, ksize=5)
    I_y = cv2.Sobel(src, cv2.CV_64F, 0, 1, ksize=5)

    # Ixx = gaussian_filter(source=I_x ** 2, sigma=1)
    # Ixy = gaussian_filter(source=I_y * I_x, sigma=1)
    # Iyy = gaussian_filter(source=I_y ** 2, sigma=1)

    Ixx = cv2.GaussianBlur(src=I_x ** 2, ksize=(5, 5), sigmaX=0)
    Ixy = cv2.GaussianBlur(src=I_y * I_x, ksize=(5, 5), sigmaX=0)
    Iyy = cv2.GaussianBlur(src=I_y ** 2, ksize=(5, 5), sigmaX=0)


    height, width = src.shape
    harris_response = []
    offset = int(window_size/2)

    # Loop over each column in the image
    for y in range(offset, height-offset):
        # Loop over each row in the image
        for x in range(offset, width-offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])

            # Find determinant and trace, use to get corner response
            det   = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r     = det - k*(trace**2)

            harris_response.append(r)

    # Convert response from list to numpy array
    new_width = src.shape[0]-(window_size-offset)
    new_height = src.shape[1]-(window_size-offset)
    harris_response = np.array(harris_response).reshape((new_width, new_height))

    return harris_response

def get_harris_indices(harris_response: np.ndarray, threshold: float = 0.01):
    """

    :param harris_response:
    :return:
    """

    harris_copy = np.copy(harris_response)
    harris_matrix = cv2.dilate(harris_copy, None)
    max_response = np.max(harris_matrix)

    # Indices of each corner, edges and flat area
    # Threshold for an optimal value, it may vary depending on the image.
    corner_indices = np.array(harris_matrix > (max_response*threshold), dtype="int8")
    edges_indices  = np.array(harris_matrix < (max_response*threshold), dtype="int8")
    flat_indices   = np.array(harris_matrix == (max_response*threshold), dtype="int8")

    # Threshold for an optimal value, it may vary depending on the image.
    # img[dst > 0.01 * dst.max()] = [0, 0, 255]

    return corner_indices, edges_indices, flat_indices

def map_indices_to_image(source: np.ndarray, indices: np.ndarray, color: list):
    """

    :param source:
    :param indices:
    :return:
    """

    src = np.copy(source)

    # Make sure that the original source shape == indices shape
    src = src[:indices.shape[0], :indices.shape[1]]

    # Mark each index with red dot
    src[indices == 1] = color

    return src

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

    # img = cv2.imread("../resources/Images/harris_image_400.jpg")
    #
    # img_corners, img_edges = apply_harris_operator(img)
    #
    # display_image(img_corners)
    # display_image(img_edges)

    # h_mat = np.array([[10, 20], [50, 60]])
    # print(np.trace(h_mat))
    # print(linalg.det(h_mat))
    # eigen_values, eigen_vectors = np.linalg.eig(h_mat)
    # print(eigen_values)
    # print(eigen_vectors)

    filename = "../resources/Images/cow_step_harris.png"
    # filename = "../resources/Images/bill.png"
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    print(f"dst shape: {dst.shape}")
    print(f"dst max: {dst.max()}")
    print(f"dst min: {dst.min()}")

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

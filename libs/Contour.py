import cv2
import numpy as np
import matplotlib.pyplot as plt
from EdgeDetection import sobel_edge
from LowPass import gaussian_filter


def active_contour(source: np.ndarray, alpha: float, beta: float, gamma: float, num_iterations: int,
                   num_points: int = 12) -> np.ndarray:
    # TODO Apply Active Contour algorithm

    contour_x, contour_y, = create_initial_contour(source, 65)
    ext_energy = external_energy(source)

    # TODO Add loops for greedy algorithm from "ZhangK Paper"

    


def create_initial_contour(source, num_points):
    """
        Represent the snake with a set of n points
        Vi = (Xi, Yi) , where i = 0, 1, ... n-1

    :param num_points:
    :return:
    """

    t = np.arange(0, num_points/10, 0.1)
    contour_x = (source.shape[0] // 2) + 160 * np.cos(t)
    contour_y = (source.shape[1] // 2) + 245 * np.sin(t)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(source, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, source.shape[1])
    ax.set_ylim(source.shape[0], 0)
    ax.plot(np.r_[contour_x, contour_x[0]],
            np.r_[contour_y, contour_y[0]], c=(0, 1, 0), lw=2)

    plt.show()

    return contour_x, contour_y


def compute_total_energy():
    pass


def internal_energy(flattened_pts, alpha: int, beta: int):
    """
    The internal energy is responsible for:
        1. Forcing the contour to be continuous (E_cont)
        2. Forcing the contour to be smooth     (E_curv)
        3. Deciding if the snake wants to shrink/expand

    Internal Energy Equation:
        E_internal = E_cont + E_curv

    E_cont
        alpha * ||dc/ds||^2

        - Minimizing the first derivative.
        - The contour is approximated by N points P1, P2, ..., Pn.
        - The first derivative is approximated by a finite difference:

        E_cont = | (Vi+1 - Vi) | ^ 2
        E_cont = (Xi+1 - Xi)^2 + (Yi+1 - Yi)^2

    E_curv
        beta * ||d^2c / d^2s||^2

        - Minimizing the second derivative
        - We want to penalize if the curvature is too high
        - The curvature can be approximated by the following finite difference:

        E_curv = (Xi-1 - 2Xi + Xi+1)^2 + (Yi-1 - 2Yi + Yi+1)^2

    ==============================

    Alpha and Beta
        - Small alpha make the energy function insensitive to the amount of stretch
        - Big alpha increases the internal energy of the snake as it stretches more and more

        - Small beta causes snake to allow large curvature so that snake will curve into bends in the contour
        - Big beta leads to high price for curvature so snake prefers to be smooth and not curving

    :return:
    """

    pts = np.reshape(flattened_pts, (int(len(flattened_pts) / 2), 2))

    # spacing energy (favors equi-distant points)
    prev_pts = np.roll(pts, 1, axis=0)
    next_pts = np.roll(pts, -1, axis=0)
    displacements = pts - prev_pts
    point_distances = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
    mean_dist = np.mean(point_distances)
    spacing_energy = np.sum((point_distances - mean_dist) ** 2)

    # curvature energy (favors smooth curves)
    curvature_1d = prev_pts - 2 * pts + next_pts
    curvature = (curvature_1d[:, 0] ** 2 + curvature_1d[:, 1] ** 2)
    curvature_energy = np.sum(curvature)

    return  alpha * spacing_energy + beta * curvature_energy


def external_energy(source):
    """
    The External Energy is responsible for:
        1. Attracts the contour towards the closest image edge with dependence on the energy map.
        2. Determines whether the snake feels attracted to object boundaries

    An energy map is a function f (x, y) that we extract from the image – I(x, y):

        By given an image – I(x, y), we can build an energy map – f(x, y),
        that will attract our snake to edges on our image.

    External Energy Equation:
        E_external = w_line * E_line + w_edge * E_edge


    E_line
        I(x, y)

        Depending on the sign of w_line the snake will be attracted either to bright lines or dark lines


    E_curv
        -|| Gradiant(I(x,y)) ||^2

    ==============================

    :param source: Image source
    :return:
    """

    # Calculate E_edge
    # Convert to gray Scale
    # gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Filter to smooth the image
    filtered_image = gaussian_filter(source, 3, 9)

    # Get Gradient Magnitude & Direction
    gradient_magnitude, gradient_direction = sobel_edge(filtered_image, True)
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    e_edge = gradient_magnitude

    # TODO Calculate E_line

    # TODO return E_external = w_line * E_line + w_edge * E_edge
    e_external = None

    return e_external


def main():
    """
    the application startup functions
    :return:
    """

    alpha = 0.001
    beta = 0.4
    gamma = 100
    iterations = 50

    img = cv2.imread("../src/Images/pepsi_can.png", 0)
    active_contour(img, alpha, beta, gamma, iterations)


if __name__ == "__main__":
    main()

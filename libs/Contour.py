import itertools
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from EdgeDetection import sobel_edge
from LowPass import gaussian_filter


def active_contour(source: np.ndarray, contour_x: np.ndarray, contour_y: np.ndarray,
                   alpha: float, beta: float, gamma: float,
                   w_line, w_dge, num_iterations: int) -> Tuple[np.ndarray, np.ndarray]:
    """

    :param source:
    :param alpha:
    :param beta:
    :param gamma:
    :param w_line:
    :param w_dge:
    :param num_iterations:
    :param num_points:
    :return:
    """

    cont_x = np.copy(contour_x)
    cont_y = np.copy(contour_y)

    ExternalEnergy = gamma * external_energy(source, w_line, w_dge)
    # WindowCoordinates = [[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, 1], [2, 2]]
    WindowCoordinates = GenerateWindowCoordinates(5)
    contour_points = len(cont_x)

    for Iteration in range(num_iterations):
        for Point in range(contour_points):
            MinEnergy = np.inf
            NewX = None
            NewY = None
            for Window in WindowCoordinates:
                # Create Temporary Contours With Point Shifted To A Coordinate
                CurrentX, CurrentY = np.copy(cont_x), np.copy(cont_y)
                CurrentX[Point] = CurrentX[Point] + Window[0] if CurrentX[Point] < source.shape[1] else source.shape[
                                                                                                            1] - 1
                CurrentY[Point] = CurrentY[Point] + Window[1] if CurrentY[Point] < source.shape[0] else source.shape[
                                                                                                            0] - 1
                # Calculate Energy At The New Point
                TotalEnergy = - ExternalEnergy[CurrentY[Point], CurrentX[Point]] + internal_energy(CurrentX, CurrentY,
                                                                                                   alpha, beta)
                # Save The Point If It Has The Lowest Energy In The Window
                if TotalEnergy < MinEnergy:
                    MinEnergy = TotalEnergy
                    NewX = CurrentX[Point] if CurrentX[Point] < source.shape[1] else source.shape[1] - 1
                    NewY = CurrentY[Point] if CurrentY[Point] < source.shape[0] else source.shape[0] - 1

            # Shift The Point In The Contour To It's New Location With The Lowest Energy
            cont_x[Point] = NewX
            cont_y[Point] = NewY

    return cont_x, cont_y


def create_initial_contour(source, num_points):
    """
        Represent the snake with a set of n points
        Vi = (Xi, Yi) , where i = 0, 1, ... n-1
    :param source: Image Source
    :param num_points:
    :return:
    """

    t = np.arange(0, num_points / 10, 0.1)
    contour_x = (source.shape[1] // 2) + 110 * np.cos(t) - 100
    contour_y = (source.shape[0] // 2) + 110 * np.sin(t) + 50
    contour_x = contour_x.astype(int)
    contour_y = contour_y.astype(int)
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

def GenerateWindowCoordinates(Size: int):
    """
    Generates A List of All Possible Coordinates Inside A Window of Size "Size"
    :param Size: Size of The Window
    :return Coordinates: List of All Possible Coordinates
    """
    # Generate List of All Possible Point Values Based on Size
    Points = list(range(-Size // 2 + 1, Size // 2 + 1))
    PointsList = [Points, Points]
    # Generates All Possible Coordinates Inside The Window
    Coordinates = list(itertools.product(*PointsList))
    return Coordinates


def internal_energy(CurrentX, CurrentY, alpha: float, beta: float):
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
    JoinedXY = np.array((CurrentX, CurrentY))
    Points = JoinedXY.T

    # Continuous  Energy
    PrevPoints = np.roll(Points, 1, axis=0)
    NextPoints = np.roll(Points, -1, axis=0)
    Displacements = Points - PrevPoints
    PointDistances = np.sqrt(Displacements[:, 0] ** 2 + Displacements[:, 1] ** 2)
    MeanDistance = np.mean(PointDistances)
    ContinuousEnergy = np.sum((PointDistances - MeanDistance) ** 2)

    # Curvature Energy
    CurvatureSeparated = PrevPoints - 2 * Points + NextPoints
    Curvature = (CurvatureSeparated[:, 0] ** 2 + CurvatureSeparated[:, 1] ** 2)
    CurvatureEnergy = np.sum(Curvature)

    return alpha * ContinuousEnergy + beta * CurvatureEnergy


def external_energy(source, WLine, WEdge):
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

    # convert to gray scale if not already
    if len(source.shape) > 2:
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    else:
        gray = source

    # Apply Gaussian Filter to smooth the image
    ELine = gaussian_filter(gray, 7, 7 * 7)

    # Get Gradient Magnitude & Direction
    EEdge, gradient_direction = sobel_edge(ELine, True)
    # EEdge *= 255 / EEdge.max()
    # EEdge = EEdge.astype("int16")

    return WLine * ELine + WEdge * EEdge[1:-1, 1:-1]


def main():
    """
    the application startup functions
    :return:
    """
    # Continuous
    alpha = 19.86
    # Curvature
    beta = 30.7
    gamma = 50
    iterations = 50
    WLine = 1
    WEdge = 1

    img = cv2.imread("../src/Images/circles_v2.png", 0)
    contour_x, contour_y = create_initial_contour(source=img, num_points=65)
    cont_x, cont_y = active_contour(source=img, contour_x=contour_x, contour_y=contour_y,
                                            alpha=alpha, beta=beta, gamma=gamma, w_line=WLine,
                                            w_dge=WEdge, num_iterations=iterations)

    # active_contour(img, alpha, beta, gamma, WLine, WEdge, iterations, )

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.plot(np.r_[cont_x, cont_x[0]],
            np.r_[cont_y, cont_y[0]], c=(0, 1, 0), lw=2)

    plt.show()

if __name__ == "__main__":
    main()

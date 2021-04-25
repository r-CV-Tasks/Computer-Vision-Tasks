import cv2
import matplotlib.pyplot as plt
import numpy as np

from EdgeDetection import sobel_edge
from LowPass import gaussian_filter


def active_contour(source: np.ndarray, alpha: float, beta: float, gamma: float, WLine, WEdge, num_iterations: int,
                   num_points: int = 12) -> np.ndarray:
    contour_x, contour_y, = create_initial_contour(source, 65)
    Grad = external_energy(source, WLine, WEdge)
    ExternalEnergy = gamma * external_energy(source, WLine, WEdge)
    WindowCoordinates = [[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, 1]]
    # TODO Fix The Code
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(Grad, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, Grad.shape[1])
    ax.set_ylim(Grad.shape[0], 0)
    ax.plot(np.r_[contour_x, contour_x[0]],
            np.r_[contour_y, contour_y[0]], c=(0, 1, 0), lw=2)

    plt.show()
    for n in range(num_iterations):
        for i in range(len(contour_x)):
            MinEnergy = None
            NewX = None
            NewY = None
            for k in WindowCoordinates:
                CurrentX, CurrentY = np.copy(contour_x), np.copy(contour_y)
                CurrentX[i] = CurrentX[i] + k[0] if CurrentX[i] < 511 else 511
                CurrentY[i] = CurrentY[i] + k[1] if CurrentY[i] < 511 else 511
                TotalEnergy = - ExternalEnergy[CurrentX[i], CurrentY[i]] + internal_energy(CurrentX, CurrentY, alpha,
                                                                                           beta)
                if MinEnergy is None:
                    MinEnergy = TotalEnergy
                    NewX = CurrentX[i] if CurrentX[i] < 512 else 511
                    NewY = CurrentY[i] if CurrentY[i] < 512 else 511
                if TotalEnergy < MinEnergy:
                    MinEnergy = TotalEnergy
                    NewX = CurrentX[i] if CurrentX[i] < 512 else 511
                    NewY = CurrentY[i] if CurrentY[i] < 512 else 511
            contour_x[i] = NewX
            contour_y[i] = NewY

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



def create_initial_contour(source, num_points):
    """
        Represent the snake with a set of n points
        Vi = (Xi, Yi) , where i = 0, 1, ... n-1

    :param num_points:
    :return:
    """

    t = np.arange(0, num_points / 10, 0.1)
    contour_x = (source.shape[0] // 2) + 160 * np.cos(t)
    contour_y = (source.shape[1] // 2) + 245 * np.sin(t)
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


def compute_total_energy():
    pass


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

    # Calculate E_edge
    # Convert to gray Scale
    # gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Filter to smooth the image
    ELine = gaussian_filter(source, 5, 25)

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

    alpha = 0.06
    beta = 0.7
    gamma = 30
    iterations = 50
    WLine = 0
    WEdge = 4

    img = cv2.imread("../src/Images/pepsi_can.png", 0)
    active_contour(img, alpha, beta, gamma, WLine, WEdge, iterations)


if __name__ == "__main__":
    main()


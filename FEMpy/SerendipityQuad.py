"""
==============================================================================
8 Node Serendipity Quad Element
==============================================================================
@File    :   SerendipityQuad.py
@Date    :   2021/03/30
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from numba import njit

# ==============================================================================
# Extension modules
# ==============================================================================
from .QuadElement import QuadElement


@njit(cache=True)
def serendipityShapeFuncs(x, y):
    """Compute the shape functions of an 8 node serendipity Quad element



    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        x coordinates of points to compute polynomial values at, should be between -1.0 and 1.0
    y : array of length nP (0D, nPx1 or 1xnP)
        y coordinates of points to compute polynomial values at, should be between -1.0 and 1.0

    Returns
    -------
    N : nP x n array
        Shape function values
    """
    psi = x.flatten()
    eta = y.flatten()
    N = np.zeros((len(psi), 8), dtype=x.dtype)

    N[:, 0] = 0.25 * (1.0 - psi) * (1.0 - eta) * (-psi - eta - 1.0)
    N[:, 1] = 0.25 * (1.0 + psi) * (1.0 - eta) * (psi - eta - 1.0)
    N[:, 2] = 0.25 * (1.0 + psi) * (1.0 + eta) * (psi + eta - 1.0)
    N[:, 3] = 0.25 * (1.0 - psi) * (1.0 + eta) * (-psi + eta - 1.0)
    N[:, 4] = 0.5 * (1.0 - psi ** 2) * (1.0 - eta)
    N[:, 5] = 0.5 * (1.0 + psi) * (1.0 - eta ** 2)
    N[:, 6] = 0.5 * (1.0 - psi ** 2) * (1.0 + eta)
    N[:, 7] = 0.5 * (1.0 - psi) * (1.0 - eta ** 2)

    return N


@njit(cache=True)
def serendipityShapeFuncDerivs(x, y):
    """Compute the derivatives of the shape functions of an 8 node serendpity Quad element at a series of points in 2d space



    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        x coordinates of points to compute polynomial values at, should be between -1 and 1
    y : array of length nP (0D, nPx1 or 1xnP)
        y coordinates of points to compute polynomial values at, should be between -1 and 1s

    Returns
    -------
    dNdx : nP x 2 x n array
        Shape function derivative values
    """
    psi = x.flatten()
    eta = y.flatten()
    dNdxy = np.zeros((len(psi), 2, 8))

    # --- dNdPsi ---
    dNdxy[:, 0, 0] = 0.25 * (1.0 - eta) * (2.0 * psi + eta)
    dNdxy[:, 0, 1] = 0.25 * (1.0 - eta) * (2.0 * psi - eta)
    dNdxy[:, 0, 2] = 0.25 * (1.0 + eta) * (2.0 * psi + eta)
    dNdxy[:, 0, 3] = 0.25 * (1.0 + eta) * (2.0 * psi - eta)
    dNdxy[:, 0, 4] = -psi * (1.0 - eta)
    dNdxy[:, 0, 5] = 0.5 * (1 - eta ** 2)
    dNdxy[:, 0, 6] = -psi * (1.0 + eta)
    dNdxy[:, 0, 7] = -0.5 * (1 - eta ** 2)

    # --- dNdEta ---
    dNdxy[:, 1, 0] = 0.25 * (1.0 - psi) * (2 * eta + psi)
    dNdxy[:, 1, 1] = 0.25 * (1.0 + psi) * (2 * eta - psi)
    dNdxy[:, 1, 2] = 0.25 * (1.0 + psi) * (2 * eta + psi)
    dNdxy[:, 1, 3] = 0.25 * (1.0 - psi) * (2 * eta - psi)
    dNdxy[:, 1, 4] = -0.5 * (1.0 - psi ** 2)
    dNdxy[:, 1, 5] = -eta * (1.0 + psi)
    dNdxy[:, 1, 6] = 0.5 * (1.0 - psi ** 2)
    dNdxy[:, 1, 7] = -eta * (1.0 - psi)

    return dNdxy


class serendipityQuadElement(QuadElement):
    """An 8 noded quadratic quad element, also known as a seredipity quad"""

    def __init__(self, numDisplacements=2):
        # --- Initialise a 9 noded quad and then change the number of nodes to 8 ---
        super().__init__(order=2, numDisplacements=numDisplacements)
        self.numNodes = 8
        self.numDOF = self.numNodes * numDisplacements
        self.name = "SerendipityQuad"

    def getShapeFunctions(self, paramCoords):
        """Compute shape function values at a set of parametric coordinates

        Parameters
        ----------
        paramCoords : n x nDim array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        N : n x numNode array
            Shape function values, N[i][j] is the value of the jth shape function at the ith point
        """
        return serendipityShapeFuncs(paramCoords[:, 0], paramCoords[:, 1])

    def getShapeFunctionDerivs(self, paramCoords):
        """Compute shape function derivatives at a set of parametric coordinates

        These are the derivatives of the shape functions with respect to the parametric coordinates (si, eta, gamma)

        Parameters
        ----------
        paramCoords : n x nD array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        NPrime : n x numDim x numNode array
            Shape function derivative values, N[i][j][k] is the value of the derivative of the kth shape function in the
            jth direction at the ith point
        """
        return serendipityShapeFuncDerivs(paramCoords[:, 0], paramCoords[:, 1])

    def getRandomNodeCoords(self):
        """Generate a random, but valid, set of node coordinates for an element

        Here we simply call the getRandomNodeCoords method of the parent 2nd order QuadElement class and then remove
        the central point, the points need to be reordered too because the 2nd order QuadElement points are not in CCW
        order as the Serendipity quad points are

        Returns
        -------
        nodeCoords : numNode x numDim array
            Node coordinates
        """
        nodeCoords = super().getRandomNodeCoords()
        return nodeCoords[[0, 2, 8, 6, 1, 5, 7, 3]]


if __name__ == "__main__":
    import time
    import niceplots
    import matplotlib.pyplot as plt

    niceplots.setRCParams()
    # ==============================================================================
    # Speed test
    # ==============================================================================
    x = np.array([np.linspace(-1.0, 1.0, 3)])
    y = -np.ones_like(x)
    # Call functions so they get jit compiled
    serendipityShapeFuncs(x, y)
    serendipityShapeFuncDerivs(x, y)
    startTime = time.time()
    for _ in range(1000):
        serendipityShapeFuncs(x, y)
        # print("\n")
        serendipityShapeFuncDerivs(x, y)
        # print("\n\n\n")
    print(time.time() - startTime)

    # ==============================================================================
    # Plot shape functions
    # ==============================================================================
    n = 101
    psi, eta = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(-1.0, 1.0, n))
    N = serendipityShapeFuncs(psi.flatten(), eta.flatten())

    fig, axes = plt.subplots(nrows=2, ncols=4, subplot_kw={"projection": "3d"})
    axes = axes.flatten()
    for i in range(8):
        ax = axes[i]
        Ni = N[:, i].reshape((n, n))
        ax.plot_surface(psi, eta, Ni, cmap=niceplots.parula_map)
        ax.set_title(f"N{i}")
    plt.show()

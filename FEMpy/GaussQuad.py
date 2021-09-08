"""
==============================================================================
Gauss Quadrature Points and Weights
==============================================================================
@File    :   guassQuad.py
@Date    :   2021/03/10
@Author  :   Alasdair Christison Gray
@Description : This file contains the coordinates and weights for numerical integration using Gauss quadrature on 1D
intervals, 2D quads and triangles and 3d hexahedrons
"""

import numpy as np
import pickle
import os

# --- Load the Gauss quadrature weights and coordinates ---
dataDir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dataDir, "GaussQuadWeights.pkl"), "rb") as f:
    gaussWeights = pickle.load(f)
with open(os.path.join(dataDir, "GaussQuadCoords.pkl"), "rb") as f:
    gaussCoords = pickle.load(f)

# --- Define gauss quadrature points and weights for triangles ---
TriGaussPoints = {}
TriGaussPoints[1] = np.array([[1.0 / 3.0, 1.0 / 3.0]])
TriGaussPoints[2] = np.array([[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]])
TriGaussPoints[3] = np.array([[1.0 / 3.0, 1.0 / 3.0], [0.2, 0.2], [0.6, 0.2], [0.2, 0.6]])
TriGaussPoints[4] = np.array(
    [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.0, 0.5], [1.0 / 3.0, 1.0 / 3.0]]
)

TriGaussWeights = {}
TriGaussWeights[1] = np.array([0.5])
TriGaussWeights[2] = 1.0 / 6.0 * np.ones(3)
TriGaussWeights[3] = np.array([-9.0 / 32.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
TriGaussWeights[4] = np.array([0.025, 1.0 / 15.0, 0.025, 1.0 / 15.0, 0.025, 1.0 / 15.0, 0.225])


def getgaussWeights(n):
    """Get the weights for n-point numerical integration using Gauss Quadrature

    Gauss Quadrature integration with n points will exactly integrate polynomials of order <= 2n-1

    Values generated using numpy's leggauss function

    Parameters
    ----------
    n : int
        number of integration points

    Returns
    -------
    array
        array of n point weights
    """
    return gaussWeights[n - 1]


def getGaussPoints(n):
    """Get the coordinates for n-point numerical integration using Gauss Quadrature on the interval (-1, 1))

    Gauss Quadrature integration with n points will exactly integrate polynomials of order <= 2n-1

    Values generated using numpy's leggauss function

    Parameters
    ----------
    n : int
        number of integration points

    Returns
    -------
    array
        array of n point coordinates
    """
    return gaussCoords[n - 1]


def gaussQuad1d(f, n, a=-1.0, b=1.0):
    """Perform a one dimensional integration using Gauss Quadrature

    Parameters
    ----------
    f : function
        The function to be integrated, should be able to accept vectorised input
    n : int
        Number of points to use
    a : float, optional
        Lower limit of integration, by default -1.0
    b : float, optional
        Upper limit of integration, by default 1.0

    Returns
    -------
    return type of f
        The integrand
    """
    scale = 0.5 * (b - a)
    offset = 0.5 * (b + a)
    xGauss = getGaussPoints(n)
    xReal = scale * xGauss + offset
    W = getgaussWeights(n)

    return scale * np.sum(f(xReal).T * W, axis=-1).T


def gaussQuad2d(f, n, a=-1.0, b=1.0):
    """Perform a two dimensional integration using Gauss Quadrature

    Parameters
    ----------
    f : function
        The function to be integrated, should be able to accept vectorised input in each dimension
    n : int or list of ints
        Number of gauss quadrature points, can be different in each dimension
    a : float or list of floats, optional
        Lower integration limit can be different in each dimension, by default -1.0 for all dimensions
    b : float or lists of floats, optional
        Upper integration limit can be different in each dimension, by default 1.0 for all dimensions

    Returns
    -------
    return type of f
        The integrand
    """

    if isinstance(n, int):
        n = [n] * 2
    if isinstance(a, float):
        a = [a] * 2
    if isinstance(b, float):
        b = [b] * 2

    Eta = []
    W = []
    scale = []
    offset = []
    for i in range(1):
        Eta.append(np.array(getGaussPoints(n[i])))
        W.append(np.array(getgaussWeights(n[i])))
        scale.append(np.array(0.5 * (b[i] - a[i])))
        offset.append(np.array(0.5 * (a[i] + b[i])))

    # --- Initialize the integrand, doesn't matter if function returns an array as scalar + array = array ---
    intF = 0.0

    for i in range(n[0]):
        x1 = Eta[0][i] * scale[0] + offset[0]
        func = lambda x2: f(x1 * np.ones_like(x2), x2)  # noqa: E731
        intF += scale[0] * W[0][i] * gaussQuad1d(func, n[-1], a[-1], b[-1])

    return intF


def gaussQuad3d(f, n, a=-1.0, b=1.0):
    """Perform a two dimensional integration using Gauss Quadrature

    Parameters
    ----------
    f : function
        The function to be integrated, should be able to accept vectorised input in each dimension
    n : int or list of ints
        Number of gauss quadrature points, can be different in each dimension
    a : float or list of floats, optional
        Lower integration limit can be different in each dimension, by default -1.0 for all dimensions
    b : float or lists of floats, optional
        Upper integration limit can be different in each dimension, by default 1.0 for all dimensions

    Returns
    -------
    return type of f
        The integrand
    """

    if isinstance(n, int):
        n = [n] * 3
    if isinstance(a, float):
        a = [a] * 3
    if isinstance(b, float):
        b = [b] * 3

    Eta = []
    W = []
    scale = []
    offset = []
    for i in range(2):
        Eta.append(np.array(getGaussPoints(n[i])))
        W.append(np.array(getgaussWeights(n[i])))
        scale.append(np.array(0.5 * (b[i] - a[i])))
        offset.append(np.array(0.5 * (a[i] + b[i])))
    totalScale = scale[0] * scale[1]

    # --- Initialize the integrand, doesn't matter if function returns an array as scalar + array = array ---
    intF = 0.0

    for i in range(n[0]):
        x1 = Eta[0][i] * scale[0] + offset[0]
        for j in range(n[1]):
            x2 = Eta[1][j] * scale[1] + offset[1]
            func = lambda x3: f(x1 * np.ones_like(x3), x2 * np.ones_like(x3), x3)  # noqa: E731
            intF += totalScale * W[0][i] * W[1][j] * gaussQuad1d(func, n[-1], a[-1], b[-1])

    return intF


def getTriGaussPoints(n):
    """Get the coordinates of the points for nth order Gaussian integration over a triangular domain

    Values taken from https://kratos-wiki.cimne.upc.edu/index.php/Numerical_Integration
    and https://link.springer.com/content/pdf/bbm%3A978-3-540-32609-0%2F1.pdf

    Parameters
    ----------
    n : int
        Order of integration

    Returns
    -------
    xGauss : 2 x N array
        Paramteric coordinates of Gauss integration points
    """

    return TriGaussPoints[n]


def getTriGaussWeights(n):
    """Get the weights of the points for nth order Gaussian integration over a triangular domain

    Values taken from https://kratos-wiki.cimne.upc.edu/index.php/Numerical_Integration
    and https://link.springer.com/content/pdf/bbm%3A978-3-540-32609-0%2F1.pdf

    Parameters
    ----------
    n : int
        Order of integration

    Returns
    -------
    xGauss : 1 x N array
        Weights of Gauss integration points
    """

    return TriGaussWeights[n]

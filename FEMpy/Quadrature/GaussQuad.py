"""
==============================================================================
Gauss Quadrature Points and Weights
==============================================================================
@File    :   guassQuad.py
@Date    :   2021/03/10
@Author  :   Alasdair Christison Gray
@Description : Methods for computing Gauss Quadrature points and weights
"""

import numpy as np
import pickle
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def getGaussQuadWeights(numDimensions, order):
    """Compute arbitrary order Gauss Quadrature weights in up to 3 dimensions

    Parameters
    ----------
    numDimensions : int
        Number of dimensions to compute weights for
    order : int
        Order of the quadrature rule

    Returns
    -------
    np.array of length order ** numDimensions
        Gauss quadrature weights

    Raises
    ------
    ValueError
        If requested number of dimensions is not supported
    """
    W, _ = _getGaussQuad1dData(order)
    if numDimensions == 1:
        return W
    elif numDimensions >= 2:
        W1 = np.repeat(W, order)  # Rows
        W2 = np.tile(W, order)  # Columns
        if numDimensions == 2:
            return W1 * W2
        elif numDimensions == 3:
            W1 = np.tile(W1, order)
            W2 = np.tile(W2, order)
            W3 = np.repeat(W, order**2)  # 3rd dimension
            return W1 * W2 * W3
        else:
            raise ValueError(
                f"Gauss quadrature weights only computable for 1, 2 or 3 dimensions, you asked for {numDimensions} dimensions"
            )


@lru_cache(maxsize=None)
def getGaussQuadPoints(numDimensions, order):
    """Compute arbitrary order Gauss Quadrature point coordinates in up to 3 dimensions

    Parameters
    ----------
    numDimensions : int
        Number of dimensions to compute points for
    order : int
        Order of the quadrature rule

    Returns
    -------
    (order ** numDimensions) x numDimensions np.array
        Gauss quadrature point coordinates

    Raises
    ------
    ValueError
        If requested number of dimensions is not supported
    """
    _, points = _getGaussQuad1dData(order)
    if numDimensions == 1:
        return points
    elif numDimensions >= 2:
        X = np.repeat(points, order)  # Rows
        Y = np.tile(points, order)  # Columns
        if numDimensions == 2:
            return np.vstack((X, Y)).T
        elif numDimensions == 3:
            X = np.tile(X, order)
            Y = np.tile(Y, order)
            Z = np.repeat(points, order**2)  # 3rd dimension
            return np.vstack((X, Y, Z)).T
        else:
            raise ValueError(
                f"Gauss quadrature weights only computable for 1, 2 or 3 dimensions, you asked for {numDimensions} dimensions"
            )


@lru_cache(maxsize=None)
def _getGaussQuad1dData(order):
    return np.polynomial.legendre.leggauss(order)


# ==============================================================================
# EVERYTHING BELOW HERE SHOULD BE DELETED ONCE NEW FEMPY IS COMPLETE
# ==============================================================================

# --- Load the Gauss quadrature weights and coordinates ---
dataDir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dataDir, "GaussQuadWeights.pkl"), "rb") as f:
    gaussWeights = pickle.load(f)
with open(os.path.join(dataDir, "GaussQuadCoords.pkl"), "rb") as f:
    gaussCoords = pickle.load(f)


def getgaussWeights(n):
    """Get the weights for n-point numerical integration using Gauss Quadrature

    Gauss Quadrature integration with n points will exactly integrate polynomials of order <= 2n-1

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

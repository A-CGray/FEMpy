"""
==============================================================================
Gauss Quadrature Points and Weights
==============================================================================
@File    :   guassQuad.py
@Date    :   2021/03/10
@Author  :   Alasdair Christison Gray
@Description : Methods for computing Gauss Quadrature points and weights
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
from functools import lru_cache

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


@lru_cache(maxsize=None)
def getGaussQuadWeights(numDim, order):
    """Compute arbitrary order Gauss Quadrature weights in up to 3 dimensions

    Parameters
    ----------
    numDim : int
        Number of dimensions to compute weights for
    order : int
        Order of the quadrature rule

    Returns
    -------
    np.array of length order ** numDim
        Gauss quadrature weights

    Raises
    ------
    ValueError
        If requested number of dimensions is not supported
    """
    _, W = _getGaussQuad1dData(order)
    if numDim == 1:
        return W
    elif numDim >= 2:
        W1 = np.repeat(W, order)  # Rows
        W2 = np.tile(W, order)  # Columns
        if numDim == 2:
            return W1 * W2
        elif numDim == 3:
            W1 = np.tile(W1, order)
            W2 = np.tile(W2, order)
            W3 = np.repeat(W, order**2)  # 3rd dimension
            return W1 * W2 * W3
        else:
            raise ValueError(
                f"Gauss quadrature weights only computable for 1, 2 or 3 dimensions, you asked for {numDim} dimensions"
            )


@lru_cache(maxsize=None)
def getGaussQuadPoints(numDim, order):
    """Compute arbitrary order Gauss Quadrature point coordinates in up to 3 dimensions

    Parameters
    ----------
    numDim : int
        Number of dimensions to compute points for
    order : int
        Order of the quadrature rule

    Returns
    -------
    (order ** numDim) x numDim np.array
        Gauss quadrature point coordinates

    Raises
    ------
    ValueError
        If requested number of dimensions is not supported
    """
    points, _ = _getGaussQuad1dData(order)
    if numDim == 1:
        return points
    elif numDim >= 2:
        X = np.repeat(points, order)  # Rows
        Y = np.tile(points, order)  # Columns
        if numDim == 2:
            return np.vstack((X, Y)).T
        elif numDim == 3:
            X = np.tile(X, order)
            Y = np.tile(Y, order)
            Z = np.repeat(points, order**2)  # 3rd dimension
            return np.vstack((X, Y, Z)).T
        else:
            raise ValueError(
                f"Gauss quadrature weights only computable for 1, 2 or 3 dimensions, you asked for {numDim} dimensions"
            )


@lru_cache(maxsize=None)
def _getGaussQuad1dData(order):
    """Just a wrapper around numpy's leggauss function that caches the results

    Parameters
    ----------
    order : int
        integration order

    Returns
    -------
    1d array
        1D gauss quadrature point coordinates
    1d array
        1D gauss quadrature weights
    """
    return np.polynomial.legendre.leggauss(order)

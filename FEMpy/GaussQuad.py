"""
==============================================================================
Gauss Quadrature Points and Weights
==============================================================================
@File    :   guassQuad.py
@Date    :   2021/03/10
@Author  :   Alasdair Christison Gray
@Description : This file contains the coordinates and weights (to far too many decimal places) for numerical integration
on the domain (-1, 1) using Gauss Quadrature. Weights and coordinates are given to 256 decimal places for up to 64 point
integration which can integrate up to 127th order polynomials exactly. These values were taken from:

https://pomax.github.io/bezierinfo/legendre-gauss.html

To use these values include `from guassQuad import getGaussPoints, getgaussWeights` in your code and then use
`getGaussPoints(n)`, `getGaussPoints(n)` to get lists of the points and weights for n-point integration.
"""

import numpy as np
import pickle
import os

# --- Load the Gauss quadrature weights and coordinates ---
dataDir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(dataDir,"GaussQuadWeights.pkl"), "rb") as f:
    gaussWeights = pickle.load(f)
with open(os.path.join(dataDir,"GaussQuadCoords.pkl"), "rb") as f:
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


if __name__ == "__main__":
    import scipy.integrate as integrate

    def TestFunc(x):
        f = 1.0
        for i in range(1, 10):
            f += x ** i
        return f

    def TestFunc2d(x1, x2):
        f = 1.0
        for i in range(1, 10):
            f += x1 ** i - 3.0 * x2 ** i
        return f

    def TestFunc3d(x1, x2, x3):
        f = 1.0
        for i in range(1, 10):
            f += x1 ** i - 4.0 * x2 ** i + 3.0 * x3 ** i
        return f

    def TestMatFunc3d(x1, x2, x3):
        A = np.zeros((len(x1), 3, 3))
        for i in range(len(x1)):
            A[i] = np.array([[x1[i], 2.0, 3.0], [1.0, x2[i], 3.0], [1.0, 2.0, x3[i]]])
        return A

    gaussInt = gaussQuad1d(TestFunc, 6, a=-4.0, b=1.0)
    scipyInt = integrate.quad(TestFunc, -4.0, 1.0)[0]
    print(gaussInt)
    print(scipyInt)

    gaussInt = gaussQuad2d(TestFunc2d, [6, 6], a=[-10.0, -1.0], b=[1.0, 1.0])
    scipyInt = integrate.dblquad(TestFunc2d, -1.0, 1.0, -10.0, 1.0)[0]
    print(gaussInt)
    print(scipyInt)

    gaussInt = gaussQuad3d(TestFunc3d, [6, 6, 6], a=[-4.0, 2.0, 0.0], b=[1.0, 3.0, 4.0])
    scipyInt = integrate.tplquad(TestFunc3d, 0.0, 4.0, 2.0, 3.0, -4.0, 1.0)[0]
    print(gaussInt)
    print(scipyInt)

    # Integrate a function that returns a matrix
    gaussInt = gaussQuad3d(TestMatFunc3d, [6, 6, 6], a=[-4.0, 2.0, 0.0], b=[1.0, 3.0, 4.0])
    print(gaussInt)

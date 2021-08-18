"""
==============================================================================
Lagrange polynomial interpolation
==============================================================================
@File    :   LagrangePoly.py
@Date    :   2021/03/12
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


@njit(cache=True)
def LagrangePoly1d(x, n):
    """Compute the values of the 1d Lagrange polynomials at a series of points



    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        Points to compute polynomial values at, should be between -1 and 1
    n : int
        Number of Lagrange polynomials, equal to the number of nodes along the edge, n points results in n-1 order
        polynomials.

    Returns
    -------
    N : nP x n array
        Shape function values
    """
    xp = x.flatten()
    xi = np.linspace(-1.0, 1.0, n)
    N = np.ones((len(xp), n), dtype=xp.dtype)
    for m in range(n):
        for i in list(range(m)) + list(range(m + 1, n)):
            N[:, m] *= (xp - xi[i]) / (xi[m] - xi[i])
    # print("computing lagrange shape funcs")
    return N


@njit(cache=True)
def LagrangePoly1dDeriv(x, n):
    """Compute the derivatives of the 1d Lagrange polynomials at a series of points

    This method computes the derivates using the complex step method because I couldn't be bothered implementing the
    analytical derivatives

    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        Points to compute polynomial values at, should be between -1 and 1
    n : int
        Number of Lagrange polynomials, equal to the number of nodes along the edge, n points results in n-1 order
        polynomials.

    Returns
    -------
    dNdx : nP x n array
        Shape function derivative values
    """
    xp = x.flatten() + 1j * 1e-200
    dNdx = 1e200 * np.imag(LagrangePoly1d(xp, n))
    return dNdx


@njit(cache=True)
def LagrangePoly2d(x, y, n):
    """Compute the derivatives of the 2d Lagrange polynomials at a series of points in 2d space



    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        x coordinates of points to compute polynomial values at, should be between -1 and 1
    y : array of length nP (0D, nPx1 or 1xnP)
        y coordinates of points to compute polynomial values at, should be between -1 and 1
    n : int
        Number of Lagrange polynomials, in 2d, there are n nodes in each direction, giving n^2, n-1 order polynomials

    Returns
    -------
    N : nP x n array
        Shape function values
    """
    xp = x.flatten()
    yp = y.flatten()
    Nx = LagrangePoly1d(xp, n)
    Ny = LagrangePoly1d(yp, n)
    N = np.zeros((len(xp), n ** 2), dtype=xp.dtype)
    for i in range(len(xp)):
        for j in range(n):
            for k in range(n):
                N[i, j * n + k] = Nx[i, k] * Ny[i, j]
    return N


@njit(cache=True)
def LagrangePoly2dDeriv(x, y, n):
    """Compute the derivatives of the 2d Lagrange polynomials at a series of points in 2d space



    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        x coordinates of points to compute polynomial values at, should be between -1 and 1
    y : array of length nP (0D, nPx1 or 1xnP)
        y coordinates of points to compute polynomial values at, should be between -1 and 1s
    n : int
        Number of Lagrange polynomials, in 2d, there are n nodes in each direction, giving n^2, n-1 order polynomials

    Returns
    -------
    dNdx : nP x 2 x n array
        Shape function derivative values
    """
    xp = x.flatten()
    yp = y.flatten()
    Nx = LagrangePoly1d(xp, n)
    Ny = LagrangePoly1d(yp, n)
    dNdx = LagrangePoly1dDeriv(xp, n)
    dNdy = LagrangePoly1dDeriv(yp, n)
    N = np.zeros((len(xp), 2, n ** 2))
    for i in range(len(xp)):
        for j in range(n):
            for k in range(n):
                N[i, 0, j * n + k] = dNdx[i, k] * Ny[i, j]
                N[i, 1, j * n + k] = Nx[i, k] * dNdy[i, j]
    return N


if __name__ == "__main__":
    import time

    x = np.array([np.linspace(-1.0, 1.0, 4)])
    y = -np.ones_like(x)
    # Call functions so they get jit compiled and we can check their output
    print(LagrangePoly1d(x, 3), "\n")
    print(LagrangePoly1dDeriv(x, 3), "\n")
    print(LagrangePoly2d(x, y, 3), "\n")
    print(LagrangePoly2dDeriv(x, y, 3), "\n\n\n")
    startTime = time.time()
    for i in range(1000):
        LagrangePoly1d(x, 3)
        LagrangePoly1dDeriv(x, 3)
        LagrangePoly2d(x, y, 3)
        LagrangePoly2dDeriv(x, y, 3)
    print(time.time() - startTime)

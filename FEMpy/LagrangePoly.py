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
    return 1e200 * np.imag(LagrangePoly1d(xp, n))


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
        y coordinates of points to compute polynomial values at, should be between -1 and 1
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


@njit(cache=True)
def LagrangePolyTri(x, y, n):
    """Compute the values of the Lagrangian polynomial for a triangular basis up to order 3

    [extended_summary]

    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        x coordinates of points to compute polynomial values at, should be between 0 and 1
    y : array of length nP (0D, nPx1 or 1xnP)
        y coordinates of points to compute polynomial values at, should be between 0 and 1
    n : int
        Polynomial/element order

    Returns
    -------
    N : nP x (n+1)*(n+2)/2 array
        Shape function values
    """
    xp = x.flatten()
    yp = y.flatten()
    N = np.zeros((len(xp), (n + 1) * (n + 2) // 2), dtype=xp.dtype)

    if n == 1:
        N[:, 0] = 1.0 - xp - yp
        N[:, 1] = xp
        N[:, 2] = yp

    elif n == 2:
        x2 = xp ** 2
        y2 = yp ** 2
        xy = xp * yp
        N[:, 0] = 2 * x2 + 4 * xy - 3 * xp + 2 * y2 - 3 * y + 1
        N[:, 1] = 2 * x2 - xp
        N[:, 2] = 2 * y2 - yp
        N[:, 3] = 4 * xy
        N[:, 4] = -4 * (xy + y2 - 1)
        N[:, 5] = -4 * (x2 + xy - 1)

    elif n == 3:
        x2 = xp ** 2
        y2 = yp ** 2
        x3 = xp ** 3
        y3 = yp ** 3
        xy = xp * yp
        x2y = x2 * yp
        xy2 = xp * y2
        N[:, 0] = (
            -4.5 * x3
            - 0.5 * 27.0 * x2y
            + 9 * x2
            - 0.5 * 27.0 * xy2
            + 18.0 * xy
            - 6.5 * xp
            - 4.5 * y3
            + 9.0 * y2
            - 6.5 * yp
            + 1.0
        )
        N[:, 1] = 0.5 * (9.0 * x3 - 9.0 * x2 + 2.0 * xp)
        N[:, 2] = 0.5 * (9.0 * y3 - 9.0 * y2 + 2.0 * yp)
        N[:, 3] = 4.5 * (3.0 * x2y - xy)
        N[:, 4] = 4.5 * (3.0 * xy2 - xy)
        N[:, 5] = 4.5 * (3.0 * x2y + 6 * xy2 - 5.0 * xy + 3.0 * y3 - 5.0 * y2 + 2.0 * yp)
        N[:, 6] = 4.5 * (-3.0 * xy2 + xy - 3.0 * y3 + 4.0 * y2 - yp)
        N[:, 7] = 4.5 * (3.0 * x3 + 6.0 * x2y - 5.0 * x2 + 3.0 * xy2 - 5.0 * xy + 2.0 * xp)
        N[:, 8] = 4.5 * (-3.0 * x3 - 3.0 * x2y + 4.0 * x2 + xy - xp)
        N[:, 9] = -27.0 * (x2y + xy2 - xy)
        return N


@njit(cache=True)
def LagrangePolyTriDeriv(x, y, n):
    """Compute the derivatives of the triangular basis Lagrange polynomials at a series of points in 2d space



    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        x coordinates of points to compute polynomial values at, should be between 0 and 1
    y : array of length nP (0D, nPx1 or 1xnP)
        y coordinates of points to compute polynomial values at, should be between 0 and 1s
    n : int
        Number of Lagrange polynomials, in 2d, there are n nodes in each direction, giving n^2, n-1 order polynomials

    Returns
    -------
    dNdx : nP x 2 x (n+1)*(n+2)/2 array
        Shape function derivative values
    """

    xp = x.flatten()
    yp = y.flatten()
    dNdx = np.zeros((len(xp), 2, (n + 1) * (n + 2) // 2))

    if n == 1:
        dNdx[:, :, 0] = -1.0
        dNdx[:, 0, 1] = 1.0
        dNdx[:, 1, 2] = 1.0

    elif n == 2:
        dNdx[:, 0, 0] = 4.0 * xp + 4.0 * yp - 3.0
        dNdx[:, 1, 0] = 4.0 * xp + 4.0 * yp - 3.0
        dNdx[:, 0, 1] = 4.0 * xp - 1.0
        dNdx[:, 1, 2] = 4.0 * yp - 1.0
        dNdx[:, 0, 3] = 4.0 * yp
        dNdx[:, 1, 3] = 4.0 * xp
        dNdx[:, 0, 4] = -4.0 * yp
        dNdx[:, 1, 4] = -4.0 * (xp + 2.0 * yp)
        dNdx[:, 0, 5] = -4.0 * (2.0 * xp + yp)
        dNdx[:, 1, 5] = -4.0 * xp

    elif n == 3:
        x2 = xp ** 2
        y2 = yp ** 2
        xy = xp * yp
        dNdx[:, 0, 0] = -13.5 * x2 - 27.0 * xy + 18 * xp - 0.5 * 27.0 * y2 + 18.0 * yp - 6.5
        dNdx[:, 1, 0] = -0.5 * 27.0 * x2 - 27.0 * xy + 18.0 * xp - 13.5 * y2 + 18.0 * yp - 6.5

        dNdx[:, 0, 1] = 0.5 * (27.0 * x2 - 18.0 * xp + 2.0)

        dNdx[:, 1, 2] = 0.5 * (27.0 * y2 - 18.0 * yp + 2.0)

        dNdx[:, 0, 3] = 4.5 * (6.0 * xy - yp)
        dNdx[:, 1, 3] = 4.5 * (3.0 * x2 - xp)

        dNdx[:, 0, 4] = 4.5 * (3.0 * y2 - yp)
        dNdx[:, 1, 4] = 4.5 * (6.0 * xy - xp)

        dNdx[:, 0, 5] = 4.5 * (6.0 * xy + 6.0 * y2 - 5.0 * yp)
        dNdx[:, 1, 5] = 4.5 * (3.0 * x2 + 12.0 * xy - 5.0 * xp + 9.0 * y2 - 10.0 * yp + 2.0)

        dNdx[:, 0, 6] = 4.5 * (-3.0 * y2 + yp)
        dNdx[:, 1, 6] = 4.5 * (-6.0 * xy + xp - 9.0 * y2 + 8.0 * yp - 1.0)

        dNdx[:, 0, 7] = 4.5 * (9.0 * x2 + 12.0 * xy - 10.0 * xp + 3.0 * y2 - 5.0 * yp + 2.0)
        dNdx[:, 1, 7] = 4.5 * (6.0 * x2 + 6.0 * xy - 5.0 * xp)

        dNdx[:, 0, 8] = 4.5 * (-9.0 * x2 - 6.0 * xy + 8.0 * xp + yp - 1.0)
        dNdx[:, 1, 8] = 4.5 * (-6.0 * xy + xp)

        dNdx[:, 0, 9] = -27.0 * (2.0 * xy + y2 - yp)
        dNdx[:, 1, 9] = -27.0 * (x2 + 2.0 * xy - xp)

        return dNdx


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
    for _ in range(1000):
        LagrangePoly1d(x, 3)
        LagrangePoly1dDeriv(x, 3)
        LagrangePoly2d(x, y, 3)
        LagrangePoly2dDeriv(x, y, 3)
    print(time.time() - startTime)

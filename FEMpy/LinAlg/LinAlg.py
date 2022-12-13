"""
==============================================================================
FEMpy linear algebra module
==============================================================================
@File    :   LinAlg.py
@Date    :   2022/12/07
@Author  :   Alasdair Christison Gray
@Description : This module contains function for computing determinants and
inverses of 1x1, 2x2 and 3x3 matrices waaaaay faster than numpy. This is very
useful inside the element classes for mapping between the real and reference
elements.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
from numba import njit, prange
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


@njit(cache=True)
def convertTo3D(A):
    """Convert a numpy array from (n+2)D to 3D by flattening the first n dimensions.

    Parameters
    ----------
    A : (n+2)D numpy array
        Array to convert

    Returns
    -------
    3D numpy array
        The converted 3D array
    int
        The new flattened first dimension
    """
    origShape = A.shape
    n = 1
    for ii in range(len(origShape) - 2):
        n *= origShape[ii]
    Anew = A.reshape(n, *origShape[-2:])

    return Anew, n


def det1(A):
    """Compute the determinants of a series of 1x1 matrices.

    Parameters
    ----------
    A : nx1x1 array_like
        Arrays to compute determinants of

    Returns
    -------
    dets : array of length n
        Matrix determinants
    """
    return np.ascontiguousarray(A.reshape(A.shape[:-2]))


@njit(cache=True, fastmath=True, parallel=True)
def det2(A):
    """Compute the determinants of a series of 2x2 matrices.

    Parameters
    ----------
    A : nxmx...x2x2 array_like
        Multidimensional array of 2x2 arrays to compute determinants of

    Returns
    -------
    dets : nxmx... array
        Matrix determinants
    """
    origShape = A.shape
    A, n = convertTo3D(A)

    dets = np.zeros(n)
    for i in prange(n):
        dets[i] = A[i, 0, 0] * A[i, 1, 1] - A[i, 0, 1] * A[i, 1, 0]
    return np.ascontiguousarray(dets.reshape(origShape[:-2]))


@njit(cache=True, fastmath=True, parallel=True)
def det3(A):
    """Compute the determinants of a series of 3x3 matrices.

    Parameters
    ----------
    A : nx3x3 array_like
        Arrays to compute determinants of

    Returns
    -------
    dets : array of length n
        Matrix determinants
    """
    origShape = A.shape
    A, n = convertTo3D(A)
    dets = np.zeros(n)
    for i in prange(n):
        dets[i] = (
            A[i, 0, 0] * (A[i, 1, 1] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 1])
            - A[i, 0, 1] * (A[i, 1, 0] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 0])
            + A[i, 0, 2] * (A[i, 1, 0] * A[i, 2, 1] - A[i, 1, 1] * A[i, 2, 0])
        )
    return np.ascontiguousarray(dets.reshape(origShape[:-2]))


def inv1(A):
    """Compute the inverses of a series of 1x1 matrices.

    Parameters
    ----------
    A : nx1x1 array_like
        Arrays to compute inverses of

    Returns
    -------
    dets : nx1x1 array
        Matrix inverses
    """
    return 1.0 / A


@njit(cache=True, fastmath=True, parallel=True)
def inv2(A):
    """Compute the inverses of a series of 2x2 matrices.

    Parameters
    ----------
    A : nx2x2 array_like
        Arrays to compute inverses of

    Returns
    -------
    dets : nx2x2 array
        Matrix inverses
    """
    origShape = A.shape
    A, n = convertTo3D(A)
    invdets = 1.0 / det2(A)
    invs = np.zeros((n, 2, 2))
    for i in prange(n):
        invs[i, 0, 0] = invdets[i] * A[i, 1, 1]
        invs[i, 1, 1] = invdets[i] * A[i, 0, 0]
        invs[i, 0, 1] = -invdets[i] * A[i, 0, 1]
        invs[i, 1, 0] = -invdets[i] * A[i, 1, 0]
    return np.ascontiguousarray(invs.reshape(origShape))


@njit(cache=True, fastmath=True, parallel=True)
def inv3(A):
    """Compute the inverses of a series of 3x3 matrices.

    Parameters
    ----------
    A : nx3x3 array_like
        Arrays to compute inverses of

    Returns
    -------
    dets : nx3x3 array
        Matrix inverses
    """
    origShape = A.shape
    A, n = convertTo3D(A)
    invdets = 1.0 / det3(A)
    invs = np.zeros((n, 3, 3))
    for i in prange(n):
        invs[i, 0, 0] = invdets[i] * (A[i, 1, 1] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 1])
        invs[i, 0, 1] = -invdets[i] * (A[i, 0, 1] * A[i, 2, 2] - A[i, 0, 2] * A[i, 2, 1])
        invs[i, 0, 2] = invdets[i] * (A[i, 0, 1] * A[i, 1, 2] - A[i, 0, 2] * A[i, 1, 1])
        invs[i, 1, 0] = -invdets[i] * (A[i, 1, 0] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 0])
        invs[i, 1, 1] = invdets[i] * (A[i, 0, 0] * A[i, 2, 2] - A[i, 0, 2] * A[i, 2, 0])
        invs[i, 1, 2] = -invdets[i] * (A[i, 0, 0] * A[i, 1, 2] - A[i, 0, 2] * A[i, 1, 0])
        invs[i, 2, 0] = invdets[i] * (A[i, 1, 0] * A[i, 2, 1] - A[i, 1, 1] * A[i, 2, 0])
        invs[i, 2, 1] = -invdets[i] * (A[i, 0, 0] * A[i, 2, 1] - A[i, 0, 1] * A[i, 2, 0])
        invs[i, 2, 2] = invdets[i] * (A[i, 0, 0] * A[i, 1, 1] - A[i, 0, 1] * A[i, 1, 0])
    return np.ascontiguousarray(invs.reshape(origShape))


if __name__ == "__main__":
    np.random.seed(0)
    A2 = np.random.rand(1000, 2, 2)
    A3 = np.random.rand(1000, 3, 3)
    dets2 = det2(A2)
    dets3 = det3(A3)
    invs2 = inv2(A2)
    invs3 = inv3(A3)
    # Check error between this implementation and numpy, use hybrid error measure (f_ref - f_test)/(f_ref + 1) which
    # measures the relative error for large numbers and absolute error for small numbers
    errors = {}
    errors["det2"] = np.linalg.norm((dets2 - np.linalg.det(A2)) / (dets2 + 1.0))
    errors["det3"] = np.linalg.norm((dets3 - np.linalg.det(A3)) / (dets3 + 1.0))
    errors["inv2"] = np.linalg.norm((invs2 - np.linalg.inv(A2)) / (invs2 + 1.0))
    errors["inv3"] = np.linalg.norm((invs3 - np.linalg.inv(A3)) / (invs3 + 1.0))
    for name, error in errors.items():
        print(f"Error norm in {name} = {error:.03e}")

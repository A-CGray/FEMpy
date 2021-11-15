from numba import njit
import numpy as np


def det1(A):
    """Compute the determinants of a series of 1x1 matrices."""
    return A.flatten()


@njit(cache=True, fastmath=True)
def det2(A):
    """Compute the determinants of a series of 2x2 matrices.

    Parameters
    ----------
    A : nx2x2 array_like
        Arrays to compute detrminents of

    Returns
    -------
    dets : array of length n
        Matrix determinants
    """
    n = np.shape(A)[0]
    dets = np.zeros(n)
    for i in range(n):
        dets[i] = A[i, 0, 0] * A[i, 1, 1] - A[i, 0, 1] * A[i, 1, 0]
    return dets


@njit(cache=True, fastmath=True)
def det3(A):
    """Compute the determinants of a series of 3x3 matrices.

    Parameters
    ----------
    A : nx3x3 array_like
        Arrays to compute detrminents of

    Returns
    -------
    dets : array of length n
        Matrix determinants
    """

    n = np.shape(A)[0]
    dets = np.zeros(n)
    for i in range(n):
        dets[i] = (
            A[i, 0, 0] * (A[i, 1, 1] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 1])
            - A[i, 0, 1] * (A[i, 1, 0] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 0])
            + A[i, 0, 2] * (A[i, 1, 0] * A[i, 2, 1] - A[i, 1, 1] * A[i, 2, 0])
        )
    return dets


def inv1(A):
    """Compute the determinants of a series of 1x1 matrices."""
    return 1.0 / A


@njit(cache=True, fastmath=True)
def inv2(A):
    """Compute the inverses of a series of 2x2 matrices.

    Parameters
    ----------
    A : nx2x2 array_like
        Arrays to compute detrminents of

    Returns
    -------
    dets : nx2x2 array
        Matrix inverses
    """
    invdets = 1.0 / det2(A)
    n = len(invdets)
    invs = np.zeros((n, 2, 2))
    for i in range(n):
        invs[i, 0, 0] = invdets[i] * A[i, 1, 1]
        invs[i, 1, 1] = invdets[i] * A[i, 0, 0]
        invs[i, 0, 1] = -invdets[i] * A[i, 0, 1]
        invs[i, 1, 0] = -invdets[i] * A[i, 1, 0]
    return invs


@njit(cache=True, fastmath=True)
def inv3(A):
    """Compute the inverses of a series of 3x3 matrices.

    Parameters
    ----------
    A : nx3x3 array_like
        Arrays to compute detrminents of

    Returns
    -------
    dets : nx3x3 array
        Matrix inverses
    """
    invdets = 1.0 / det3(A)
    n = len(invdets)
    invs = np.zeros((n, 3, 3))
    for i in range(n):
        invs[i, 0, 0] = invdets[i] * (A[i, 1, 1] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 1])
        invs[i, 0, 1] = -invdets[i] * (A[i, 0, 1] * A[i, 2, 2] - A[i, 0, 2] * A[i, 2, 1])
        invs[i, 0, 2] = invdets[i] * (A[i, 0, 1] * A[i, 1, 2] - A[i, 0, 2] * A[i, 1, 1])
        invs[i, 1, 0] = -invdets[i] * (A[i, 1, 0] * A[i, 2, 2] - A[i, 1, 2] * A[i, 2, 0])
        invs[i, 1, 1] = invdets[i] * (A[i, 0, 0] * A[i, 2, 2] - A[i, 0, 2] * A[i, 2, 0])
        invs[i, 1, 2] = -invdets[i] * (A[i, 0, 0] * A[i, 1, 2] - A[i, 0, 2] * A[i, 1, 0])
        invs[i, 2, 0] = invdets[i] * (A[i, 1, 0] * A[i, 2, 1] - A[i, 1, 1] * A[i, 2, 0])
        invs[i, 2, 1] = -invdets[i] * (A[i, 0, 0] * A[i, 2, 1] - A[i, 0, 1] * A[i, 2, 0])
        invs[i, 2, 2] = invdets[i] * (A[i, 0, 0] * A[i, 1, 1] - A[i, 0, 1] * A[i, 1, 0])
    return invs


if __name__ == "__main__":
    A2 = np.random.rand(1000, 2, 2)
    A3 = np.random.rand(1000, 3, 3)
    dets2 = det2(A2)
    dets3 = det3(A3)
    invs2 = inv2(A2)
    invs3 = inv3(A3)
    print(np.linalg.norm(dets2 - np.linalg.det(A2)))
    print(np.linalg.norm(dets3 - np.linalg.det(A3)))
    print(np.linalg.norm(invs2 - np.linalg.inv(A2)))
    print(np.linalg.norm(invs3 - np.linalg.inv(A3)))

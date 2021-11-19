from numba import njit
import numpy as np


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
    return A.flatten()


@njit(cache=True)
def det2(A):
    """Compute the determinants of a series of 2x2 matrices.

    Parameters
    ----------
    A : nx2x2 array_like
        Arrays to compute determinants of

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


@njit(cache=True)
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


@njit(cache=True)
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
    invdets = 1.0 / det2(A)
    n = len(invdets)
    invs = np.zeros((n, 2, 2))
    for i in range(n):
        invs[i, 0, 0] = invdets[i] * A[i, 1, 1]
        invs[i, 1, 1] = invdets[i] * A[i, 0, 0]
        invs[i, 0, 1] = -invdets[i] * A[i, 0, 1]
        invs[i, 1, 0] = -invdets[i] * A[i, 1, 0]
    return invs


@njit(cache=True)
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

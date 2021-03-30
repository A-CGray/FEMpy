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


@njit(cache=True)
def serendipityShapeFuncs(x, y):
    """Compute the shape functions of an 8 node serendipity Quad element

    [extended_summary]

    Parameters
    ----------
    x : array of length nP (0D, nPx1 or 1xnP)
        x coordinates of points to compute polynomial values at, should be between -1.0 and 1.0
    y : array of length nP (0D, nPx1 or 1xnP)
        y coordinates of points to compute polynomial values at, should be between -1.0 and 1s

    Returns
    -------
    N : nP x n array
        Shape function values
    """
    psi = x.flatten()
    eta = y.flatten()
    N = np.zeros((len(psi), 8))

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

    [extended_summary]

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
    for i in range(1000):
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
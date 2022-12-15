"""
==============================================================================
Functions for computing the von Mises stress
==============================================================================
@File    :   VonMises.py
@Date    :   2022/11/30
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
def vonMises2DPlaneStress(stresses):
    """Compute the von Mises stress for a 2d plane stress state

    Parameters
    ----------
    stresses : numPoints x 3 array
        Stresses in the form [sigma_xx, sigma_yy, tau_xy]

    Returns
    -------
    array of length numPoints
        Von Mises stress at each point
    """
    s11 = stresses[:, 0]
    s22 = stresses[:, 1]
    s12 = stresses[:, 2]
    return np.sqrt(s11**2 - s11 * s22 + s22**2 + 3 * s12**2)


@njit(cache=True)
def vonMises2DPlaneStrain(stresses, nu):
    """Compute the von Mises stress for a 2d plane strain state

    Parameters
    ----------
    stresses : numPoints x 3 array
        Stresses in the form [sigma_xx, sigma_yy, tau_xy]
    nu : float
        Poisson's ratio

    Returns
    -------
    array of length numPoints
        Von Mises stress at each point
    """
    s11 = stresses[:, 0]
    s22 = stresses[:, 1]
    s12 = stresses[:, 2]
    s33 = nu * (s11 + s22)
    return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * s12**2)


@njit(cache=True)
def vonMises3D(stresses):
    """Compute the von Mises stress for a 3d stress state

    Parameters
    ----------
    stresses : numPoints x 6 array
        Stresses in the form [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_xz, tau_yz]

    Returns
    -------
    array of length numPoints
        Von Mises stress at each point
    """
    s11 = stresses[:, 0]
    s22 = stresses[:, 1]
    s33 = stresses[:, 2]
    s12 = stresses[:, 3]
    s13 = stresses[:, 4]
    s23 = stresses[:, 5]
    return np.sqrt(
        0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s12**2 + s13**2 + s23**2)
    )

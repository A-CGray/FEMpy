"""
==============================================================================
Functions for computing strains in 2D
==============================================================================
@File    :   Strain2D.py
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


@njit(cache=True, fastmath=True, parallel=True)
def strain2D(UPrime, nonlinear=False):
    """Compute 2D strains from the displacement gradient

    Parameters
    ----------
    UPrime : numPoints x 2 x 2 array
        Displacement gradients at each point
    nonlinear : bool, optional
        Whether to compute the nonlinear Greene strain, by default False

    Returns
    -------
    strains : numPoints x 3 array
        Strains at each point, these are the engineering strains [e_xx, e_yy, gamma_xy]
    """
    numPoints = UPrime.shape[0]
    strains = np.zeros((numPoints, 3), dtype=UPrime.dtype)

    # e_xx = du_x/dx
    strains[:, 0] = UPrime[:, 0, 0]

    # e_yy = du_y/dy
    strains[:, 1] = UPrime[:, 1, 1]

    # gamma_xy = du_x/dy + du_y/dx
    strains[:, 2] = UPrime[:, 0, 1] + UPrime[:, 1, 0]

    if nonlinear:
        # e_xx = du_x/dx + 0.5 * (du_x/dx^2 + du_y/dx^2)
        strains[:, 0] += 0.5 * (UPrime[:, 0, 0] ** 2 + UPrime[:, 1, 0] ** 2)

        # e_yy = du_y/dy + 0.5 * (du_x/dy^2 + du_y/dy^2)
        strains[:, 1] += 0.5 * (UPrime[:, 0, 1] ** 2 + UPrime[:, 1, 1] ** 2)

        # gamma_xy = du_x/dy + du_y/dx + (du_x/dx * du_x/dy + du_y/dx * du_y/dy)
        strains[:, 2] += UPrime[:, 0, 0] * UPrime[:, 0, 1] + UPrime[:, 1, 0] * UPrime[:, 1, 1]

    return strains


@njit(cache=True, fastmath=True, parallel=True)
def strain2DSens(UPrime, nonlinear=False):
    """Compute the sensitivity of the 2D strains to the displacement gradients

    Parameters
    ----------
    UPrime : numPoints x 2 x 2 array
        Displacement gradients at each point
    nonlinear : bool, optional
        Whether to use the nonlinear Greene strain, by default False

    Returns
    -------
    strainSens : numPoints x numStrains x numStates x numDim array
        Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
    """
    numPoints = UPrime.shape[0]
    strainSens = np.zeros((numPoints, 3, 2, 2), dtype=UPrime.dtype)

    # e_xx = du_x/dx
    strainSens[:, 0, 0, 0] = 1.0
    # e_yy = du_y/dy
    strainSens[:, 1, 1, 1] = 1.0
    # gamma_xy = du_x/dy + du_y/dx
    strainSens[:, 2, 0, 1] = 1.0
    strainSens[:, 2, 1, 0] = 1.0

    if nonlinear:
        # e_xx = du_x/dx + 0.5 * (du_x/dx^2 + du_y/dx^2)
        strainSens[:, 0, 0, 0] += UPrime[:, 0, 0]
        strainSens[:, 0, 1, 0] += UPrime[:, 1, 0]

        # e_yy = du_y/dy + 0.5 * (du_x/dy^2 + du_y/dy^2)
        strainSens[:, 1, 0, 1] += UPrime[:, 0, 1]
        strainSens[:, 1, 1, 1] += UPrime[:, 1, 1]

        # gamma_xy = du_x/dy + du_y/dx + (du_x/dx * du_x/dy + du_y/dx * du_y/dy)
        strainSens[:, 2, 0, 0] += UPrime[:, 0, 1]
        strainSens[:, 2, 0, 1] += UPrime[:, 0, 0]

        strainSens[:, 2, 1, 0] += UPrime[:, 1, 1]
        strainSens[:, 2, 1, 1] += UPrime[:, 1, 0]

    return strainSens

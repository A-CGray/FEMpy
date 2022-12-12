"""
==============================================================================
Functions for computing strains in 3D
==============================================================================
@File    :   Strain3D.py
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
def strain3D(UPrime, nonlinear=False):
    """Compute 3D strains from the displacement gradient

    Parameters
    ----------
    UPrime : numPoints x 3 x 3 array
        Displacement gradients at each point
    nonlinear : bool, optional
        Whether to compute the nonlinear strain, by default False

    Returns
    -------
    strains : numPoints x 6 array
        Strains at each point, these are the engineering strains [e_xx, e_yy, e_zz, gamma_xy, gamma_xz, gamma_yz]
    """
    numPoints = UPrime.shape[0]
    strains = np.zeros((numPoints, 6), dtype=UPrime.dtype)

    # e_xx = du_x/dx
    strains[:, 0] = UPrime[:, 0, 0]

    # e_yy = du_y/dy
    strains[:, 1] = UPrime[:, 1, 1]

    # e_zz = du_z/dz
    strains[:, 2] = UPrime[:, 2, 2]

    # gamma_xy = du_x/dy + du_y/dx
    strains[:, 3] = UPrime[:, 0, 1] + UPrime[:, 1, 0]

    # gamma_xz = du_x/dz + du_z/dx
    strains[:, 4] = UPrime[:, 0, 2] + UPrime[:, 2, 0]

    # gamma_yz = du_y/dz + du_z/dy
    strains[:, 5] = UPrime[:, 1, 2] + UPrime[:, 2, 1]

    if nonlinear:
        # e_xx = du_x/dx + 0.5 * (du_x/dx^2 + du_y/dx^2+ du_z/dx^2)
        strains[:, 0] += 0.5 * (UPrime[:, 0, 0] ** 2 + UPrime[:, 1, 0] ** 2 + UPrime[:, 2, 0] ** 2)

        # e_yy = du_y/dy + 0.5 * (du_x/dy^2 + du_y/dy^2 + du_z/dy^2)
        strains[:, 1] += 0.5 * (UPrime[:, 0, 1] ** 2 + UPrime[:, 1, 1] ** 2 + UPrime[:, 2, 1] ** 2)

        # e_zz = du_z/dz + 0.5 * (du_x/dz^2 + du_y/dz^2 + du_z/dz^2)
        strains[:, 2] += 0.5 * (UPrime[:, 0, 2] ** 2 + UPrime[:, 1, 2] ** 2 + UPrime[:, 2, 2] ** 2)

        # gamma_xy = du_x/dy + du_y/dx + (du_x/dx * du_x/dy + du_y/dx * du_y/dy + du_z/dx * du_z/dy)
        strains[:, 3] += (
            UPrime[:, 0, 0] * UPrime[:, 0, 1] + UPrime[:, 1, 0] * UPrime[:, 1, 1] + UPrime[:, 2, 0] * UPrime[:, 2, 1]
        )

        # gamma_xz = du_x/dz + du_z/dx + (du_x/dx * du_x/dz + du_y/dx * du_y/dz + du_z/dx * du_z/dz)
        strains[:, 4] += (
            UPrime[:, 0, 0] * UPrime[:, 0, 2] + UPrime[:, 1, 0] * UPrime[:, 1, 2] + UPrime[:, 2, 0] * UPrime[:, 2, 2]
        )

        # gamma_yz = du_y/dz + du_z/dy + (du_x/dy * du_x/dz + du_y/dy * du_y/dz + du_z/dy * du_z/dz)
        strains[:, 5] += (
            UPrime[:, 0, 1] * UPrime[:, 0, 2] + UPrime[:, 1, 1] * UPrime[:, 1, 2] + UPrime[:, 2, 1] * UPrime[:, 2, 2]
        )

    return strains


@njit(cache=True, fastmath=True, parallel=True)
def strain3DSens(UPrime, nonlinear=False):
    """Compute the sensitivity of the 3D strains to the displacement gradients

    Parameters
    ----------
    UPrime : numPoints x 3 x 3 array
        Displacement gradients at each point
    nonlinear : bool, optional
        Whether to use the nonlinear Greene strain, by default False

    Returns
    -------
    strainSens : numPoints x numStrains x numStates x numDim array
        Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
    """
    numPoints = UPrime.shape[0]
    strainSens = np.zeros((numPoints, 6, 3, 3), dtype=UPrime.dtype)

    # e_xx = du_x/dx
    strainSens[:, 0, 0, 0] = 1.0

    # e_yy = du_y/dy
    strainSens[:, 1, 1, 1] = 1.0

    # e_zz = du_z/dz
    strainSens[:, 2, 2, 2] = 1.0

    # gamma_xy = du_x/dy + du_y/dx
    strainSens[:, 3, 0, 1] = 1.0
    strainSens[:, 3, 1, 0] = 1.0

    # gamma_xz = du_x/dz + du_z/dx
    strainSens[:, 4, 0, 2] = 1.0
    strainSens[:, 4, 2, 0] = 1.0

    # gamma_yz = du_y/dz + du_z/dy
    strainSens[:, 5, 1, 2] = 1.0
    strainSens[:, 5, 2, 1] = 1.0

    if nonlinear:
        # e_xx = du_x/dx + 0.5 * (du_x/dx^2 + du_y/dx^2+ du_z/dx^2)
        strainSens[:, 0, 0, 0] += UPrime[:, 0, 0]
        strainSens[:, 0, 1, 0] += UPrime[:, 1, 0]
        strainSens[:, 0, 2, 0] += UPrime[:, 2, 0]

        # e_yy = du_y/dy + 0.5 * (du_x/dy^2 + du_y/dy^2 + du_z/dy^2)
        strainSens[:, 1, 0, 1] += UPrime[:, 0, 1]
        strainSens[:, 1, 1, 1] += UPrime[:, 1, 1]
        strainSens[:, 1, 2, 1] += UPrime[:, 2, 1]

        # e_zz = du_z/dz + 0.5 * (du_x/dz^2 + du_y/dz^2 + du_z/dz^2)
        strainSens[:, 2, 0, 2] += UPrime[:, 0, 2]
        strainSens[:, 2, 1, 2] += UPrime[:, 1, 2]
        strainSens[:, 2, 2, 2] += UPrime[:, 2, 2]

        # gamma_xy = du_x/dy + du_y/dx + (du_x/dx * du_x/dy + du_y/dx * du_y/dy + du_z/dx * du_z/dy)
        strainSens[:, 3, 0, 0] += UPrime[:, 0, 1]
        strainSens[:, 3, 1, 0] += UPrime[:, 1, 1]
        strainSens[:, 3, 2, 0] += UPrime[:, 2, 1]

        strainSens[:, 3, 0, 1] += UPrime[:, 0, 0]
        strainSens[:, 3, 1, 1] += UPrime[:, 1, 0]
        strainSens[:, 3, 2, 1] += UPrime[:, 2, 0]

        # gamma_xz = du_x/dz + du_z/dx + (du_x/dx * du_x/dz + du_y/dx * du_y/dz + du_z/dx * du_z/dz)
        strainSens[:, 3, 0, 0] += UPrime[:, 0, 2]
        strainSens[:, 3, 1, 0] += UPrime[:, 1, 2]
        strainSens[:, 3, 2, 0] += UPrime[:, 2, 2]

        strainSens[:, 3, 0, 2] += UPrime[:, 0, 0]
        strainSens[:, 3, 1, 2] += UPrime[:, 1, 0]
        strainSens[:, 3, 2, 2] += UPrime[:, 2, 0]

        # gamma_yz = du_y/dz + du_z/dy + (du_x/dy * du_x/dz + du_y/dy * du_y/dz + du_z/dy * du_z/dz)
        strainSens[:, 3, 0, 1] += UPrime[:, 0, 2]
        strainSens[:, 3, 1, 1] += UPrime[:, 1, 2]
        strainSens[:, 3, 2, 1] += UPrime[:, 2, 2]

        strainSens[:, 3, 0, 2] += UPrime[:, 0, 1]
        strainSens[:, 3, 1, 2] += UPrime[:, 1, 1]
        strainSens[:, 3, 2, 2] += UPrime[:, 2, 1]

    return strainSens

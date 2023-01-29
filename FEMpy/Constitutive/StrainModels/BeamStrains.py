"""
==============================================================================
Strain equations for beams
==============================================================================
@File    :   BeamStrains.py
@Date    :   2023/01/10
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
def timoshenkoStrain(u, UPrime):
    """Compute Timoshenko beam generalised strains from displacement and displacement gradient

    Timoshenko generalised strains are: ::

        [  phi  ] = [0    d/dx] [   v   ]
        [ gamma ] = [d/dx   -1] [ theta ]

    Parameters
    ----------
    u : numPoints x 2 array
        Vertical displacement and rotation angle
    UPrime : numPoints x 2 x 1 array
        Displacement gradients

    Returns
    -------
    strains : numpoints x 2 array
        Generalised moshenko beam strains
    """
    numPoints = UPrime.shape[0]
    strains = np.zeros((numPoints, 2))
    if np.iscomplex(u).any() or np.iscomplex(UPrime).any():
        strains = strains.astype(np.complex)

    strains[:, 0] = UPrime[:, 0, 0]
    strains[:, 1] = UPrime[:, 1, 0] - u[:, 1]

    return strains


@njit(cache=True, fastmath=True, parallel=True)
def timoshenkoStrainUPrimeSens(UPrime):
    """Compute the derivative of Timoshenko beam strains with respect to the displacement gradients

    Parameters
    ----------
    UPrime: numPoints x 2 x 1 array
        Displacement gradients
    """
    numPoints = UPrime.shape[0]
    strainSens = np.zeros((numPoints, 2, 2, 1), dtype=UPrime.dtype)

    strainSens[:, 0, 0, 1] = 1.0
    strainSens[:, 0, 1, 0] = 1.0

    return strainSens


@njit(cache=True, fastmath=True, parallel=True)
def timoshenkoStrainUSens(u):
    """Compute the derivative of Timoshenko beam strains with respect to the displacements

    Parameters
    ----------
    u : numPoints x 2 array
        Vertical displacement and rotation angle
    """
    numPoints = u.shape[0]
    strainSens = np.zeros((numPoints, 2, 2), dtype=u.dtype)

    strainSens[:, 1, 1] = -1.0

    return strainSens

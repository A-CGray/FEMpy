"""
==============================================================================
Stress equations for Timoshenko beams
==============================================================================
@File    :   BeamStresses.py
@Date    :   2023/01/28
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
def timoshenkoStress(strains, E, I, A, G, k):
    """Compute the generalised stresses for a Timoshenko beam from the generalised strains

    Timoshenko generalised stresses are: ::

        $$M_{xx} = -EI\phi, \quad \kappa AG\gamma$$

    Parameters
    ----------
    strains : numPoints x 2 array
        Timoshenko generalised strains (phi and gamma) at a series of points
    E : float
        Material elastic modulus
    I : numPoints array
        Second moment of area at each points
    A : numPoints array
        Cross-sectional area at each points
    G : float
        Material shear modulus
    k : float
        Shear correction factor
    """
    stress = np.zeros_like(strains)
    stress[:, 0] = -E * I * strains[:, 0]
    stress[:, 1] = k * A * G * strains[:, 1]

    return stress


@njit(cache=True, fastmath=True, parallel=True)
def timoshenkoStressStrainSens(strains, E, I, A, G, k):
    """Compute the sensitivity of the generalised stresses for a Timoshenko beam with respect to the generalised strains

    Timoshenko generalised stresses are: ::

        $$M_{xx} = -EI\phi, \quad \kappa AG\gamma$$

    Parameters
    ----------
    strains : numPoints x 2 array
        Timoshenko generalised strains (phi and gamma) at a series of points
    E : float
        Material elastic modulus
    I : numPoints array
        Second moment of area at each points
    A : numPoints array
        Cross-sectional area at each points
    G : float
        Material shear modulus
    k : float
        Shear correction factor
    """
    stressSens = np.zeros((strains.shape[0], 2, 2))
    stressSens[:, 0, 0] = -E * I
    stressSens[:, 1, 1] = k * A * G

    return stressSens

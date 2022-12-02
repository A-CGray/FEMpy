"""
==============================================================================
Stress equations for isotropic materials
==============================================================================
@File    :   IsoStress.py
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

# ==============================================================================
# Extension modules
# ==============================================================================

# ==============================================================================
# 2D plane stress
# ==============================================================================
def isoPlaneStressStress(strains, E, nu):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    strains : numPoints x numStrains array
        Strains at each point (e_xx, e_yy, gamma_xy)
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    stresses : numPoints x numStresses array
        Stresses at each point (sigma_xx, sigma_yy, tau_xy)
    """

    const_planeStress = (
        E
        / (1 - nu**2)
        * np.array(
            [
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2],
            ]
        )
    )
    stress_plane = np.einsum("as,pa->ps", const_planeStress, strains)
    return stress_plane


def isoPlaneStressStressStrainSens(strains, E, nu):
    """Compute the sensitivity of the stress with respect to the strain

    _extended_summary_

    Parameters
    ----------
    strains : numPoints x numStrains array
        Strains at each point (e_xx, e_yy, gamma_xy)
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    numPoints x numStresses x numStrains array
        Sensitivity of each stress to each strain at each point
    """
    const_planeStress = (
        E
        / (1 - nu**2)
        * np.array(
            [
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1 - nu) / 2],
            ]
        )
    )

    strainSens = np.tile(const_planeStress, (strains.shape[0], 1, 1))

    return strainSens


# ==============================================================================
# 2D plane strain
# ==============================================================================


def isoPlaneStrainStress(strains, E, nu):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    strains : numPoints x numStrains array
        Strains at each point (e_xx, e_yy, gamma_xy)
    E : _type_
        _description_
    nu : _type_
        _description_

    Returns
    -------
    stresses : numPoints x numStresses array
        Stresses at each point (sigma_xx, sigma_yy, tau_xy)
    """

    const_planeStrain = (
        E
        / (1 + nu)
        / (1 - 2 * nu)
        * np.array(
            [
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2 * nu) / 2],
            ]
        )
    )
    stress_plane = np.einsum("as,pa->ps", const_planeStrain, strains)
    return stress_plane


def isoPlaneStrainStressStrainSens(strains, E, nu):
    """Compute the sensitivity of the stress with respect to the strain

    _extended_summary_

    Parameters
    ----------
    strains : numPoints x numStrains array
        Strains at each point (e_xx, e_yy, gamma_xy)
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    numPoints x numStresses x numStrains array
        Sensitivity of each stress to each strain at each point
    """
    const_planeStrain = (
        E
        / (1 + nu)
        / (1 - 2 * nu)
        * np.array(
            [
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2 * nu) / 2],
            ]
        )
    )

    strainSens = np.tile(const_planeStrain, (strains.shape[0], 1, 1))

    return strainSens


# ==============================================================================
# 3D
# ==============================================================================


def iso3DStress(strains, E, nu):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    strains : numPointsx numStrains array
        Strains at each point (e_xx, e_yy, gamma_xy)
    E : _type_
        _description_
    nu : _type_
        _description_
    """

    const_3D = (
        E
        / (1 + nu)
        / (1 - 2 * nu)
        * np.array(
            [
                [1 - nu, nu, nu, 0, 0, 0],
                [nu, 1 - nu, nu, 0, 0, 0],
                [nu, nu, 1 - nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
            ]
        )
    )

    stress_3D = np.einsum("as,pa->ps", const_3D, strains)
    return stress_3D


def iso3DStressStrainSens(strains, E, nu):
    """Compute the sensitivity of the stress with respect to the strain

    _extended_summary_

    Parameters
    ----------
    strains : numPoints x numStrains array
        Strains at each point (e_xx, e_yy, gamma_xy)
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    numPoints x numStresses x numStrains array
        Sensitivity of each stress to each strain at each point
    """
    const_3D = (
        E
        / (1 + nu)
        / (1 - 2 * nu)
        * np.array(
            [
                [1 - nu, nu, nu, 0, 0, 0],
                [nu, 1 - nu, nu, 0, 0, 0],
                [nu, nu, 1 - nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
            ]
        )
    )

    strainSens = np.tile(const_3D, (strains.shape[0], 1, 1))

    return strainSens

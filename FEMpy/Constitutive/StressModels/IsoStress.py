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


def planeStressMat(E, nu):
    """Get constitutive (stress-strain) matrix for plane stress

    Parameters
    ----------
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    3x3 array
        Constitutive matrix for plane stress
    """
    return (
        E
        / (1.0 - nu**2)
        * np.array(
            [
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0],
            ]
        )
    )


def planeStrainMat(E, nu):
    """Get constitutive (stress-strain) matrix for plane stress

    Parameters
    ----------
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    3x3 array
        Constitutive matrix for plane stress
    """
    return (
        E
        / ((1.0 + nu) * (1.0 - 2.0 * nu))
        * np.array(
            [
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
            ]
        )
    )


def threeDMat(E, nu):
    """Get constitutive (stress-strain) matrix for a 3D stress state

    Parameters
    ----------
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    3x3 array
        Constitutive matrix for plane stress
    """
    return (
        E
        / ((1.0 + nu) * (1.0 - 2.0 * nu))
        * np.array(
            [
                [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
                [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
                [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
            ]
        )
    )


# ==============================================================================
# 2D plane stress
# ==============================================================================
def isoPlaneStressStress(strains, E, nu):
    """Compute stress from strains for a 2D plane stress state

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

    mat = planeStressMat(E, nu)
    stress = np.einsum("as,pa->ps", mat, strains)
    return stress


def isoPlaneStressStressStrainSens(strains, E, nu):
    """Compute the sensitivity of the stress with respect to the strain for a 2D plane stress state

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
    mat = planeStressMat(E, nu)
    strainSens = np.tile(mat, (strains.shape[0], 1, 1))
    return strainSens


# ==============================================================================
# 2D plane strain
# ==============================================================================


def isoPlaneStrainStress(strains, E, nu):
    """Compute stress from strains for a 2D plane strain state

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

    mat = planeStrainMat(E, nu)
    stress = np.einsum("as,pa->ps", mat, strains)
    return stress


def isoPlaneStrainStressStrainSens(strains, E, nu):
    """Compute the sensitivity of the stress with respect to the strain



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
    mat = planeStrainMat(E, nu)
    strainSens = np.tile(mat, (strains.shape[0], 1, 1))
    return strainSens


# ==============================================================================
# 3D
# ==============================================================================


def iso3DStress(strains, E, nu):
    """Compute the 3D stress state in a 3D isotropic material from strains

    Parameters
    ----------
    strains : numPointsx numStrains array
        Strains at each point (e_xx, e_yy, e_zz, gamma_xy, gamm_xz, gamma_yz)
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio
    """

    mat = threeDMat(E, nu)
    stress_3D = np.einsum("as,pa->ps", mat, strains)
    return stress_3D


def iso3DStressStrainSens(strains, E, nu):
    """Compute the sensitivity of the stress with respect to the strain



    Parameters
    ----------
    strains : numPoints x numStrains array
        Strains at each point (e_xx, e_yy, e_zz, gamma_xy, gamm_xz, gamma_yz)
    E : float
        Elastic modulus
    nu : float
        Poisson's ratio

    Returns
    -------
    numPoints x numStresses x numStrains array
        Sensitivity of each stress to each strain at each point
    """
    mat = threeDMat(E, nu)
    strainSens = np.tile(mat, (strains.shape[0], 1, 1))
    return strainSens

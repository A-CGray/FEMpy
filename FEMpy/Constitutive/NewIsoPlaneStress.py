"""
==============================================================================
2D Isotropic plane stress constitutive model
==============================================================================
@File    :   NewIsoPlaneStress.py
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
from FEMpy.Constitutive.StrainModels import Planar2DStrain, Planar2DStrainSens
from FEMpy.Constitutive.StressModels import isoPlaneStress
from FEMpy.Constitutive import ConstitutiveModel


class IsoPlaneStress(ConstitutiveModel):
    def __init__(self, E, nu, rho, t, linear=True):
        """Create an isotropic plane stress constitutive model

        _extended_summary_

        Parameters
        ----------
        E : float
            Elastic Modulus
        nu : float
            Poisson's ratio
        rho : float
            Density
        t : float
            Thickness, this will be used as the initial thickness value for all elements but can be changed later by calling setDesignVariables in the model
        linear : bool, optional
            Whether to use the linear kinematic equations for strains, by default True
        """
        # --- Design variables ---
        # This model has only one design variable, the thickness of the material
        designVars = {}
        designVars["Thickness"] = {"defaultValue": t}

        # --- States ---
        stateNames = ["X Displacement", "Y Displacement"]

        # --- Strains ---
        strainNames = ["e_xx", "e_yy", "gamma_xy"]

        # --- Stresses ---
        stressNames = ["sigma_xx", "sigma_yy", "tau_xy"]

        # --- Functions ---
        functionNames = ["Mass", "Von Mises Stress", "Tresca Stress"]

        numDim = 2

        self.super.__init__(numDim, stateNames, strainNames, stressNames, designVars, functionNames, linear)

    def computeStrains(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the strains at each one

        _extended_summary_

        Parameters
        ----------
        states : numPoints x numStates array
            State values at each point
        stateGradients : numPoints x numStates x numDim array
            State gradients at each point
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : _type_
            _description_

        Returns
        -------
        numPoints x numStrains array
            Strain components at each point
        """
        return Planar2DStrain(UPrime=stateGradients, nonlinear=not self.linear)

    def computeStrainStateGradSens(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the
        sensitivity of the strains to the state gradient at each one

        _extended_summary_

        Parameters
        ----------
        states : numPoints x numStates array
            State values at each point
        stateGradients : numPoints x numStates x numDim array
            State gradients at each point
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : _type_
            _description_

        Returns
        -------
        numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
        """

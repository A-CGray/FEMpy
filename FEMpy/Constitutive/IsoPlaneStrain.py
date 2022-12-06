"""
==============================================================================
2D Isotropic plane strain constitutive model
==============================================================================
@File    :   NewIsoPlaneStrain.py
@Date    :   2022/11/30
@Author  :   Alasdair Christison Gray and M.A. Saja A.Kaiyoom
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
from FEMpy.Constitutive.StressModels import isoPlaneStrainStress, isoPlaneStrainStressStrainSens, vonMises2DPlaneStrain
from FEMpy.Constitutive import ConstitutiveModel


class IsoPlaneStrain(ConstitutiveModel):
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
        stateNames = ["X-Displacement", "Y-Displacement"]

        # --- Strains ---
        strainNames = ["e_xx", "e_yy", "gamma_xy"]

        # --- Stresses ---
        stressNames = ["sigma_xx", "sigma_yy", "tau_xy"]

        # --- Functions ---
        functionNames = ["Mass", "Von-Mises-Stress"]

        # --- Material properties ---
        self.E = E
        self.nu = nu

        numDim = 2

        super().__init__(numDim, stateNames, strainNames, stressNames, designVars, functionNames, linear)

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
        return Planar2DStrain(UPrime=stateGradients, nonlinear=not self.isLinear)

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
        return Planar2DStrainSens(UPrime=stateGradients, nonlinear=not self.isLinear)

    def computeStresses(self, strains, dvs):
        """Given the strains and design variables at a bunch of points, compute the stresses at each one

        _extended_summary_

        Parameters
        ----------
        strains : numPoints x numStrains array
            Strain components at each point
        dvs : _type_
            _description_

        Returns
        -------
        numPoints x numStresses array
            Stress components at each point
        """
        return isoPlaneStrainStress(strains, E=self.E, nu=self.nu)

    def computeStressStrainSens(self, strains, dvs):
        """Given the strains and design variables at a bunch of points, compute the sensitivity of the stresses to the strains at each one

        _extended_summary_

        Parameters
        ----------
        strains : numPoints x numStrains array
            Strain components at each point
        dvs : _type_
            _description_

        Returns
        -------
        sens : numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
        """
        return isoPlaneStrainStressStrainSens(strains, E=self.E, nu=self.nu)

    def computeVolumeScaling(self, coords, dvs):
        """Given the coordinates and design variables at a bunch of points, compute the volume scaling parameter at each one

        For this 2D model, the volume scaling is just the thickness

        Parameters
        ----------
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : _type_
            _description_

        Returns
        -------
        numPoints length array
            Volume scaling parameter at each point
        """
        return dvs["Thickness"]

    def getFunction(self, name):
        """Return a function that can be computed for this constitutive model

        Parameters
        ----------
        name : str
            Name of the function to compute

        Returns
        -------
        callable
            A function that can be called to compute the desired function at a bunch of points with the signature, f(states, stateGradients, coords, dvs)
            where:
            states is a numPoints x numStates array
            stateGradients is a numPoints x numStates x numDim array
            coords is a numPoints x numDim array
            dvs is a dictionary of numPoints length arrays
        """
        lowerCaseFuncNames = [func.lower() for func in self.functionNames]
        if name.lower() not in self.lowerCaseFuncNames:
            raise ValueError(
                f"{name} is not a valid function name for this constitutive model, valid choices are {self.functionNames}"
            )

        if name.lower() == "mass":

            def massFunc(states, stateGradients, coords, dvs):
                return np.ones(states.shape[0]) * self.rho

            return massFunc

        if name.lower() == "von-mises-stress":

            def VMFunc(states, stateGradients, coords, dvs):
                strains = self.computeStrains(states, stateGradients, coords, dvs)
                stresses = self.computeStresses(strains, dvs)
                return vonMises2DPlaneStrain(stresses, self.nu)

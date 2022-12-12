"""
==============================================================================
3D Isotropic elasticity constitutive model
==============================================================================
@File    :   Iso3D.py
@Date    :   2022/12/11
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
from FEMpy.Constitutive.StrainModels import strain3D, strain3DSens
from FEMpy.Constitutive.StressModels import iso3DStress, iso3DStressStrainSens, vonMises3D
from FEMpy.Constitutive import ConstitutiveModel


class Iso3D(ConstitutiveModel):
    """A constitutive model for a 3D isotropic material

    Inherits
    ----------
    ConstitutiveModel : _type_
        The base class for FEMpy constitutive models
    """

    def __init__(self, E, nu, rho, linear=True):
        """Create an isotropic plane stress constitutive model



        Parameters
        ----------
        E : float
            Elastic Modulus
        nu : float
            Poisson's ratio
        rho : float
            Density
        linear : bool, optional
            Whether to use the linear kinematic equations for strains, by default True
        """
        # --- Design variables ---
        # This model has no design variables
        designVars = {}

        # --- States ---
        stateNames = ["X-Displacement", "Y-Displacement", "Z-Displacement"]

        # --- Strains ---
        strainNames = ["e_xx", "e_yy", "e_zz", "gamma_xy", "gamma_xz", "gamma_yz"]

        # --- Stresses ---
        stressNames = ["sigma_xx", "sigma_yy", "sigma_zz", "tau_xy", "tau_xz", "tau_yz"]

        # --- Functions ---
        functionNames = ["Mass", "Von-Mises-Stress"]

        # --- Material properties ---
        self.E = E
        self.nu = nu
        self.rho = rho

        numDim = 3

        super().__init__(numDim, stateNames, strainNames, stressNames, designVars, functionNames, linear)

    def computeStrains(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the strains at each one

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
        return strain3D(UPrime=stateGradients, nonlinear=not self.isLinear)

    def computeStrainStateGradSens(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the
        sensitivity of the strains to the state gradient at each one



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
        return strain3DSens(UPrime=stateGradients, nonlinear=not self.isLinear)

    def computeStresses(self, strains, dvs):
        """Given the strains and design variables at a bunch of points, compute the stresses at each one



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
        return iso3DStress(strains, E=self.E, nu=self.nu)

    def computeStressStrainSens(self, strains, dvs):
        """Given the strains and design variables at a bunch of points, compute the sensitivity of the stresses to the strains at each one



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
        return iso3DStressStrainSens(strains, E=self.E, nu=self.nu)

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
        return np.ones(coords.shape[0])

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

            def func(states, stateGradients, coords, dvs):
                return np.ones(states.shape[0]) * self.rho

        if name.lower() == "von-mises-stress":

            def func(states, stateGradients, coords, dvs):
                strains = self.computeStrains(states, stateGradients, coords, dvs)
                stresses = self.computeStresses(strains, dvs)
                return vonMises3D(stresses)

        return func

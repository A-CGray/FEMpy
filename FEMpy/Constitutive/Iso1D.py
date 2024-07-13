"""
==============================================================================
1D elasticity constitutive model base class
==============================================================================
@File    :   Iso1D.py
@Date    :   2022/12/20
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
from FEMpy.Constitutive import ConstitutiveModel
from FEMpy.Constitutive.StrainModels import strain1D, strain1DSens
from FEMpy.Constitutive.StressModels import iso1DStress, iso1DStressStrainSens


class Iso1D(ConstitutiveModel):
    """Constitutive model for an axial bar

    Inherits
    ----------
    ConstitutiveModel : FEMpy.Constitutive.ConstitutiveModel
        The base class for FEMpy constitutive models
    """

    def __init__(self, E, rho, A, linear=True):
        """Create an axial bar constitutive model

        Parameters
        ----------
        E : float
            Elastic Modulus
        rho : float
            Density
        A : float
            Cross-sectional area
        linear : bool, optional
            Whether to use the linear kinematic equations for strains, by default True
        """
        # --- Design variables ---
        designVars = {}
        designVars["Area"] = {"defaultValue": A}

        # --- States ---
        stateNames = ["X-Displacement"]

        # --- Strains ---
        strainNames = ["e_xx"]

        # --- Stresses ---
        stressNames = ["sigma_xx"]

        # --- Functions ---
        functionNames = ["Mass", "Stress", "Strain"]

        # --- Material properties ---
        self.E = E
        self.rho = rho

        numDim = 1

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
        dvs : dict of arrays of length numPoints
            Design variable values at each point

        Returns
        -------
        numPoints x numStrains array
            Strain components at each point
        """
        return strain1D(UPrime=stateGradients, nonlinear=not self.isLinear)

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
        dvs : dict of arrays of length numPoints
            Design variable values at each point

        Returns
        -------
        numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
        """
        return strain1DSens(UPrime=stateGradients, nonlinear=not self.isLinear)

    def computeStresses(self, strains, dvs):
        """Given the strains and design variables at a bunch of points, compute the stresses at each one



        Parameters
        ----------
        strains : numPoints x numStrains array
            Strain components at each point
        dvs : dict of arrays of length numPoints
            Design variable values at each point

        Returns
        -------
        numPoints x numStresses array
            Stress components at each point
        """
        return iso1DStress(strains, E=self.E)

    def computeStressStrainSens(self, strains, dvs):
        """Given the strains and design variables at a bunch of points, compute the sensitivity of the stresses to the strains at each one

        Parameters
        ----------
        strains : numPoints x numStrains array
            Strain components at each point
        dvs : dict of arrays of length numPoints
            Design variable values at each point

        Returns
        -------
        sens : numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
        """
        return iso1DStressStrainSens(strains, E=self.E)

    def computeVolumeScaling(self, coords, dvs):
        """Given the coordinates and design variables at a bunch of points, compute the volume scaling parameter at each one

        For this 2D model, the volume scaling is just the thickness

        Parameters
        ----------
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : dict of arrays of length numPoints
            Design variable values at each point

        Returns
        -------
        numPoints length array
            Volume scaling parameter at each point
        """
        return dvs["Area"]

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

        if name.lower() not in self.lowerCaseFuncNames:
            raise ValueError(
                f"{name} is not a valid function name for this constitutive model, valid choices are {self.functionNames}"
            )

        if name.lower() == "mass":

            def massFunc(states, stateGradients, coords, dvs):
                return np.ones(states.shape[0]) * self.rho

            func = massFunc

        if name.lower() == "strain":

            def strainFunc(states, stateGradients, coords, dvs):
                return self.computeStrains(states, stateGradients, coords, dvs).flatten()

            func = strainFunc

        if name.lower() == "stress":

            def stressFunc(states, stateGradients, coords, dvs):
                strains = self.computeStrains(states, stateGradients, coords, dvs)
                return iso1DStress(strains, E=self.E).flatten()

            func = stressFunc

        return func

"""
==============================================================================
FEMpy Constitutive model base class
==============================================================================
@File    :   NewConstitutive.py
@Date    :   2022/11/28
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import abc

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from numba import njit

# ==============================================================================
# Extension modules
# ==============================================================================


class ConstitutiveModel:
    """The base class for all FEMpy constitutive models

    The constitutive model defines the underlying PDE being solved. Currently, this base class is defined for
    elasticity problems, but in future it may be extended for other PDE types.

    It contains information on:
    - The number of spatial dimensions the model is valid for
    - The number and names of the PDE states
    - The number and names of the stresses and strains for this model
    - The number and names of the design variables associated with the PDE
    - The names of functions which can be computed for this constitutive model (e.g mass, Von Mises stress etc)

    #### What do we need a constitutive model to do?:
    - Given the coordinates, state value, state gradient, and design variables at a point, compute:
        - The strain components
        - The sensitivities of the strain components
        - The stress components
        - The sensitivities of the stress components
        - The pointwise mass
        - The volume integral scaling parameter (e.g thickness for 2D plane models or 2*pi*r for 2D axisymmetric problems)
        - The weak form residual
        - The weak form residual Jacobian
        - Other arbitrary output values (e.g failure criterion)
    """

    def __init__(self, numDim, stateNames, strainNames, stressNames, designVars, functionNames, linear=True) -> None:
        """_summary_

        _extended_summary_

        Parameters
        ----------
        numDim : int
            Number of spatial dimensions the model is valid for
        stateNames : list of str
            Names for each state variable
        strainNames : list of str
            Names for each strain component
        stressNames : list of str
            Names for each stress component
        designVars : dict
            A nested dictionary of design variables, with the key being the name of the design variable and the value
            being a dictionary that contains various pieces of information about that DV, including:
                - "defaultValue" : The default value of that DV
        functionNames : list of str
            The names of functions that can be computed with this constitutive model
        linear : bool, optional
            Whether the constitutive model is linear or not, a.k.a whether the weak residual is a linear function of the states/state gradients, by default True
        """
        self.numDim = numDim

        self.stateNames = stateNames
        self.numStates = len(stateNames)

        self.strainNames = strainNames
        self.numStrains = len(strainNames)
        if len(stressNames) != self.numStrains:
            raise ValueError("Number of strains must equal number of stresses")
        self.stressNames = stressNames

        self.designVariables = designVars
        self.numDesignVariables = len(designVars)

        self.functionNames = functionNames

        self.isLinear = linear

    # ==============================================================================
    # Abstract methods: To be implemented by derived classes
    # ==============================================================================
    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def computeVolumeScaling(self, coords, dvs):
        """Given the coordinates and design variables at a bunch of points, compute the volume scaling parameter at each one

        The volume scaling parameter is used to scale functions that are integrated over the element to get a true
        volume integral. For example, in a 2D plane stress model, we need to multiply by the thickness of the element
        to get a true volume integral. In a 2D axisymmetric model, we need to multiply by 2*pi*r to get a true volume
        integral.

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
        raise NotImplementedError

    @abc.abstractmethod
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

        raise NotImplementedError

    # ==============================================================================
    # Public methods
    # ==============================================================================
    def computeWeakResiduals(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the weak residual integrand

        For an elasticity problem, the weak residual, derived from the virtual work principle is:

        R = int r dV = int du'/dq^T * de/du'^T * sigma * scale d(element)

        Where:
        - du'/dq is the sensitivity of the state gradient to the nodal state values, this is handled by the element
        - de/du' is the sensitivity of the strain to the state gradient
        - sigma are the stresses
        - scale is the volume scaling parameter

        This function computes `de/du'^T * sigma * scale` at each point

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
        numPoints x self.numDim x self.numStates array

        """
        strain = self.computeStrains(states, stateGradients, coords, dvs)
        stress = self.computeStresses(strain, dvs)
        scale = self.computeVolumeScaling(coords, dvs)

        strainSens = self.computeStrainStateGradSens(states, stateGradients, coords, dvs)

        # r = np.zeros((numPoints, self.numDim, self.numStates))

        # _computeWeakResidualProduct(strainSens, stress, scale, r)

        # return r

        # The einsum below is equivalent to:
        # r = np.zeros((numPoints, self.numDim, self.numStates))
        # numPoints = strainSens.shape[0]
        # for ii in range(numPoints):
        #     r[ii] += strainSens[ii].T @ stress[ii] * volScaling[ii]
        return np.einsum("pesd,pe,p->pds", strainSens, stress, scale, optimize=["einsum_path", (0, 1), (0, 1)])

    # ==============================================================================
    # Private methods
    # ==============================================================================


@njit(cache=True, fastmath=True, boundscheck=False)
def _computeWeakResidualProduct(dStraindUPrime, stress, volScaling, result):
    """Compute a nasty product of high dimensional arrays to compute the weak residual

    Computing the weak residual requires computing the following product at each point:

    `de/du'^T * sigma * scale`

    Where:
        - de/du' is the sensitivity of the strain to the state gradient
        - sigma are the stresses
        - scale is the volume scaling parameter

    Parameters
    ----------
    dStraindUPrime : numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
    stress : numPoints x numStrains array
        Stresses at each point
    volScaling : numPoints array
        Volume scaling at each point
    result : numPoints x numDim x numStates array
        _description_
    """
    numPoints = dStraindUPrime.shape[0]
    numStrains = dStraindUPrime.shape[1]

    for ii in range(numPoints):
        for jj in range(numStrains):
            result[ii] += dStraindUPrime[ii, jj].T * stress[ii, jj] * volScaling[ii]

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
        - Other arbitrary output values (e.g failure criterion)
    """

    def __init__(self, numDim, stateNames, strainNames, stressNames, designVars, functionNames) -> None:
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
            _description_
        dvs : _type_
            _description_

        Returns
        -------
        numPoints x numStrains array
            Strain components at each point
        """
        raise NotImplementedError

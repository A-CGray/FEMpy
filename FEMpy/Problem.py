"""
==============================================================================
FEMpy Problem Class
==============================================================================
@File    :   Problem.py
@Date    :   2022/11/21
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
from typing import Iterable, Union, Callable

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np


# ==============================================================================
# Extension modules
# ==============================================================================


class FEMpyProblem:
    """The FEMpy problem class represents a single finite element problem, many such problems may be associated with a
    single FEMpy model to represent different loading and boundary conditions.

    The problem class contains:
    - The state vactor
    - The residual vector
    - The Jacobian matrix

    And contains methods for:
    - Assembling local vectors and matrices into global vectors and matrices
    - Solving the problem
    - Writing the solution to a file
    """

    def __init__(self, name, model) -> None:
        self.name = name
        self.model = model
        self.states = np.zeros((self.numNodes, self.numStates))
        self.Jacobian = None
        self.RHS = np.zeros(self.numNodes * self.numStates)

        # --- Dictionary of boundary conditions ---
        self.BCDict = {}

    @property
    def constitutiveModel(self):
        """Get the constitutive model object associated with this problem

        Returns
        -------
        FEMpy constitutive model object
            The constitutive model object associated with this problem
        """
        return self.model.constitutiveModel

    @property
    def numDimensions(self) -> int:
        """Get the number of dimensions of the problem

        Returns
        -------
        int
            Number of dimensions of the problem
        """
        return self.model.numDimensions

    @property
    def numNodes(self) -> int:
        """Get the number of nodes in the model

        Returns
        -------
        int
            Number of nodes in the model
        """
        return self.model.numNodes

    @property
    def numDOFs(self) -> int:
        """Get the number of degrees of freedom in the problem

        Returns
        -------
        int
            Number of degrees of freedom in the problem
        """
        return self.model.numDOF

    @property
    def numStates(self) -> int:
        """Get the number of states

        This is not the total number of states in the problem, but the number of states associated with the problem's
        constitutive model, e.g for a heat transfer problem there is a single state (Temperature), and for a 2D
        elasticity problem there are 2 states (u and v displacements)

        Returns
        -------
        int
            Number of states
        """
        return self.constitutiveModel.numStates

    def computeFunction(name, elementReductionType=None, globalReductionType=None):
        """Compute a function over the whole model

        Parameters
        ----------
        name : str, optional
            Name of the function to compute, this must be one of the function names of the problem's constitutive model, by default ""
        elementReductionType : _type_, optional
            Type of reduction to do over each element (average, min, max, sum, integral etc), by default None
        globalReductionType : _type_, optional
            Type of reduction to do over all elements (average, sum, min, max etc), by default None
        """
        # for each element type:
        #   - Get the node coordinates and states and DVs for those elements
        #   - Compute the function values for all elements of that type, using:
        #       values = Element.computeFunction(nodeCoords, nodeStates, elementDVs, function, elementReductionType)
        # Then, if globalReductionType is not None:
        #   - Do the global reduction, return single value

        # nodeCoords : numElements x numNodes x numDim array
        #     Node coordinates for each element
        # nodeStates : numElements x numNodes x numStates array
        #     State values at the nodes of each element
        # dvs : numElements x numDVs array
        #     Design variable values for each element
        # values: array of length numElements
        #     Values of the function at each element

    def addFixedBC(
        self, name: str, nodeInds: Iterable[int], dof: Union[int, Iterable[int]], value: Union[float, Iterable[float]]
    ) -> None:
        """Add a fixed boundary condition to a set of nodes

        Parameters
        ----------
        name : str
            Name for this boundary condition
        nodeInds : int or iterable of ints
            Indicies of nodes to apply the boundary condition to
        dof : int or iterable of ints
            Degrees of freedom to apply this boundary condition to
        values : float or iterable of floats
            Values to fix states at, if a single value is supplied then this value is applied to all specified degrees of freedom
        """
        return None

    def addLoadToNodes(
        self,
        name: str,
        nodeInds: Iterable[int],
        dof: Union[int, Iterable[int]],
        value: Union[float, Iterable[float]],
        totalLoad: bool = False,
    ) -> None:
        """Add a load to a set of nodes

        Parameters
        ----------
        name : str
            Name for this load
        nodeInds : int or iterable of ints
            Indicies of nodes to apply the load to
        dof : int or iterable of ints
            Degrees of freedom to apply this load to at each node
        values : float or iterable of floats
            Load values, if a single value is supplied then this value is applied to all specified degrees of freedom
        totalLoad : bool, optional
            If true then the `values` are treated as total loads and split uniformly between the nodes, by default False, in which case the `values` are applied at each node
        """
        return None

    def addBodyLoad(self, name: str, loadingFunction: Union[Callable, Iterable[float]]) -> None:
        """Add a volumetric forcing term, commonly known as a "body force"

        _extended_summary_

        Parameters
        ----------
        name : str
            Name for the load
        loadingFunction : function or array of length numDimensions
            Pass an array to define a uniform field, otherwise, pass a function with the signature `F = loadingFunction(coord)` where coord is an n x numDimensions array of coordinates and F is an n x numDimensions array of loads at each point.
        """
        return None

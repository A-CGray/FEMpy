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
from typing import Iterable, Union, Callable, Optional

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from scipy.sparse import csc_array  # ,coo_array


# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy.Utils import AssemblyUtils


class FEMpyProblem:
    """The FEMpy problem class represents a single finite element problem, many such problems may be associated with a
    single FEMpy model to represent different loading and boundary conditions.

    The problem class contains:
    - The state vector
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
        self.BCs = {}
        self.loads = {}

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
    def numDim(self) -> int:
        """Get the number of dimensions of the problem

        Returns
        -------
        int
            Number of dimensions of the problem
        """
        return self.model.numDim

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

    @property
    def elements(self):
        """The element data structure"""
        return self.model.elements

    def computeFunction(self, name, elementReductionType=None, globalReductionType=None):
        """Compute a function over the whole model

        Parameters
        ----------
        name : str, optional
            Name of the function to compute, this must be one of the function names of the problem's constitutive model, by default ""
        elementReductionType : _type_, optional
            Type of reduction to do over each element (average, min, max, sum, integral etc), by default None
        globalReductionType : _type_, optional
            Type of reduction to do over all elements (average, sum, min, max etc), by default None

        Returns
        -------
        float
            for global reduction
        dict
            for element reduction
        """
        # for each element type:
        #   - Get the node coordinates and states and DVs for those elements
        #   - Compute the function values for all elements of that type, using:
        #       values = Element.computeFunction(nodeCoords, nodeStates, elementDVs, function, elementReductionType)
        # Then, if globalReductionType is not None:
        #   - Do the global reduction, return single value

        functionValues = {}

        elementDvs = self.model.dvs  # need to confirm size
        eval_func = self.model.constitutiveModel.getFunction(name)
        print(self.model.elements)
        for elType in self.model.elements:
            print(elType)
            elObject = self.model.elements[elType]["elementObject"]
            nodeCoords = self.getElementCoordinates(elType)
            nodeStates = self.getElementStates(elType)
            elementDvs = self.model.getElementDVs(elType)
            # nodeCoords : numElements x numNodes x numDim array
            #     Node coordinates for each element
            # nodeStates : numElements x numNodes x numStates array
            #     State values at the nodes of each element
            # dvs : numElements x numDVs array
            #     Design variable values for each element
            # values: array of length numElements
            #     Values of the function at each element
            functionValues[elType] = elObject.computeFunction(
                nodeCoords, nodeStates, elementDvs, eval_func, elementReductionType
            )

        # perform global reduction if specified
        if globalReductionType is not None:
            assert globalReductionType in ["average", "sum", "min", "max"], "globalReductionType not valid"

            # create reduction function
            if globalReductionType == "average":
                reductionFunc = np.average
            if globalReductionType == "sum":
                reductionFunc = np.sum
            if globalReductionType == "min":
                reductionFunc = np.min
            if globalReductionType == "max":
                reductionFunc = np.max

            globalValues = np.zeros(len(self.model.elements))
            for i, elType in enumerate(functionValues):
                globalValues[i] = reductionFunc(functionValues[elType])

            return reductionFunc(globalValues)

        return functionValues

    def addFixedBC(
        self,
        name: str,
        nodeInds: Union[int, Iterable[int]],
        dof: Union[int, Iterable[int]],
        value: Union[float, Iterable[float]],
    ) -> None:
        """Add a fixed boundary condition to a set of nodes

        For example ``addFixedBC("BCName", [0, 1, 2], [0,1], 0.0)`` would fix DOF 0 and 1 of nodes 0, 1 and 2 to 0.0

        Parameters
        ----------
        name : str
            Name for this boundary condition
        nodeInds : int or iterable of ints
            Indicies of nodes to apply the boundary condition to
        dof : int or iterable of ints
            Degrees of freedom to apply this boundary condition to, the same DOF are fixed at every node in ``nodeInds``
        values : float or iterable of floats
            Values to fix states at, if a single value is supplied then this value is applied to all specified degrees of freedom
        """

        # store the BC
        self.BCs[name] = {}
        dofNodes = []
        valDOF = []

        if isinstance(nodeInds, int):
            nodeInds = [nodeInds]

        if isinstance(dof, int):
            dof = [dof]

        if len(value) == 1:
            value = (np.ones(len(dof)) * value).tolist()
        elif len(value) != len(dof):
            raise Exception("value should be a single entry or a list of values the same length as the DOF list.")
            # value = np.ones(len(nodeInds)) * value

        for i in range(len(nodeInds)):
            for j in range(len(dof)):
                dofNodes.append(nodeInds[i] * self.numStates + dof[j])
                valDOF.append(value[j])
        self.BCs[name]["DOF"] = dofNodes
        self.BCs[name]["Value"] = valDOF

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

        # store the BC
        totalNodes = 1
        if totalLoad:
            totalNodes = len(nodeInds)

        self.loads[name] = {}
        dofNodes = []
        valDOF = []

        if isinstance(nodeInds, int):
            nodeInds = [nodeInds]

        if isinstance(dof, int):
            dof = [dof]

        if len(value) == 1:
            value = (np.ones(len(dof)) * value).tolist()
        elif len(value) != len(dof):
            raise Exception("value should be a single entry or a list of values the same length as the DOF list.")
            # value = np.ones(len(nodeInds)) * value

        for i in range(len(nodeInds)):
            for j in range(len(dof)):
                dofNodes.append(nodeInds[i] * self.numStates + dof[j])
                valDOF.append((value[j] / totalNodes))
        self.loads[name]["DOF"] = dofNodes
        self.loads[name]["Value"] = valDOF

    def addBodyLoad(self, name: str, loadingFunction: Union[Callable, Iterable[float]]) -> None:
        """Add a volumetric forcing term, commonly known as a "body force"

        _extended_summary_

        Parameters
        ----------
        name : str
            Name for the load
        loadingFunction : function or array of length numDim
            Pass an array to define a uniform field, otherwise, pass a function with the signature `F = loadingFunction(coord)` where coord is an n x numDim array of coordinates and F is an n x numDim array of loads at each point.
        """
        return None

    def assembleMatrix(self, states: np.ndarray, applyBCs: Optional[bool] = True) -> csc_array:
        """Assemble the global residual Jacobian matrix for the problem (a.k.a the stiffness matrix)

        _extended_summary_

        Parameters
        ----------
        stateVector : numNodes x numStates array
            The current system states
        applyBCs : bool, optional
            Whether to modify the matrix to include the boundary conditions, by default True

        Returns
        -------
        scipy csc_array
            The residual Jacobian
        """
        # - For each element type:
        #     - Get the node coordinates, node states and design variable values for all elements of that type
        #     - Compute the local matrix for all elements of that type
        #     - Convert to COO row, col, value lists
        # - Combine the lists from all element types
        # - Apply boundary conditions
        # - Create sparse matrix from lists

        # MatRows = []
        # MatColumns = []
        # MatEntries = []

        # for elementType, elementData in self.elements.items():
        # element = elementData["elementObject"]
        # numElements = elementData["numElements"]
        # nodeCoords = self.getElementCoordinates(elementType)
        # nodeStates = self.getElementStates(elementType)
        # elementDVs = self.model.getElementDVs(elementType)

        # localMats = element.computeJacobian(self, nodeCoords, nodeStates, dvs, self.constitutiveModel)

        # assemble local matrices into global matrix

        return None

    def assembleResidual(self, states: np.ndarray, applyBCs: Optional[bool] = True):
        """Assemble the global residual for the problem

        _extended_summary_

        Parameters
        ----------
        states : numNodes x numStates array
            The current system states
        applyBCs : bool, optional
            Whether to modify the vector to include the boundary conditions, by default True

        Returns
        -------
        numDOF array
            The residual vector
        """
        # - For each element type:
        #     - Get the node coordinates, node states and design variable values for all elements of that type
        #     - Compute the local residual for all elements of that type
        #     - Scatter local residuals into global residual vector
        # - Add loads to residual to the global residual vector
        # - Apply boundary conditions to global residual vector

        globalResidual = np.zeros(self.numDOF)

        for elementType, elementData in self.elements.items():
            element = elementData["elementObject"]
            nodeCoords = self.getElementCoordinates(elementType)
            nodeStates = self.getElementStates(elementType)
            elementDVs = self.model.getElementDVs(elementType)

            elementResiduals = element.computeResidual(self, nodeCoords, nodeStates, elementDVs, self.constitutiveModel)
            AssemblyUtils.scatterLocalResiduals(elementResiduals, elementData["connectivity"], globalResidual)

        # Add external loads to the residual
        globalResidual += AssemblyUtils.convertLoadsDictToVector(self.loads, self.numDOF)

        # Apply boundary conditions
        if applyBCs:
            AssemblyUtils.applyBCsToVector()

        return globalResidual

    def getElementCoordinates(self, elementType: str) -> np.ndarray:
        """Get the coordinates of the nodes for all elements of the specified type

        Parameters
        ----------
        elementType : str
            Name of the element type to get the coordinates for

        Returns
        -------
        numElements x numNodes x numDim array
            Node coordinates
        """
        numElements = self.elements[elementType]["numElements"]
        element = self.elements[elementType]["elementObject"]
        nodeCoords = np.zeros((numElements, element.numNodes, self.numDim))
        for ii in range(numElements):
            nodeInds = self.elements[elementType]["connectivity"][ii]
            nodeCoords[ii] = self.model.nodeCoords[nodeInds]
        return nodeCoords

    def getElementStates(self, elementType: str) -> np.ndarray:
        """Get the states of the nodes for all elements of the specified type

        Parameters
        ----------
        elementType : str
            Name of the element type to get the coordinates for

        Returns
        -------
        numElements x numNodes x numStates array
            Node coordinates
        """
        numElements = self.elements[elementType]["numElements"]
        element = self.elements[elementType]["elementObject"]
        nodeStates = np.zeros((numElements, element.numNodes, self.numStates))
        for ii in range(numElements):
            nodeInds = self.elements[elementType]["connectivity"][ii]
            nodeStates[ii] = self.states[nodeInds]
        return nodeStates

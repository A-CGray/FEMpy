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
from typing import Iterable, Union, Callable, Optional, Dict
import copy
import time
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from scipy.sparse import csc_array, coo_array
from baseclasses import BaseSolver

try:
    from pypardiso import factorized
except ModuleNotFoundError:
    from scipy.sparse.linalg import factorized

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy.Utils import AssemblyUtils
from FEMpy.Utils.KSAgg import ksAgg


class FEMpyProblem(BaseSolver):
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

    def __init__(self, name, model, options=None) -> None:
        self.model = model
        self.states = np.zeros((self.numNodes, self.numStates))
        self.Jacobian = None
        self.jacUpToDate = False
        self.factorizedJac = None
        self.Res = np.zeros(self.numNodes * self.numStates)
        self.resUpToDate = False

        self.solveCounter = 0

        # --- Dictionary of boundary conditions ---
        self.BCs = {}
        self.loads = {}

        # --- Set problem options by getting the defaults and updating them with any that the user supplied ---
        defaultOptions = self._getDefaultOptions()
        if options is None:
            options = {}

        # instantiate the solver
        super().__init__(name, "Finite Element Problem", defaultOptions=defaultOptions, options=options)

        # --- Take some options from the model if they are not defined for this problem ---
        inheritableOptions = ["outputDir", "outputFormat"]
        for option in inheritableOptions:
            if self.getOption(option) is None:
                self.setOption(option, self.model.getOption(option))

    @property
    def constitutiveModel(self):
        """The constitutive model object associated with this problem"""
        return self.model.constitutiveModel

    @property
    def isLinear(self) -> bool:
        """Whether the problem is linear"""
        return self.constitutiveModel.isLinear

    @property
    def numDim(self) -> int:
        """The number of dimensions of the problem"""
        return self.model.numDim

    @property
    def numNodes(self) -> int:
        """The number of nodes in the model"""
        return self.model.numNodes

    @property
    def numDOF(self) -> int:
        """The number of degrees of freedom in the problem"""
        return self.model.numDOF

    @property
    def numStates(self) -> int:
        """Get the number of states

        This is not the total number of states in the problem, but the number of states associated with the problem's
        constitutive model, e.g for a heat transfer problem there is a single state (Temperature), and for a 2D
        elasticity problem there are 2 states (u and v displacements)
        """
        return self.constitutiveModel.numStates

    @property
    def elements(self):
        """The element data structure"""
        return self.model.elements

    def solve(self):
        """Solve the finite element problem

        The solution is stored in the ``states`` attribute of the problem object
        """

        printTiming = self.getOption("printTiming")
        if printTiming:
            times = {}
            times["Start"] = time.time()
        # --- Assemble the residual ---
        self.updateResidual(applyBCs=True)
        if printTiming:
            times["ResAssembled"] = time.time()

        # --- Assemble the stiffness matrix ---
        self.updateJacobian(applyBCs=True)
        if printTiming:
            times["JacAssembled"] = time.time()

        # --- Solve for an update in the state ---
        update = self.solveJacLinear(-self.Res)
        if printTiming:
            times["Solved"] = time.time()

        # --- Apply the state update ---
        self.incrementState(update)

        if printTiming:
            self._printTiming(times)

        self.solveCounter += 1

    def solveJacLinear(self, RHS: np.ndarray) -> np.ndarray:
        """Solve the linearised residual equations with a right hand side

        This function solves the linearised residual equations with a right hand side, i.e. it solves the equation

        ``Jacobian @ update = RHS``

        Some checks are first made as to whether the Jacobian needs to be recomputed before the solve, and the
        factorized Jacobian matrix is stored for future solves.

        Parameters
        ----------
        RHS : np.ndarray
            The right hand side of the linear system

        Returns
        -------
        np.ndarray
            The solution to the linear system
        """
        if not self.jacUpToDate:
            self.updateJacobian()
        if self.factorizedJac is None or not self.jacIsFactorized:
            print("Factorising Jacobian")
            self.factorizedJac = factorized(self.Jacobian)
            self.jacIsFactorized = True
        print("Solving linear system")
        return self.factorizedJac(RHS)

    def incrementState(self, update: np.ndarray) -> None:
        """Add an increment to the state vector, effectively equivalent to ``self.states += update``

        This method should be used instead of ``self.states += update``  so that the problem can keep track of whether
        the residual and Jacobian are up to date.

        Parameters
        ----------
        update : _type_
            _description_
        """
        if update.shape == self.states.shape:
            self.states += update
        else:
            self.states += update.reshape(self.states.shape)
        self.markResOutOfDate()
        if not self.isLinear:
            self.markJacOutOfDate()

    def updateState(self, update: np.ndarray) -> None:
        """Update the state vector

        Parameters
        ----------
        update : np.ndarray
            The update to the state vector
        """
        if update.shape == self.states.shape:
            self.states[:] = update
        else:
            self.states[:] = update.reshape(self.states.shape)
        self.markResOutOfDate()
        if not self.isLinear:
            self.markJacOutOfDate()

    def reset(self):
        """Reset the problem to its initial state"""
        self.updateState(np.zeros((self.numNodes, self.numStates)))

    def updateResidual(self, applyBCs: bool = True):
        if not self.resUpToDate:
            print("Updating Residual")
            self.Res = self._assembleResidual(self.states, applyBCs=applyBCs)
            self.markResUpToDate()

    def updateJacobian(self, applyBCs: bool = True):
        if not self.jacUpToDate:
            print("Updating Jacobian")
            self.Jacobian = self._assembleMatrix(self.states, applyBCs=applyBCs)
            self.markJacUpToDate()
            self.jacIsFactorized = False

    def markResOutOfDate(self):
        """Mark the status of the residual as out of date"""
        self.resUpToDate = False

    def markJacOutOfDate(self):
        """Mark the status of the jacobian as out of date"""
        self.jacUpToDate = False

    def markResUpToDate(self):
        """Mark the status of the residual as up to date"""
        self.resUpToDate = True

    def markJacUpToDate(self):
        """Mark the status of the jacobian as up to date"""
        self.jacUpToDate = True

    def computeFunction(self, name, elementReductionType, globalReductionType=None):
        """Compute a function over the whole model

        Parameters
        ----------
        name : str, optional
            Name of the function to compute, this must be one of the function names of the problem's constitutive model, by default ""
        elementReductionType : _type_, optional
            Type of reduction to do over each element (sum, mean, integrate, max, min, ksmax, ksmin), by default None
        globalReductionType : _type_, optional
            Type of reduction to do over all elements (sum, mean, max, min, ksmax, ksmin), by default None

        Returns
        -------
        float
            for global reduction
        dict
            for element reduction
        """
        functionValues = {}

        elementDvs = self.model.dvs  # need to confirm size
        eval_func = self.model.constitutiveModel.getFunction(name)
        for elType in self.model.elements:
            elObject = self.model.elements[elType]["elementObject"]
            nodeCoords = self.model.getElementCoordinates(elType)
            nodeStates = self.getElementStates(elType)
            elementDvs = self.model.getElementDVs(elType)
            
            functionValues[elType] = elObject.computeFunction(
                nodeCoords, nodeStates, elementDvs, eval_func, elementReductionType
            )

        # perform global reduction if specified
        if globalReductionType is not None:
            assert globalReductionType.lower() in ["sum", "mean", "min", "max", "ksmax", "ksmin"], "globalReductionType not valid"

            # create reduction function
            if globalReductionType.lower() == "mean":
                reductionFunc = np.average
            if globalReductionType.lower() == "sum":
                reductionFunc = np.sum
            if globalReductionType.lower() == "min":
                reductionFunc = np.min
            if globalReductionType.lower() == "max":
                reductionFunc = np.max
            if globalReductionType.lower() == "ksmax":
                reductionFunc = lambda values: ksAgg(values, "max")
            if globalReductionType.lower() == "ksmin":
                reductionFunc = lambda values: ksAgg(values, "min")

            globalValues = np.zeros(len(self.model.elements))
            for i, elType in enumerate(functionValues):
                globalValues[i] = reductionFunc(functionValues[elType])

            return reductionFunc(globalValues)

        return functionValues

    def addFixedBCToNodes(
        self,
        name: str,
        nodeInds: Union[int, Iterable[int]],
        dof: Union[int, Iterable[int]],
        value: Union[float, Iterable[float]],
    ) -> None:
        """Add a fixed boundary condition to a set of nodes

        For example ``addFixedBCToNodes("BCName", [0, 1, 2], [0,1], 0.0)`` would fix DOF 0 and 1 of nodes 0, 1 and 2 to 0.0

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

        if isinstance(value, float) or isinstance(value, int):
            value = [value] * len(dof)
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
        scale = 1.0
        if totalLoad:
            scale = float(len(nodeInds))

        self.loads[name] = {}
        dofNodes = []
        valDOF = []

        if isinstance(nodeInds, int):
            nodeInds = [nodeInds]

        if isinstance(dof, int):
            dof = [dof]

        if isinstance(value, float) or isinstance(value, int):
            value = [value] * len(dof)
        elif len(value) != len(dof):
            raise Exception("value should be a single entry or a list of values the same length as the DOF list.")

        for i in range(len(nodeInds)):
            for j in range(len(dof)):
                dofNodes.append(nodeInds[i] * self.numStates + dof[j])
                valDOF.append((value[j] / scale))
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

    def getBCs(self) -> Dict[str, Dict[str, Union[Iterable[int], Iterable[float]]]]:
        """Get the full set of boundary conditions for this problem

        This combines the BCs defined specifically for this problem with those defined at the model level

        Returns
        -------
        Dict[str, Dict[str, Union[Iterable[int], Iterable[float]]]]
            Dictionary of boundary conditions, each entry is a dictionary with keys "DOF" and "Value"
        """
        BCs = copy.deepcopy(self.model.BCs)
        BCs.update(self.BCs)
        return BCs

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

    def writeSolution(self, baseName: Optional[str] = None, fileFormat: Optional[str] = None):
        """Write the current solution to a file

        _extended_summary_

        Parameters
        ----------
        baseName : str, optional
            Filename, extension will be ignored if included, by default uses ``f"{self.name}_{self.solveCounter:04d}"``
        fileFormat : str, optional
            File format/extension, can be any format supported by meshio, by default uses the format defined by the
            model or problem ``outputFormat`` option
        """
        # Get nodal state values
        nodeValues = {}
        for ii, state in enumerate(self.constitutiveModel.stateNames):
            nodeValues[state] = self.states[:, ii]

        # Get element DV values
        elementValues = {}
        for elementType in self.elements:
            elementValues[elementType] = {}
            elementDVs = self.model.getElementDVs(elementType)
            for name, values in elementDVs.items():
                elementValues[elementType][name] = values

        outputMesh = self.model.createOutputData(nodeValues=nodeValues, elementValues=elementValues)

        if baseName is None:
            baseName = f"{self.name}_{self.solveCounter:04d}"
        else:
            # Get filename without extension
            baseName - os.path.splitext(baseName)[0]

        if fileFormat is None:
            fileFormat = self.getOption("outputFormat")
        if fileFormat[0] != ".":
            fileFormat = "." + fileFormat

        # --- Create the output directory if it doesn't already exist ---
        outDir = self.getOption("outputDir")
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        outFileName = os.path.join(self.getOption("outputDir"), baseName + fileFormat)
        outputMesh.write(outFileName)

    # ==============================================================================
    # Private methods
    # ==============================================================================

    def _assembleMatrix(self, states: np.ndarray, applyBCs: Optional[bool] = True) -> csc_array:
        """Assemble the global residual Jacobian matrix for the problem (a.k.a the stiffness matrix)

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

        matRows = []
        matColumns = []
        matEntries = []

        # - For each element type:
        #     - Get the node coordinates, node states and design variable values for all elements of that type
        #     - Compute the local matrices for all elements of that type
        #     - Convert to COO row, col, value lists
        #       - Add to the global lists
        for elementType, elementData in self.elements.items():
            element = elementData["elementObject"]
            nodeCoords = self.model.getElementCoordinates(elementType)
            nodeStates = self.getElementStates(elementType)
            elementDVs = self.model.getElementDVs(elementType)

            localMats = element.computeResidualJacobians(nodeStates, nodeCoords, elementDVs, self.constitutiveModel)
            row_inds, col_inds, values = AssemblyUtils.localMatricesToCOOArrays(localMats, elementData["DOF"])
            matRows += row_inds
            matColumns += col_inds
            matEntries += values

        # Apply boundary conditions to COO data
        if applyBCs:
            BCs = self.getBCs()
            BCDOF, _ = AssemblyUtils.convertBCDictToLists(BCs)
            matRows, matColumns, matEntries = AssemblyUtils.applyBCsToMatrix(matRows, matColumns, matEntries, BCDOF)

        # create and return sparse matrix, we need to create a coo array first as this correctly handle duplicate
        # entries, then we convert to a csc array as that's what the scipy sparse linear solver likes to work with
        mat = coo_array((matEntries, (matRows, matColumns)), shape=(self.numDOF, self.numDOF))

        return csc_array(mat)

    def _assembleResidual(self, states: np.ndarray, applyBCs: Optional[bool] = True):
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
            nodeCoords = self.model.getElementCoordinates(elementType)
            nodeStates = self.getElementStates(elementType)
            elementDVs = self.model.getElementDVs(elementType)

            elementResiduals = element.computeResiduals(nodeStates, nodeCoords, elementDVs, self.constitutiveModel)
            AssemblyUtils.scatterLocalResiduals(elementResiduals, elementData["connectivity"], globalResidual)

        # Add external loads to the residual
        globalResidual -= AssemblyUtils.convertLoadsDictToVector(self.loads, self.numDOF)

        # Apply boundary conditions
        if applyBCs:
            BCs = self.getBCs()
            BCDOF, BCValues = AssemblyUtils.convertBCDictToLists(BCs)
            AssemblyUtils.applyBCsToVector(globalResidual, states.flatten(), BCDOF, BCValues)

        return globalResidual

    @staticmethod
    def _getDefaultOptions():
        """Return the default FEMpy model options"""
        defaultOptions = {
            "printTiming": [bool, False],
            "outputDir": [(str, type(None)), None],
            "outputFormat": [(str, type(None)), None],
        }
        return defaultOptions

    def _printTiming(self, times):
        resAssemblyTime = times["ResAssembled"] - times["Start"]
        matAssemblyTime = times["JacAssembled"] - times["ResAssembled"]
        solveTime = times["Solved"] - times["JacAssembled"]
        print("\n")
        print("+-----------------------------------------------------------------+")
        print(f" Timing information for FEMpy problem: {self.name}")
        print("+-----------------------------------------------------------------+")
        print("+ Residual Assembly: {:11.5e} s".format(resAssemblyTime))
        print("+ Jacobian Assembly: {:11.5e} s".format(matAssemblyTime))
        print("+ Linear Solution:   {:11.5e} s".format(solveTime))
        print("+-----------------------------------------------------------------+\n")

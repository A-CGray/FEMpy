"""
==============================================================================
FEMpy Model Class
==============================================================================
@File    :   Model.py
@Date    :   2022/11/14
@Author  :   Alasdair Christison Gray
@Description : This file contains the FEMpy model class, this is the main object
that the user interfaces with to read in a mesh and setup a finite element model.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
from typing import Iterable, Union, Optional
import copy
import warnings

# ==============================================================================
# External Python modules
# ==============================================================================
import meshio
import numpy as np
from baseclasses.solvers import BaseSolver
from scipy.sparse import csc_array  # ,coo_array

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy import Elements
from FEMpy import __version__


class FEMpyModel(BaseSolver):
    """_summary_

    A FEMpy model contains overall information about the finite element model, including but not limited to:
    - The node coordinates
    - The element-node associativity
    - The constitutive model
    - The constitutive model design variables

    It also contains methods for:
    - Setting/Getting node coordinates
    - Setting/Getting design variable values
    """

    # ==============================================================================
    # Public methods
    # ==============================================================================

    def __init__(self, meshFileName: str, constitutiveModel, options: Optional[dict] = None) -> None:
        """Create a FEMpy model

        Parameters
        ----------
        meshFileName : str
            Filename of the mesh file to load
        constitutiveModel : FEMpy constitutive model object
            Filename of the mesh file to load
        elementMap : function, optional
            A dictionary that maps from meshio element types to the FEMpy element types, the user can use this to
            specify what type of elements FEMpy should use, if the map returns None then this element type is ignored
            and not included in the model. If not supplied then the default FEMpy element type for each meshio element
            type will be used.
        options : dict, optional
            Dictionary of options to pass to the model, by default None, for a list of options see the documentation
        """

        # --- Set solver options by getting the defaults and updating them with any that the user supplied ---
        defaultOptions = self._getDefaultOptions()
        if options is None:
            options = {}

        # instantiate the solver
        super().__init__("FEMpy", "FEA Solver", defaultOptions=defaultOptions, options=options)

        self._printWelcomeMessage()

        # --- Save the mesh file name and extension ---
        self.meshFileName = meshFileName
        self.meshType = os.path.splitext(meshFileName)[1][1:].lower()

        # --- Read in the mesh using meshio ---
        self.mesh = meshio.read(self.meshFileName)
        self.numNodes = self.mesh.points.shape[0]

        # Extract mesh coordinates, detect whether the mesh is 1D, 2D or 3D and only keep the active dimensions
        self.nodeCoords = copy.deepcopy(self.mesh.points)
        self.activeDimensions = []

        # Keep track of the inactive coordinates so we can add them back in when writing output files
        self.inactiveDimensions = []
        for ii in range(3):
            if self.nodeCoords[:, ii].max() != self.nodeCoords[:, ii].min():
                self.activeDimensions.append(ii)
            else:
                self.inactiveDimensions.append(ii)

        if self.numDimensions != len(self.activeDimensions):
            raise ValueError(
                f"You have chosen a {self.numDimensions}D constitutive model model but the mesh is {len(self.activeDimensions)}D"
            )

        self.nodeCoords = self.nodeCoords[:, self.activeDimensions]

        # --- Set the consitutive model ---
        self.constitutiveModel = constitutiveModel
        self.numDOFs = self.numNodes * self.numStates

        # --- For each element type in the mesh, we need to assign a FEMpy element object ---
        self.elements = {}
        for elType in self.mesh.cells_dict:
            elObject = self._getElementObject(elType)
            if elObject is None:
                warnings.warn(f"Element type {elType} is not supported by FEMpy and will be ignored")
            else:
                self.elements[elType] = {}
                self.elements[elType]["connectivity"] = copy.deepcopy(self.mesh.cells_dict[elType])
                self.elements[elType]["DOF"] = self._getDOFfromNodeInds(self.elements[elType]["connectivity"])
                self.elements[elType]["elementObject"] = elObject
                self.elements[elType]["numElements"] = self.elements[elType]["connectivity"].shape[0]

                # --- Store dersign variable values for each element set ---
                # The constitutive model has a certain number of design variables, each of which has a name, we store
                # the values of the design variables for each element set in a dictionary with the design variable name
                # as the key and the values as a numElements array
                self.elements[elType]["DVs"] = {}

                for dvName in self.constitutiveModel.designVariables:
                    defaultVal = self.constitutiveModel.designVariables[dvName]["defaultValue"]
                    self.elements[elType]["DVs"][dvName] = defaultVal * np.ones(self.elements[elType]["numElements"])

        # --- List for keeping track of all problems associated with this model ---
        self.problems = []

        # --- Dictionary of global boundary conditions ---
        self.BCDict = {}

    @property
    def numDimensions(self):
        """Number of active dimensions in the model"""
        return self.constitutiveModel.numDimensions

    @property
    def numStates(self):
        """Number of states in the model"""
        return self.constitutiveModel.numStates

    def createOutputData(self, nodeValues={}, elementValues={}):
        """Create the meshio data structure for writing out results

        _extended_summary_

        Parameters
        ----------
        nodeValues : dictionary
            {"Temperature": array(one value per node), "Pressure": array()}
        elementValues : _type_
            {"elementType1":{"VariableName1": array(one value per element), "VariableName2": array()},
            "elementType2":{"VariableName1": array(one value per element), "VariableName2": array()}}

        Returns
        -------
        meshio mesh object
            Mesh object containing the results
        """
        if nodeValues:  # if dictionary is not empty
            for varName in nodeValues:
                # check arrays are correct length
                assert (
                    nodeValues[varName].shape[0] == self.numNodes
                ), f"nodeValues array for variable {varName} must be length of number of nodes"

        cellData = {}
        if elementValues:  # if dictionary is not empty
            for elType in elementValues:
                for varName in elementValues[elType]:
                    # first, check arrays are the correct length
                    assert (
                        elementValues[elType][varName].shape[0] == self.elements[elType]["numElements"]
                    ), f"elementValues array of element type {elType} for variable {varName} must be length number of {elType} elements"

                    # store values in meshio element data format
                    if varName in cellData:
                        cellData[varName].append(elementValues[elType][varName])
                    else:
                        cellData[varName] = [elementValues[elType][varName]]

        outputMesh = meshio.Mesh(
            self.getCoordinates(force3D=True), self.mesh.cells, point_data=nodeValues, cell_data=cellData
        )
        return outputMesh

    # Element.computeFunction(paramCoordinates, nodeCoordinates, nodeStates, elementDVs, function, elementReductionType)
    # problem.computeFunction(name="", elementReductionType, globalReductionType)
    # constitutiveModel.computeFunction(name="", other inputs)

    def getCoordinates(self, force3D: Optional[bool] = False) -> np.ndarray:
        """Get the current node coordinates

        Parameters
        ----------
        force3D : bool, optional
            If True then 3D coordinates will be returned, even if the model is 1D or 2D, by default False

        Returns
        -------
        numNodes x numDimensions array
            Node coordinates
        """
        currentCoords = np.copy(self.nodeCoords)
        if not force3D or self.numDimensions == 3:
            return currentCoords
        else:
            coords = np.zeros((self.numNodes, 3))
            coords[:, self.activeDimensions] = currentCoords
            return coords

    def setCoordinates(self, nodeCoords: np.ndarray) -> None:
        """Set the current node coordinates

        Parameters
        ----------
        nodeCoords : numNodes x numDimensions array
            Node coordinates
        """
        size = nodeCoords.shape[1]
        if size == self.numDimensions:
            return np.copy(nodeCoords)
        if self.numDimensions != 3 and size == 3:
            self.nodeCoords = nodeCoords[:, self.activeDimensions]
        else:
            raise ValueError(
                f"Invalid number of coordinate dimensions, problem is {self.numDimensions}D but {size}D coordinates were supplied"
            )

    def addGlobalFixedBC(
        self, name, nodeInds: Iterable[int], dof: Union[int, Iterable[int]], values: Union[float, Iterable[float]]
    ) -> None:
        """Add a boundary condition that is applied to all problems associated with this model

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

    def addProblem(self, name):
        """Add a problem to the model

        Parameters
        ----------
        name : str
            Name of the problem to add
        """
        return None

    def assembleMatrix(self, stateVector: np.ndarray, applyBCs: Optional[bool] = True) -> csc_array:
        """Assemble the global residual Jacobian matrix for the problem (a.k.a the stiffness matrix)

        _extended_summary_

        Parameters
        ----------
        stateVector : np.ndarray
            The current system states
        applyBCs : bool, optional
            Whether to modify the matrix to include the boundary conditions, by default True

        Returns
        -------
        scipy csc_array
            The residual Jacobian
        """ """"""
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

        for elementType, elementData in self.elements.items():
            element = elementData["elementObject"]
            numElements = elementData["numElements"]
            nodeCoords = np.zeros((numElements, element.numNodes, self.numDimensions))
            # nodeStates = np.zeros((numElements, numNodes, self.numStates))
            for ii in range(numElements):
                nodeInds = self.elements[elementType]["connectivity"][ii]
                nodeCoords[ii] = self.nodeCoords[nodeInds]

            # localMats = element.computeJacobian(self, nodeCoords, nodeStates, dvs, self.constitutiveModel)

        # assemble local matrices into global matrix

        return None

    # ==============================================================================
    # Private methods
    # ==============================================================================
    @staticmethod
    def _getDefaultOptions():
        """Return the default FEMpy model options"""
        defaultOptions = {
            "outputDir": [str, "./"],
        }
        return defaultOptions

    def _getElementObject(self, meshioName):
        """Given the meshio name for an element type return the corresponding FEMpy element object

        _extended_summary_

        Parameters
        ----------
        meshioName : str
            meshio element type name

        Returns
        -------
        An instantiated FEMpy element object
        """
        elName = meshioName.lower()
        elObject = None

        # --- 2D Quad elements ---
        if elName[:4] == "quad":
            if elName == "quad":
                elObject = Elements.QuadElement(order=1, numStates=self.numStates)
            else:
                numNodes = int(elName[4:])
                if numNodes == 8:
                    elObject = Elements.serendipityQuadElement(numStates=self.numStates)
                else:
                    # If the sqrt of the number of nodes is a whole number then this is a valid quad element
                    order = np.sqrt(numNodes) - 1
                    if int(order) == order:
                        elObject = Elements.QuadElement(order=int(order), numStates=self.numStates)

        # --- 2D Triangle elements ---
        if elName == "triangle":
            elObject = Elements.TriElement(order=1, numStates=self.numStates)
        if elName == "triangle6":
            elObject = Elements.TriElement(order=2, numStates=self.numStates)
        if elName == "triangle10":
            elObject = Elements.TriElement(order=3, numStates=self.numStates)

        # --- 1D Line Elements ---
        if elName[:4] == "line":
            if elName == "line":
                elObject = Elements.QuadElement(order=1, numStates=self.numStates)
            else:
                numNodes = int(elName[4:])
                order = numNodes - 1
                elObject = Elements.QuadElement(order=order, numStates=self.numStates)

        return elObject

    def _getDOFfromNodeInds(self, index):
        """Convert an array of node indices to an array of DOF indices

        Parameters
        ----------
        index : numpy array of ints
            array of node indices

        Returns
        -------
        numpy array of ints
            array of DOF indices
        """
        index_dof = []
        for i in range(len(index)):
            index_dof.append([])
            for j in range(len(index[i])):
                ind = range(index[i][j] * self.numStates, (index[i][j] + 1) * (self.numStates))
                index_dof[i] += list(ind)

        return np.array(index_dof)

    def _printWelcomeMessage(self) -> None:
        """Print a welcome message to the console"""
        self.pp("\nWelcome to")
        self.pp(
            """  ______ ______ __  __
|  ____|  ____|  \/  |
| |__  | |__  | \  / |_ __  _   _
|  __| |  __| | |\/| | '_ \| | | |
| |    | |____| |  | | |_) | |_| |
|_|    |______|_|  |_| .__/ \__, |
                    | |     __/ |
                    |_|    |___/
"""
        )
        self.pp("Version: " + __version__)
        self.pp("")

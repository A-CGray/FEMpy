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
from typing import Iterable, Union, Optional, Dict
import copy
import warnings

# ==============================================================================
# External Python modules
# ==============================================================================
import meshio
import numpy as np
from numba import njit
from baseclasses.solvers import BaseSolver

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy import Elements
from FEMpy import __version__

from Problem import FEMpyProblem


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
        self.numElements = 0
        for elType in self.mesh.cells_dict:
            elObject = self._getElementObject(elType)
            if elObject is None:
                warnings.warn(f"Element type {elType} is not supported by FEMpy and will be ignored")
            else:
                self.elements[elType] = {}
                self.elements[elType]["connectivity"] = copy.deepcopy(self.mesh.cells_dict[elType])
                self.elements[elType]["DOF"] = self.getDOFfromNodeInds(self.elements[elType]["connectivity"])
                self.elements[elType]["elementObject"] = elObject
                self.elements[elType]["numElements"] = self.elements[elType]["connectivity"].shape[0]

                # Create element indices for this set of elements, starting from the current max element index
                self.elements[elType]["elementIDs"] = np.arange(
                    self.numElements, self.numElements + self.elements[elType]["numElements"]
                )

                # Update the element count
                self.numElements += self.elements[elType]["numElements"]

        # --- Store dersign variable values for each element set ---
        # The constitutive model has a certain number of design variables, each of which has a name, we store
        # the values of each design variable
        self.dvs = {}

        for dvName in self.constitutiveModel.designVariables:
            defaultVal = self.constitutiveModel.designVariables[dvName]["defaultValue"]
            self.dvs[dvName] = defaultVal * np.ones(self.numElements)

        # --- List for keeping track of all problems associated with this model ---
        self.problems = {}

        # --- Dictionary of global boundary conditions ---
        self.BCs = {}

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
                ), f"nodeValues array for variable '{varName}' must be length of number of nodes"

        elementData = {}
        if elementValues:  # if dictionary is not empty
            for elType in elementValues:
                for varName in elementValues[elType]:
                    # first, check arrays are the correct length
                    assert (
                        elementValues[elType][varName].shape[0] == self.elements[elType]["numElements"]
                    ), f"elementValues array of element type '{elType}' for variable '{varName}' must be length number of '{elType}' elements"

                    # store values in meshio element data format
                    if varName in elementData:
                        elementData[varName].append(elementValues[elType][varName])
                    else:
                        elementData[varName] = [elementValues[elType][varName]]

        outputMesh = meshio.Mesh(
            self.getCoordinates(force3D=True), self.mesh.cells_dict, point_data=nodeValues, cell_data=elementData
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

    def setDesignVariables(self, dvs: Dict[str, np.ndarray]) -> None:
        """Set the design variables for the model

        Parameters
        ----------
        dvs : dictionary
            {"VariableName1": array(one value per element), "VariableName2": array()}
        """
        for dvName in dvs:
            self.dvs[dvName] = dvs[dvName]

    def getElementDVs(self, elementType: str):
        """Get the design variable values for a given element set

        _extended_summary_

        Parameters
        ----------
        elementType : str
            Element type name

        Returns
        -------
        dict of arrays
            Dictionary of design variable values for this element sets, one array per design variable
        """
        elementDVs = {}
        elementInds = self.elements[elementType]["elementIDs"]
        for dv in self.dvs:
            elementDVs[dv] = self.dvs[dv][elementInds]

        return elementDVs

    def addGlobalFixedBC(
        self,
        name,
        nodeInds: Union[int, Iterable[int]],
        dof: Union[int, Iterable[int]],
        value: Union[float, Iterable[float]],
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
        value : float or iterable of floats
            Value to fix states at, if a single value is supplied then this value is applied to all specified degrees of freedom
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

    def addProblem(self, name):
        """Add a problem to the model

        Parameters
        ----------
        name : str
            Name of the problem to add
        """
        self.problems[name] = FEMpyProblem(name, self)

    def getDOFfromNodeInds(self, nodeIndices):
        """Convert an array of node indices to an array of DOF indices

        Parameters
        ----------
        index : numpy array of ints
            array of node indices in arbitrary dimensions

        Returns
        -------
        numpy array of ints
            array of DOF indices, has same dimensions as input array, except that the last dimension is larger by a
            factor of the number of DOF per node
        """
        return _convertNodeIndsToDOF(nodeIndices, self.numStates)

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


@njit(cache=True)
def _convertNodeIndsToDOF(nodeIndices, numStates):
    # Figure out the output shape
    shape = nodeIndices.shape()
    shape[-1] *= numStates

    # Flatten the input array and convert to a flattened array of DOF indices
    nodeIndices = nodeIndices.flatten()
    dofIndices = []
    for i in range(len(nodeIndices)):
        ind = range(nodeIndices[i] * numStates, (nodeIndices[i] + 1) * (numStates))
        dofIndices += list(ind)

    return np.reshape(np.array(dofIndices), shape)

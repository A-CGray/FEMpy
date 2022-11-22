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
from typing import Iterable, Union
import copy
import warnings

# ==============================================================================
# External Python modules
# ==============================================================================
import meshio
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy import Elements


class FEMpyModel(object):
    """_summary_

    A FEMpy model contains overall information about the finite element model, including but not limited to:
    - The node coordinates
    - The element-node associativity
    - The constitutive model
    - The constitutive model design variables

    It also contains methods for:
    - Setting/Getting node coordinates
    - Setting/Getting design variable values

    Parameters
    ----------
    object : _type_
        _description_
    """

    # ==============================================================================
    # Public methods
    # ==============================================================================

    def __init__(self, meshFileName: str, constitutiveModel) -> None:
        """Create a FEMpy model

        _extended_summary_

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
        """

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
        self.nodeCoords = self.nodeCoords[:, self.activeDimensions]
        self.numDimensions = len(self.activeDimensions)

        # --- Set the consitutive model ---
        self.constitutiveModel = constitutiveModel
        self.numStates = self.constitutiveModel.numStates

        # --- For each element type in the mesh, we need to assign a FEMpy element object ---
        self.cells_dict = {}
        for elType in self.cells_dict:
            elObject = self._getElementObject(elType)
            if elObject is None:
                warnings.warn(f"Element type {elType} is not supported by FEMpy and will be ignored")
            else:
                self.cells_dict[elType] = copy.deepcopy(self.mesh.cells_dict[elType])
                self.cells_dict[elType]["FEMpy-Element"] = self._getElementObject(elType)

        # --- List for keeping track of all problems associated with this model ---
        self.problems = []

    def getCoordinates(self) -> np.ndarray:
        """Get the current node coordinates

        Returns
        -------
        numNodes x numDimensions array
            Node coordinates
        """
        return np.copy(self.nodeCoords)

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

    # ==============================================================================
    # Private methods
    # ==============================================================================
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

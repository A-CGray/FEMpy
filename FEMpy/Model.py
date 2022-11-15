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
from typing import Optional, Callable

# ==============================================================================
# External Python modules
# ==============================================================================
import meshio

# ==============================================================================
# Extension modules
# ==============================================================================


class FEMpyModel(object):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    object : _type_
        _description_
    """

    # ==============================================================================
    # Public methods
    # ==============================================================================

    def __init__(self, meshFileName: str, elemCallback: Optional[Callable] = None) -> None:
        """Create a FEMpy model

        _extended_summary_

        Parameters
        ----------
        meshFileName : str
            Filename of the mesh file to load
        elemCallback : function, optional
            An element type callback function which should return a FEMpy element type given a meshio element type name,
            if not supplied then the default FEMpy element type for each meshio element type will be used
        """

        # --- Save the mesh file name and extension ---
        self.meshFileName = meshFileName
        self.meshType = os.path.splitext(meshFileName)[1][1:].lower()

        # --- Read in the mesh using meshio ---
        self.mesh = meshio.read(self.meshFileName)
        self.numNodes = self.mesh.points.shape[0]

        # Extract mesh coordinates, detect whether the mesh is 1D, 2D or 3D and only keep the active dimensions
        self.nodeCoords = self.mesh.points
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

    # ==============================================================================
    # Private methods
    # ==============================================================================

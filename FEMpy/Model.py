"""
==============================================================================
FEMpy Model Class
==============================================================================
@File    :   Model.py
@Date    :   2022/11/14
@Author  :   Alasdair Christison Gray
@Description : This file contains the FEMpy model class, this is the main object that the user interfaces with to read in a mesh and setup a finite element model.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

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

    def __init__(self, meshFileName: str) -> None:

        # --- Save the mesh file name and extension ---
        self.meshFileName = meshFileName
        self.meshType = os.path.splitext(meshFileName)[1][1:].lower()

        # --- Read in the mesh using meshio ---
        self.mesh = meshio.read(self.meshFileName)

    @property
    def numNodes(self) -> int:
        """Get the number of mesh nodes

        Returns
        -------
        int
            Number of nodes in the mesh
        """
        return self.mesh.points.shape[0]

    @property
    def numElements(self) -> int:
        """Get the number of mesh elements

        meshio stores a separate CellBlock for each element type, so we need to sum the number of elements in each
        CellBlock

        Returns
        -------
        int
            Number of elements in the mesh
        """
        return sum([block.data.shape[0] for block in self.mesh.cells])

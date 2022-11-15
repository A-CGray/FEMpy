"""
==============================================================================
FEMpy Model class unit tests
==============================================================================
@File    :   testModel.py
@Date    :   2022/11/14
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest
from parameterized import parameterized_class
import os

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy import FEMpyModel

test_params = []

# Create a unit test for each mesh file in the Examples mesh folder
currentDir = os.path.dirname(os.path.realpath(__file__))
meshDir = "../Examples/Meshes/"
meshDir = os.path.join(currentDir, meshDir)
test_params.append({"meshFileName": os.path.join(meshDir, "WingboxL3.bdf"), "numPoints": 23158, "numElements": 24128})
test_params.append({"meshFileName": os.path.join(meshDir, "Plate.bdf"), "numPoints": 4225, "numElements": 4096})
test_params.append({"meshFileName": os.path.join(meshDir, "GMSHTest.msh"), "numPoints": 3622, "numElements": 798})


@parameterized_class(test_params)
class ModelUnitTest(unittest.TestCase):
    """Unit tests for the FEMpy model class"""

    def setUp(self) -> None:
        self.model = FEMpyModel(self.meshFileName)

    def testMeshRead(self):
        """Test that the mesh file was read in correctly"""
        self.assertEqual(self.model.numNodes, self.numPoints)
        self.assertEqual(self.model.numElements, self.numElements)


if __name__ == "__main__":
    unittest.main()

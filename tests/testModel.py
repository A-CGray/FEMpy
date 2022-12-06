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
import numpy as np

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy import FEMpyModel
import FEMpy as fp


test_params = []

# Create a unit test for each mesh file in the Examples mesh folder
currentDir = os.path.dirname(os.path.realpath(__file__))
meshDir = "../Examples/Meshes/"
meshDir = os.path.join(currentDir, meshDir)
# TODO: Fix the reference values for these tests
# test_params.append(
#     {"meshFileName": os.path.join(meshDir, "WingboxL3.bdf"), "numPoints": 23158, "numElements": 24128, "numDim": 3}
# )
test_params.append(
    {"meshFileName": os.path.join(meshDir, "Plate.bdf"), "numPoints": 4225, "numElements": 4096, "numDim": 2}
)
# test_params.append(
#     {"meshFileName": os.path.join(meshDir, "GMSHTest.msh"), "numPoints": 3622, "numElements": 798, "numDim": 2}
# )
# test_params.append(
#     {"meshFileName": os.path.join(meshDir, "LBracket.msh"), "numPoints": 9, "numElements": 798, "numDim": 2}
# )


# --- Create constitutive model, 7000 series Aluminium ---
E = 71.7e9
nu = 0.33
rho = 2

# This thickness value is a design variable, by default all elements will use this value, but we can change it later if
# we want
t = 5e-3
constitutiveModel = fp.Constitutive.IsoPlaneStress(E, nu, rho, t)


@parameterized_class(test_params)
class MeshReadTest(unittest.TestCase):
    """Unit tests for the FEMpy model class"""

    def setUp(self) -> None:
        self.model = FEMpyModel(constitutiveModel=constitutiveModel, meshFileName=self.meshFileName)

    def testMeshRead(self):
        """Test that the mesh file was read in correctly"""
        self.assertEqual(self.model.numNodes, self.numPoints)
        self.assertEqual(self.model.numDim, self.numDim)


class ModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FEMpyModel(constitutiveModel=constitutiveModel, meshFileName=os.path.join(meshDir, "Plate.bdf"))

    def testaddFixedBCToNodes(self):
        """Test that the BC are added correctly"""
        self.model.addFixedBCToNodes("myBC", [2, 3, 4], 0, 1)
        DOF_computed = [4, 6, 8]
        VAL_computed = [1.0, 1.0, 1.0]

        self.assertEqual(self.model.BCs["myBC"]["DOF"], DOF_computed)
        self.assertEqual(self.model.BCs["myBC"]["Value"], VAL_computed)

    def testgetDOFfromNodeInds(self):
        """Test that the BC are added correctly"""
        DOF_expected = self.model.getDOFfromNodeInds(np.array([3, 8, 11, 23]))
        DOF_computed = np.array([6, 7, 16, 17, 22, 23, 46, 47])

        np.testing.assert_equal(DOF_expected, DOF_computed)


if __name__ == "__main__":
    unittest.main()

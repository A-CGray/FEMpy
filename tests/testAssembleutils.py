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
import os
import numpy as np

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
# from FEMpy import FEMpyModel
from FEMpy.Utils import AssemblyUtils


class AssemUnitTest(unittest.TestCase):
    """Unit tests for the FEMpy model class"""

    def testConvertBCDictToLists(self):
        """Test that the mesh file was read in correctly"""
        BCDict = {
            "BC1Name": {"DOF": [0, 1, 2], "Value": [0, 0, 0]},
            "BC2Name": {"DOF": [13, 46, 1385], "Value": [1.0, 1.0, -1.0]},
            "BC3Name": {"DOF": [837, 25], "Value": [1.0, 1.0]},
        }
        bcDOF, bcValues = AssemblyUtils.convertBCDictToLists(BCDict)
        bcDOF_out = [0, 1, 2, 13, 46, 1385, 837, 25]
        bcValues_out = [0, 0, 0, 1.0, 1.0, -1.0, 1.0, 1.0]
        self.assertEqual(bcDOF, bcDOF_out)
        self.assertEqual(bcValues, bcValues_out)

    def testConvertLoadsDictToVector(self):
        """Test that the mesh file was read in correctly"""
        loadsDict = {
            "Load1Name": {"DOF": [0, 1, 2], "Value": [0, 0, -1]},
            "Load2Name": {"DOF": [4, 5, 8], "Value": [1.0, 1.0, -1.0]},
            "Load3Name": {"DOF": [2, 10], "Value": [3.0, 4.0]},
        }
        loads = AssemblyUtils.convertLoadsDictToVector(loadsDict, 12)
        loads_out = np.array([0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 4.0, 0.0])
        np.testing.assert_equal(loads, loads_out)

    def testLocalMatricesToCOOArrays(self):
        """Test that the mesh file was read in correctly"""

        localMats = np.array(
            [
                np.array(
                    [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 1, 0],
                        [0, 0, 0, 1],
                    ]
                ),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 2], [0, 0, 0, 2]]),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 2], [0, 0, 0, 2]]),
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 2], [0, 0, 0, 2]]),
            ]
        )
        COO_expected_rows = np.array([0, 1, 2, 2, 3, 2, 3, 4, 4, 5, 4, 5, 6, 6, 7, 6, 7, 0, 0, 1])
        COO_expected_cols = np.array([0, 1, 1, 2, 3, 2, 3, 4, 5, 5, 4, 5, 6, 7, 7, 6, 7, 0, 1, 1])
        COO_expected_vals = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2])
        localDOF = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 0, 1]])

        COO_comp_rows, COO_comp_cols, COO_comp_vals = AssemblyUtils.localMatricesToCOOArrays(localMats, localDOF)
        np.testing.assert_equal(COO_comp_rows, COO_expected_rows)
        np.testing.assert_equal(COO_comp_cols, COO_expected_cols)
        np.testing.assert_equal(COO_comp_vals, COO_expected_vals)


if __name__ == "__main__":
    unittest.main()

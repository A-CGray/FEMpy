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
from FEMpy import FEMpyModel
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


if __name__ == "__main__":
    unittest.main()

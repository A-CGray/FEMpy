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
from FEMpy.Constitutive.StressModels import IsoStress


class AssemUnitTest(unittest.TestCase):
    """Unit tests for the FEMpy model class"""

    # def testPlaneStrain(self):
    #     """Test that the mesh file was read in correctly"""

    #     self.assertEqual(bcValues, bcValues_out)

    # def testPlaneStress(self):
    #     """Test that the mesh file was read in correctly"""

    #     np.testing.assert_equal(loads, loads_out)

    def test3Dstress(self):
        """Test that the mesh file was read in correctly"""

        strain = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 1, 3, 3, 3]])
        E = 1000
        nu = 0.0
        const_3D = (
            E
            / (1 + nu)
            / (1 - 2 * nu)
            * np.array(
                [1 - nu, nu, nu, 0, 0, 0],
                [nu, 1 - nu, nu, 0, 0, 0],
                [nu, nu, 1 - nu, 0, 0, 0],
                [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
            )
        )
        strain_exptected = np.array([1000, 1000, 1000, 0, 0, 0], [1000, 1000, 1000, 1500, 1500, 1500])
        strain_computed = IsoStress.iso3DStress(strain, E, nu)

        np.testing.assert_equal(strain_computed, strain_exptected)


if __name__ == "__main__":
    unittest.main()

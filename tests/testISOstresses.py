"""
==============================================================================
FEMpy Model class unit tests
==============================================================================
@File    :   testModel.py
@Date    :   2022/11/14
@Author  :   M.A. Saja A.Kaiyoom and Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest
import numpy as np

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
# from FEMpy import FEMpyModel
from FEMpy.Constitutive.StressModels import IsoStress


class StressCalculationTest(unittest.TestCase):
    """Unit tests for the FEMpy stress computation"""

    def setUp(self):
        self.n = 10
        np.random.seed(1)
        self.tol = 1e-12

        self.E = 1000
        self.nu = 0.3

    def testPlanestress(self):
        """Test that the mesh file was read in correctly"""

        strain = np.random.rand(self.n, 3)
        E = self.E
        nu = self.nu
        mat = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        strain_exptected = np.zeros_like(strain)
        for i in range(self.n):
            strain_exptected[i] = mat @ strain[i]
        strain_computed = IsoStress.isoPlaneStressStress(strain, E, nu)

        np.testing.assert_allclose(strain_computed, strain_exptected, atol=self.tol, rtol=self.tol)

    def testPlanestrain(self):
        """Test that the mesh file was read in correctly"""

        strain = np.random.rand(self.n, 3)
        E = self.E
        nu = self.nu
        mat = E / ((1 + nu) * (1 - nu * 2)) * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])
        strain_exptected = np.zeros_like(strain)
        for i in range(self.n):
            strain_exptected[i] = mat @ strain[i]
        strain_computed = IsoStress.isoPlaneStrainStress(strain, E, nu)

        np.testing.assert_allclose(strain_computed, strain_exptected, atol=self.tol, rtol=self.tol)

    def test3Dstress(self):
        """Test that the mesh file was read in correctly"""

        strain = np.random.rand(self.n, 6)
        E = self.E
        nu = self.nu
        mat = (
            E
            / ((1 + nu) * (1 - nu * 2))
            * np.array(
                [
                    [1 - nu, nu, nu, 0, 0, 0],
                    [nu, 1 - nu, nu, 0, 0, 0],
                    [nu, nu, 1 - nu, 0, 0, 0],
                    [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                    [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                    [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
                ]
            )
        )
        strain_exptected = np.zeros_like(strain)
        for i in range(self.n):
            strain_exptected[i] = mat @ strain[i]
        strain_computed = IsoStress.iso3DStress(strain, E, nu)

        np.testing.assert_allclose(strain_computed, strain_exptected, atol=self.tol, rtol=self.tol)


if __name__ == "__main__":
    unittest.main()

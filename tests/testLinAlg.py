"""
==============================================================================
Linear Algebra tests
==============================================================================
@File    :   testLinAlg.py
@Date    :   2021/11/14
@Author  :   Alasdair Christison Gray
@Description : Simple unit tests to verify FEMPy's 2/3D linear algebra routines against numpy
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy import det1, det2, det3, inv1, inv2, inv3


class LinAlgUnitTest(unittest.TestCase):
    """Test FEMpy's Gauss quadrature integration against scipy's integration methods"""

    def setUp(self) -> None:
        """Create random matrix stacks to use in tests"""
        numMats = 1000
        self.precision = 8
        np.random.seed(0)
        self.Mats = []
        for i in range(1, 4):
            self.Mats.append(np.random.rand(numMats, i, i))

    def test1DDet(self):
        """Test 1D determinant"""
        np.testing.assert_almost_equal(det1(self.Mats[0]), np.linalg.det(self.Mats[0]), decimal=self.precision)

    def test2DDet(self):
        """Test 2D determinant"""
        np.testing.assert_almost_equal(det2(self.Mats[1]), np.linalg.det(self.Mats[1]), decimal=self.precision)

    def test3DDet(self):
        """Test 3D determinant"""
        np.testing.assert_almost_equal(det3(self.Mats[2]), np.linalg.det(self.Mats[2]), decimal=self.precision)

    def test1Dinv(self):
        """Test 1D inverse"""
        np.testing.assert_almost_equal(inv1(self.Mats[0]), np.linalg.inv(self.Mats[0]), decimal=self.precision)

    def test2Dinv(self):
        """Test 2D inverse"""
        np.testing.assert_almost_equal(inv2(self.Mats[1]), np.linalg.inv(self.Mats[1]), decimal=self.precision)

    def test3Dinv(self):
        """Test 3D inverse"""
        np.testing.assert_almost_equal(inv3(self.Mats[2]), np.linalg.inv(self.Mats[2]), decimal=self.precision)


if __name__ == "__main__":
    unittest.main()

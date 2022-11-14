"""
==============================================================================
Gauss Quadrature unit tests
==============================================================================
@File    :   testGaussQuad.py
@Date    :   2021/07/29
@Author  :   Alasdair Christison Gray
@Description : Unit tests for FEMpy's gauss quadrature integration scheme
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import scipy.integrate as integrate

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy.Quadrature.GaussQuad import gaussQuad1d, gaussQuad2d, gaussQuad3d


def TestFunc(x):
    f = 1.0
    for i in range(1, 10):
        f += x**i
    return f


def TestFunc2d(x1, x2):
    f = 1.0
    for i in range(1, 10):
        f += x1**i - 3.0 * x2**i
    return f


def TestFunc3d(x1, x2, x3):
    f = 1.0
    for i in range(1, 10):
        f += x1**i - 4.0 * x2**i + 3.0 * x3**i
    return f


def TestMatFunc3d(x1, x2, x3):
    A = np.zeros((len(x1), 3, 3))
    for i in range(len(x1)):
        A[i] = np.array([[x1[i], 2.0, 3.0], [1.0, x2[i], 3.0], [1.0, 2.0, x3[i]]])
    return A


class GaussQuadUnitTest(unittest.TestCase):
    """Test FEMpy's Gauss quadrature integration against scipy's integration methods"""

    def setUp(self) -> None:
        self.precision = 8

    def test_1d_gauss_quad(self):
        gaussInt = gaussQuad1d(TestFunc, 6)
        scipyInt = integrate.quad(TestFunc, -1.0, 1.0)[0]
        self.assertAlmostEqual(gaussInt, scipyInt, places=self.precision)

    def test_1d_gauss_quad_nonStandard_limits(self):
        gaussInt = gaussQuad1d(TestFunc, 6, a=-2.6, b=1.9)
        scipyInt = integrate.quad(TestFunc, -2.6, 1.9)[0]
        self.assertAlmostEqual(gaussInt, scipyInt, places=self.precision)

    def test_2d_gauss_quad(self):
        gaussInt = gaussQuad2d(TestFunc2d, 6)
        scipyInt = integrate.dblquad(TestFunc2d, -1.0, 1.0, -1.0, 1.0)[0]
        self.assertAlmostEqual(gaussInt, scipyInt, places=self.precision)

    def test_2d_gauss_quad_nonStandard_limits(self):
        gaussInt = gaussQuad2d(TestFunc2d, [6, 6], a=[-0.3, -2.3], b=[0.7, 1.6])
        scipyInt = integrate.dblquad(TestFunc2d, -2.3, 1.6, -0.3, 0.7)[0]
        self.assertAlmostEqual(gaussInt, scipyInt, places=self.precision)

    def test_3d_gauss_quad(self):
        gaussInt = gaussQuad3d(TestFunc3d, 6)
        scipyInt = integrate.tplquad(TestFunc3d, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)[0]
        self.assertAlmostEqual(gaussInt, scipyInt, places=self.precision)

    def test_3d_gauss_quad_nonStandard_limits(self):
        gaussInt = gaussQuad3d(TestFunc3d, [6, 6, 6], a=[-4.0, 2.0, 0.0], b=[1.0, 3.0, 4.0])
        scipyInt = integrate.tplquad(TestFunc3d, 0.0, 4.0, 2.0, 3.0, -4.0, 1.0)[0]
        self.assertAlmostEqual(gaussInt, scipyInt, places=self.precision)

    def test_3d_mat_gauss_quad(self):
        gaussInt = gaussQuad3d(TestMatFunc3d, 6, a=[-4.0, 2.0, 0.0], b=[1.0, 3.0, 4.0])
        trueInt = np.array([[-30.0, 40.0, 60.0], [20.0, 50.0, 60.0], [20.0, 40.0, 40.0]])
        np.testing.assert_allclose(gaussInt, trueInt, atol=10**-self.precision, rtol=10**-self.precision)


if __name__ == "__main__":
    unittest.main()

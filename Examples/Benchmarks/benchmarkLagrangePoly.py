"""
==============================================================================
Linear Algebra speed test
==============================================================================
@File    :   detTest.py
@Date    :   2021/11/14
@Author  :   Alasdair Christison Gray
@Description : Testing speed of computing determinants and inverses of 2x2 and 3x3 matrices with numpy's generic det function vs
manual implementations for 2x2 nd 3x3 matrices
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest
import time

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import FEMpy.Basis.LagrangePoly as LP

# ==============================================================================
# Extension modules
# ==============================================================================


class benchmarkLagrangePoly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Call numba methods with dummy arrays to compile them"""
        x = np.array([np.linspace(-1.0, 1.0, 4)])
        y = -np.ones_like(x)
        z = -np.ones_like(y)

        LP.LagrangePoly1d(x, 3)
        LP.LagrangePoly1dDeriv(x, 3)
        LP.LagrangePoly2d(x, y, 3)
        LP.LagrangePoly2dDeriv(x, y, 3)
        LP.LagrangePoly3d(x, y, z, 3)
        LP.LagrangePoly3dDeriv(x, y, z, 3)

        cls.x = x
        cls.y = y
        cls.z = z

    def setUp(self):
        self.numTests = int(2e5)

    def benchmark1d(self):
        start = time.time()
        for _ in range(self.numTests):
            LP.LagrangePoly1d(self.x, 3)
        end = time.time()
        print(f"LagrangePoly1d: {(end - start):9.5e} s")

    def benchmark1dDeriv(self):
        start = time.time()
        for _ in range(self.numTests):
            LP.LagrangePoly1dDeriv(self.x, 3)
        end = time.time()
        print(f"LagrangePoly1dDeriv: {(end - start):9.5e} s")

    def benchmark2d(self):
        start = time.time()
        for _ in range(self.numTests):
            LP.LagrangePoly2d(self.x, self.y, 3)
        end = time.time()
        print(f"LagrangePoly2d: {(end - start):9.5e} s")

    def benchmark2dDeriv(self):
        start = time.time()
        for _ in range(self.numTests):
            LP.LagrangePoly2dDeriv(self.x, self.y, 3)
        end = time.time()
        print(f"LagrangePoly2dDeriv: {(end - start):9.5e} s")

    def benchmark3d(self):
        start = time.time()
        for _ in range(self.numTests):
            LP.LagrangePoly3d(self.x, self.y, self.z, 3)
        end = time.time()
        print(f"LagrangePoly3d: {(end - start):9.5e} s")

    def benchmark3dDeriv(self):
        start = time.time()
        for _ in range(self.numTests):
            LP.LagrangePoly3dDeriv(self.x, self.y, self.z, 3)
        end = time.time()
        print(f"LagrangePoly3dDeriv: {(end - start):9.5e} s")

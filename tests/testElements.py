"""
==============================================================================
Element unit tests
==============================================================================
@File    :   testElements.py
@Date    :   2021/07/29
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest
from parameterized import parameterized_class

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy import QuadElement, serendipityQuadElement  # ,Lagrange1dElement

# --- Elements to test: ---
# QuadElement: 1st to 4th order
# SerendipityQuad
# Lagrange1DElement: 1st to 4th order

test_params = []

for el in [QuadElement, serendipityQuadElement]:  # ,Lagrange1dElement]:
    if el in [QuadElement]:  # ,Lagrange1dElement]:
        for order in range(1, 5):
            element = el(order=order)
            test_params.append({"element": element, "name": element.name})
    else:
        element = el()
        test_params.append({"element": element, "name": element.name})


@parameterized_class(test_params)
class ElementUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tol = 1e-10
        self.numTestPoints = 4
        np.random.seed(1)

    def testGetParamCoord(self):
        error = self.element.testGetParamCoord(self.numTestPoints, maxIter=400, tol=self.tol * 1e-3)
        np.testing.assert_allclose(error, 0, atol=self.tol, rtol=self.tol)

    def testShapeFunctionDerivatives(self):
        error = self.element.testShapeFunctionDerivatives(self.numTestPoints)
        np.testing.assert_allclose(error, 0, atol=self.tol, rtol=self.tol)

    def testShapeFunctionSum(self):
        shapeFuncSum = self.element.testShapeFunctionSum(self.numTestPoints)
        np.testing.assert_allclose(shapeFuncSum, 1, atol=self.tol, rtol=self.tol)


if __name__ == "__main__":
    unittest.main()

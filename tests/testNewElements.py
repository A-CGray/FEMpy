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
from FEMpy.Elements import QuadElement2D
from FEMpy.Constitutive import IsoPlaneStress

# --- Elements to test: ---
# QuadElement2D: 1st to 3rd order

test_params = []

for el in [QuadElement2D]:
    if el in [QuadElement2D]:
        for order in range(1, 4):
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

    def testShapeFunctionDerivatives(self):
        error = self.element.testShapeFunctionDerivatives(self.numTestPoints)
        np.testing.assert_allclose(error, 0, atol=self.tol, rtol=self.tol)

    def testShapeFunctionSum(self):
        shapeFuncSum = self.element.testShapeFunctionSum(self.numTestPoints)
        np.testing.assert_allclose(shapeFuncSum, 1, atol=self.tol, rtol=self.tol)

    def testIdentityJacobian(self):
        JacDiff = self.element.testIdentityJacobian(self.numTestPoints)
        np.testing.assert_allclose(JacDiff, 0, atol=self.tol, rtol=self.tol)

    def testStateGradient(self):
        stateGradDiff = self.element.testStateGradient(self.numTestPoints)
        np.testing.assert_allclose(stateGradDiff, 0, atol=self.tol, rtol=self.tol)

    def testGetClosestPoints(self):
        error = self.element.testGetClosestPoints(self.numTestPoints, tol=self.tol * 1e-3)
        np.testing.assert_allclose(error, 0, atol=self.tol * 1e5, rtol=self.tol * 1e5)

    def testZeroResidual(self):
        cm = IsoPlaneStress(1.0, 0.0, 1.0, 1.0)
        nodeCoordinates = self.element.getRandomElementCoordinates()
        nodeCoordinates = np.array([nodeCoordinates])
        nodeStates = np.zeros_like(nodeCoordinates)
        dvs = {"Thickness": np.array([1.0])}
        res = self.element.computeResiduals(nodeStates, nodeCoordinates, dvs, cm)
        self.assertEqual(res.shape, (1, self.element.numNodes, self.element.numStates))
        np.testing.assert_allclose(res, 0, atol=self.tol, rtol=self.tol)


if __name__ == "__main__":
    unittest.main()

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
import time

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy.Elements import QuadElement2D
from FEMpy.Constitutive import IsoPlaneStress, IsoPlaneStrain

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
        cm = IsoPlaneStrain(E=70e9, nu=0.3, rho=2700, t=1.0)
        numElements = 10
        nodeCoordinates = np.zeros((numElements, self.element.numNodes, self.element.numDim))
        for ii in range(numElements):
            nodeCoordinates[ii] = self.element.getRandomElementCoordinates()
        nodeStates = np.zeros_like(nodeCoordinates)
        dvs = {"Thickness": np.ones(numElements)}
        res = self.element.computeResiduals(nodeStates, nodeCoordinates, dvs, cm)
        self.assertEqual(res.shape, (numElements, self.element.numNodes, self.element.numStates))
        np.testing.assert_allclose(res, 0, atol=self.tol, rtol=self.tol)

    def testResidualJacobian(self):
        cm = IsoPlaneStrain(E=70e9, nu=0.3, rho=2700, t=1.0)
        numElements = 1
        nodeCoordinates = np.zeros((numElements, self.element.numNodes, self.element.numDim))
        for ii in range(numElements):
            nodeCoordinates[ii] = self.element.getRandomElementCoordinates()
        if self.element.numNodes == 4:
            nodeCoordinates[0] = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
        nodeStates = np.zeros_like(nodeCoordinates)
        dvs = {"Thickness": np.ones(numElements)}
        res = self.element.computeResidualJacobians(nodeStates, nodeCoordinates, dvs, cm)
        print("debug")


if __name__ == "__main__":
    unittest.main()

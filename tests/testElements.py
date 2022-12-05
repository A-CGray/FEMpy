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
from FEMpy.Elements import QuadElement2D, TriElement2D
from FEMpy.Constitutive import IsoPlaneStrain

# --- Elements to test: ---
# QuadElement2D: 1st to 3rd order

knownQuadStiffnessMat = 1.0e10 * np.array(
    [
        [4.4786, 2.3299, -1.5533, 0.7249, -3.1583, -2.0710, 0.2330, -0.9837],
        [2.3299, 4.6080, 0.0518, 1.5533, -2.0710, -2.8994, -0.3107, -3.2618],
        [-1.5533, 0.0518, 3.4430, -0.7766, -1.0873, -0.5695, -0.8025, 1.2944],
        [0.7249, 1.5533, -0.7766, 5.6435, -1.2426, -4.9704, 1.2944, -2.2263],
        [-3.1583, -2.0710, -1.0873, -1.2426, 5.7988, 2.5888, -1.5533, 0.7249],
        [-2.0710, -2.8994, -0.5695, -4.9704, 2.5888, 6.3166, 0.0518, 1.5533],
        [0.2330, -0.3107, -0.8025, 1.2944, -1.5533, 0.0518, 2.1228, -1.0355],
        [-0.9837, -3.2618, 1.2944, -2.2263, 0.7249, 1.5533, -1.0355, 3.9349],
    ]
)

test_params = []

cm = IsoPlaneStrain(E=70e9, nu=0.3, rho=2700, t=1.0)

for el in [QuadElement2D, TriElement2D]:
    if el in [QuadElement2D, TriElement2D]:
        for order in range(1, 4):
            element = el(order=order)
            test_params.append({"element": element, "name": element.name, "knownJac": False, "ConstitutiveModel": cm})
        if el == QuadElement2D:
            test_params[0]["knownJac"] = True
            test_params[0]["knownJacCoords"] = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
            test_params[0]["knownJacDVs"] = {"Thickness": 1.0}
            test_params[0]["knownJacMat"] = knownQuadStiffnessMat
    else:
        element = el()
        test_params.append({"element": element, "name": element.name, "ConstitutiveModel": cm})


@parameterized_class(test_params)
class ElementUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tol = 1e-10
        self.numTestPoints = 4
        np.random.seed(1)

        self.numElements = 10

        self.numPerts = 3
        self.stepSize = 1e-6
        self.relFDTol = 1e-5
        self.absFDTol = 1e-5

    def testShapeFunctionDerivatives(self):
        error = self.element.testShapeFunctionDerivatives(self.numTestPoints)
        np.testing.assert_allclose(error, 0, atol=self.tol, rtol=self.tol)

    def testShapeFunctionSum(self):
        shapeFuncSum = self.element.testShapeFunctionSum(self.numTestPoints)
        np.testing.assert_allclose(shapeFuncSum, 1, atol=self.tol, rtol=self.tol)

    def testIdentityJacobian(self):
        JacDiff = self.element.testIdentityJacobian(self.numTestPoints)
        np.testing.assert_allclose(JacDiff, 0, atol=self.tol, rtol=self.tol)

    def testInterpolation(self):
        error = self.element.testInterpolation(self.numTestPoints)
        np.testing.assert_allclose(error, 0, atol=self.tol, rtol=self.tol)

    def testStateGradient(self):
        stateGradDiff = self.element.testStateGradient(self.numTestPoints)
        np.testing.assert_allclose(stateGradDiff, 0, atol=self.tol, rtol=self.tol)

    # def testGetClosestPoints(self):
    #     self.skipTest("Not working robustly yet")
    #     error = self.element.testGetClosestPoints(self.numTestPoints, tol=self.tol * 1e-3)
    #     np.testing.assert_allclose(error, 0, atol=self.tol * 1e7, rtol=self.tol * 1e7)

    def testZeroResidual(self):
        nodeCoordinates = np.zeros((self.numElements, self.element.numNodes, self.element.numDim))
        for ii in range(self.numElements):
            nodeCoordinates[ii] = self.element.getRandomElementCoordinates()
        nodeStates = np.zeros_like(nodeCoordinates)
        dvs = {"Thickness": np.ones(self.numElements)}
        res = self.element.computeResiduals(nodeStates, nodeCoordinates, dvs, self.ConstitutiveModel)
        self.assertEqual(res.shape, (self.numElements, self.element.numNodes, self.element.numStates))
        np.testing.assert_allclose(res, 0, atol=self.tol, rtol=self.tol)

    def testResidualJacobian(self):
        """Test that the residual Jacobian is consistent with the residual using finite differences

        _extended_summary_
        """
        nodeCoordinates = np.zeros((self.numElements, self.element.numNodes, self.element.numDim))
        for ii in range(self.numElements):
            nodeCoordinates[ii] = self.element.getRandomElementCoordinates()

        nodeStates = np.random.rand(*(nodeCoordinates.shape))
        dvs = {"Thickness": np.random.rand(self.numElements)}

        # This is a special case where we know what the stiffness matrix should be
        if self.knownJac:
            nodeCoordinates[0] = self.knownJacCoords
            for dv, value in self.knownJacDVs.items():
                dvs[dv][0] = value

        # --- Compute the residual Jacobians ---
        Jac = self.element.computeResidualJacobians(nodeStates, nodeCoordinates, dvs, self.ConstitutiveModel)

        if self.knownJac:
            np.testing.assert_allclose(Jac[0], self.knownJacMat, rtol=1e-3)

        # --- Test some random finite difference perturbations of the residual against mat-vec products with the Jacobians ---
        res0 = self.element.computeResiduals(nodeStates, nodeCoordinates, dvs, self.ConstitutiveModel)
        for _ in range(self.numPerts):
            pert = np.random.rand(*(nodeStates.shape)) * self.stepSize - self.stepSize * 0.5
            resPert = self.element.computeResiduals(nodeStates + pert, nodeCoordinates, dvs, self.ConstitutiveModel)
            resChange = resPert - res0

            for ii in range(self.numElements):
                jacVecProd = Jac[ii].dot(pert[ii].flatten())
                np.testing.assert_allclose(resChange[ii].flatten(), jacVecProd, atol=self.absFDTol, rtol=self.relFDTol)


if __name__ == "__main__":
    unittest.main()

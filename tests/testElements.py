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
from FEMpy.Elements import QuadElement2D, TriElement2D, HexElement3D, LineElement1D
from FEMpy.Constitutive import IsoPlaneStrain, Iso3D, Iso1D

# --- Elements to test: ---
# QuadElement2D: 1st to 3rd order
# TruElement2D: 1st to 3rd order
# HexElement3D: 1st and 2rd order

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

knownLineStiffnessMat = 70e9 * 1e-2 * np.array([[1.0, -1.0], [-1.0, 1.0]])

testParams = []

cm1D = Iso1D(E=70e9, rho=2700, A=1e-2)
cm2D = IsoPlaneStrain(E=70e9, nu=0.3, rho=2700, t=1.0)
cm3D = Iso3D(E=70e9, nu=0.3, rho=2700)

# --- 1D element tests ---
for order in range(1, 6):
    element = LineElement1D(order=order)
    testParams.append({"element": element, "name": element.name, "knownJac": False, "ConstitutiveModel": cm1D})
    # We have a known true Jacobian for the 1st order line element
    if order == 1:
        testParams[-1]["knownJac"] = True
        testParams[-1]["knownJacCoords"] = np.array([[0.0], [1.0]])
        testParams[-1]["knownJacDVs"] = {"Area": 1e-2}
        testParams[-1]["knownJacMat"] = knownLineStiffnessMat

# --- 2D element tests ---

# Tri elements
for order in range(1, 4):
    element = TriElement2D(order=order)
    testParams.append({"element": element, "name": element.name, "knownJac": False, "ConstitutiveModel": cm2D})

# Quad elements
for order in range(1, 4):
    element = QuadElement2D(order=order)
    testParams.append({"element": element, "name": element.name, "knownJac": False, "ConstitutiveModel": cm2D})
    # We have a known true Jacobian for the 1st order quad element
    if order == 1:
        testParams[-1]["knownJac"] = True
        testParams[-1]["knownJacCoords"] = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 2.0]])
        testParams[-1]["knownJacDVs"] = {"Thickness": 1.0}
        testParams[-1]["knownJacMat"] = knownQuadStiffnessMat

# --- 3D element tests ---
for order in range(1, 3):
    element = HexElement3D(order=order)
    testParams.append({"element": element, "name": element.name, "knownJac": False, "ConstitutiveModel": cm3D})


@parameterized_class(testParams)
class ElementUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tol = 1e-10
        self.numTestPoints = 4
        self.rng = np.random.default_rng(1)
        np.random.seed(1)

        self.numElements = 10

        self.numPerts = 3
        self.stepSize = 1e-6
        self.relFDTol = 1e-5
        self.absFDTol = 1e-5

    def testShapeFunctionDerivatives(self):
        error = self.element.testShapeFunctionDerivatives(self.numTestPoints, rng=self.rng)
        np.testing.assert_allclose(error, 0, atol=self.tol, rtol=self.tol)

    def testShapeFunctionSum(self):
        shapeFuncSum = self.element.testShapeFunctionSum(self.numTestPoints, rng=self.rng)
        np.testing.assert_allclose(shapeFuncSum, 1, atol=self.tol, rtol=self.tol)

    def testIdentityJacobian(self):
        JacDiff = self.element.testIdentityJacobian(self.numTestPoints, rng=self.rng)
        np.testing.assert_allclose(JacDiff, 0, atol=self.tol, rtol=self.tol)

    def testInterpolation(self):
        error = self.element.testInterpolation(self.numTestPoints, rng=self.rng)
        np.testing.assert_allclose(error, 0, atol=self.tol, rtol=self.tol)

    def testStateGradient(self):
        stateGradDiff = self.element.testStateGradient(self.numTestPoints, rng=self.rng)
        np.testing.assert_allclose(stateGradDiff, 0, atol=self.tol, rtol=self.tol)

    def testGetClosestPoints(self):
        """Test that the closest points are found correctly

        This test works by chossing some random parametric coordinates, computing their real coordinates and then
        checking that we get back the same parametric coordinates when we ask for the closest points to the real coordinates.
        """
        # self.skipTest("Not working robustly yet")
        error = self.element.testGetClosestPoints(self.numTestPoints, tol=self.tol * 1e-3, rng=self.rng)
        np.testing.assert_allclose(error, 0, atol=self.tol * 1e7, rtol=self.tol * 1e7)

    def testZeroResidual(self):
        """Validate that the residual is zero if all states are zero"""
        nodeCoordinates = np.zeros((self.numElements, self.element.numNodes, self.element.numDim))
        for ii in range(self.numElements):
            nodeCoordinates[ii] = self.element.getRandomElementCoordinates(rng=self.rng)
        nodeStates = np.zeros_like(nodeCoordinates)
        dvs = {}
        for dv in self.ConstitutiveModel.designVariables:
            dvs[dv] = self.rng.random(self.numElements)
        res = self.element.computeResiduals(nodeStates, nodeCoordinates, dvs, self.ConstitutiveModel)
        self.assertEqual(res.shape, (self.numElements, self.element.numNodes, self.element.numStates))
        np.testing.assert_allclose(res, 0, atol=self.tol, rtol=self.tol)

    def testResidualJacobian(self):
        """Test that the residual Jacobian is consistent with the residual using finite differences"""
        nodeCoordinates = np.zeros((self.numElements, self.element.numNodes, self.element.numDim))
        for ii in range(self.numElements):
            nodeCoordinates[ii] = self.element.getRandomElementCoordinates(rng=self.rng)

        nodeStates = self.rng.random(nodeCoordinates.shape)
        dvs = {}
        for dv in self.ConstitutiveModel.designVariables:
            dvs[dv] = self.rng.random(self.numElements)

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
            pert = self.rng.random(nodeStates.shape) * self.stepSize - self.stepSize * 0.5
            resPert = self.element.computeResiduals(nodeStates + pert, nodeCoordinates, dvs, self.ConstitutiveModel)
            resChange = resPert - res0

            for ii in range(self.numElements):
                jacVecProd = Jac[ii].dot(pert[ii].flatten())
                np.testing.assert_allclose(resChange[ii].flatten(), jacVecProd, atol=self.absFDTol, rtol=self.relFDTol)


if __name__ == "__main__":
    unittest.main()

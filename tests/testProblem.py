"""
==============================================================================
Unit test for FEMpy matrix assembly
==============================================================================
@File    :   testAssembly.py
@Date    :   2022/12/03
@Author  :   Alasdair Christison Gray
@Description :
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
import FEMpy as fp


class testProblem(unittest.TestCase):
    def setUp(self):
        np.random.rand(1)
        # --- Create constitutive model, 7000 series Aluminium ---
        E = 71.7e9
        nu = 0.33
        rho = 2780

        # This thickness value is a design variable, by default all elements will use this value, but we can change it later if
        # we want
        t = 5e-3
        constitutiveModel = fp.Constitutive.IsoPlaneStress(E, nu, rho, t)

        # --- Define node coordinates and element connectivity arrays ---
        # Mesh looks like this:

        # 0 --------------- 2 --------------- 4
        # |                 |                 |
        # |                 |                 |
        # |                 |                 |
        # |                 |                 |
        # |                 |                 |
        # 1 --------------- 3 --------------- 5

        self.nodeCoordinates = np.array([[0.0, 1.0], [0.0, 0.0], [1.2, 1.0], [0.9, 0.0], [2.0, 1.0], [2.0, 0.0]])
        # Add a random perturbation to the coordinates so that there is no cancellation of terms from the local
        # stiffness matrices
        self.nodeCoordinates += 1e-14 * np.random.rand(*self.nodeCoordinates.shape)
        self.connectivity = {"quad": np.array([[0, 1, 3, 2], [2, 3, 5, 4]])}

        self.model = fp.FEMpyModel(constitutiveModel, nodeCoords=self.nodeCoordinates, connectivity=self.connectivity)
        self.problem = self.model.addProblem("Test")
        self.problem.addFixedBCToNodes(name="YFixed", nodeInds=[1, 5], dof=1, value=1.0)
        self.problem.addFixedBCToNodes(name="XFixed", nodeInds=5, dof=0, value=0.1)
        self.problem.addLoadToNodes(name="YLoad", nodeInds=2, dof=1, value=-10.0)

        self.numDOF = self.nodeCoordinates.shape[0] * 2

        expectedSparsity = np.zeros((self.numDOF, self.numDOF))
        expectedSparsity[:4, :8] = 1
        expectedSparsity[4:8, :] = 1
        expectedSparsity[8:, 4:] = 1
        self.expectedSparsity = expectedSparsity

    def testSparsityNoBC(self):
        """Check the matrix sparsity pattern for a simple mesh with no boundary conditions"""
        self.problem.updateJacobian(applyBCs=False)
        self.matSparsityTest()

    def testSolveWithBC(self):
        """Check that the sparsity is as expected when some BCs are applied

        The row associated with each constrained DOF should be zero except for the diagonal entry
        """

        expectedRHS = np.zeros(self.numDOF)
        expectedRHS[3] = -1.0
        expectedRHS[5] = 10.0
        expectedRHS[11] = -1.0
        expectedRHS[10] = -0.1

        self.problem.updateResidual()
        np.testing.assert_equal(self.problem.Res, expectedRHS, err_msg="RHS is not as expected")

        self.expectedSparsity[[3, 10, 11], :] = 0
        self.expectedSparsity[[3, 10, 11], [3, 10, 11]] = 1.0

        # Test that the BCs are applied correctly (i.e. the solution has the correct values)
        self.problem.solve()
        self.problem.updateResidual()
        np.testing.assert_allclose(
            self.problem.Res, np.zeros(self.numDOF), rtol=1e-6, atol=1e-6, err_msg="RHS is not zero after solving"
        )
        u = self.problem.states.flatten()
        np.testing.assert_approx_equal(u[3], 1.0, significant=10, err_msg="BC not applied correctly")
        np.testing.assert_approx_equal(u[11], 1.0, significant=10, err_msg="BC not applied correctly")
        np.testing.assert_approx_equal(u[10], 0.1, significant=10, err_msg="BC not applied correctly")

        self.matSparsityTest()

    def matSparsityTest(self):
        expectedNonZeros = np.nonzero(self.expectedSparsity)
        K = self.problem.Jacobian
        KNonZeros = K.nonzero()

        comparisons = {
            "Shape": [K.shape, self.expectedSparsity.shape],
            "Row Sparsity": [KNonZeros[0], expectedNonZeros[0]],
            "Column Sparsity": [KNonZeros[1], expectedNonZeros[1]],
        }
        for name, values in comparisons.items():
            with self.subTest(values=values):
                np.testing.assert_equal(
                    values[0],
                    values[1],
                    err_msg=f"{name} of the assembled matrix is not as expected",
                )


if __name__ == "__main__":
    unittest.main()

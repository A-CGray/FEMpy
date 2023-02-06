"""
==============================================================================
FEMpy 2D Quad Mesh benchmark case
==============================================================================
@File    :   QuadMeshScaling.py
@Date    :   2021/05/04
@Author  :   Alasdair Christison Gray
@Description : This file contains a simple 2D case using quad elements that is
used to benchmark the performance of FEMpy as part of a CI job. The case uses a
223x223 mesh of 2D quad elements that results in almost exactly 100k degrees of
freedom
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import FEMpy as fp
from numba import njit

# ==============================================================================
# Extension modules
# ==============================================================================


@njit(cache=True)
def warpFunc(x, y):
    return 2.0 * x, (2.0 - x) * y


# @njit( cache=True)
def createGridMesh(nx, ny, warpFunc=None):
    xNodes = np.tile(np.linspace(0.0, 1.0, nx + 1), ny + 1)
    yNodes = np.repeat(np.linspace(0.0, 1.0, ny + 1), nx + 1)

    # Warp the mesh
    if warpFunc is not None:
        xNodes, yNodes = warpFunc(xNodes, yNodes)

    # Create Connectivity matrix
    numEl = nx * ny
    conn = np.zeros((numEl, 4), dtype=int)
    conn[:, 0] = np.tile(np.arange(nx), ny) + np.repeat((nx + 1) * np.arange(ny), nx)  # Lower left
    conn[:, 1] = conn[:, 0] + 1  # lower right
    conn[:, 2] = conn[:, 1] + nx  # upper right
    conn[:, 3] = conn[:, 2] + 1  # upper left

    return np.array([xNodes, yNodes]).T, conn


class BenchmarkQuadMeshSolve(unittest.TestCase):
    def setUp(self):
        self.con = fp.Constitutive.IsoPlaneStrain(E=70e9, nu=0.3, t=1.0, rho=2700.0)
        self.nodeCoords, self.conn = createGridMesh(317, 317, warpFunc=warpFunc)
        self.conn = {"quad": self.conn}
        self.model = fp.FEMpyModel(self.con, nodeCoords=self.nodeCoords, connectivity=self.conn)
        self.prob = self.model.addProblem("Static", options={"printTiming": True})
        rightEdgeNodeInds = np.argwhere(self.nodeCoords[:, 0] == 2.0).flatten()
        leftEdgeNodeInds = np.argwhere(self.nodeCoords[:, 0] == 0.0).flatten()
        self.prob.addFixedBCToNodes("Fixed", leftEdgeNodeInds, dof=[0, 1], value=0.0)
        self.prob.addLoadToNodes("Load", rightEdgeNodeInds, dof=[0], value=1e4)

        # This is just to make sure everything is JIT compiled before the timing starts
        self.prob.solve()
        self.prob.reset()

        self.numReps = 3

    def benchmarkResidualAssembly(self):
        """Benchmark the cost of assembling the residual"""
        for _ in range(self.numReps):
            self.prob.markResOutOfDate()
            self.prob.updateResidual(applyBCs=True)

    def benchmarkResidualAssemblyNoBC(self):
        """Benchmark the cost of assembling the residual without applying any boundary conditions"""
        for _ in range(self.numReps):
            self.prob.markResOutOfDate()
            self.prob.updateResidual(applyBCs=False)

    def benchmarkJacobianAssembly(self):
        """Benchmark the cost of assembling the residual jacobian"""
        for _ in range(self.numReps):
            self.prob.markJacOutOfDate()
            self.prob.updateJacobian(applyBCs=True)

    def benchmarkJacobianAssemblyNoBC(self):
        """Benchmark the cost of assembling the residual jacobian without applying any boundary conditions"""
        for _ in range(self.numReps):
            self.prob.markJacOutOfDate()
            self.prob.updateJacobian(applyBCs=False)

    def benchmarkSolve(self):
        """Benchmark the cost of assembling and solving the problem"""
        for _ in range(self.numReps):
            self.prob.markResOutOfDate()
            self.prob.markJacOutOfDate()
            self.prob.solve()


if __name__ == "__main__":
    unittest.main()

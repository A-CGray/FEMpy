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

from numba import njit

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import FEMpy as fp


# ==============================================================================
# Extension modules
# ==============================================================================


@njit(cache=True)
def warpFunc(x, y):
    return x, 0.02 * y


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
    conn[:, 2] = conn[:, 1] + nx + 1  # upper right
    conn[:, 3] = conn[:, 2] - 1  # upper left

    return np.array([xNodes, yNodes]).T, conn


if __name__ == "__main__":
    refineVal = [1, 2, 4, 8, 16, 32, 64, 128]
    Error = []
    numDOF = []
    for refine in refineVal:

        # create constitutive model
        con = fp.Constitutive.IsoPlaneStrain(E=70e9, nu=0.0, t=1.0, rho=2700.0)

        # create mesh
        nodeCoords, conn = createGridMesh(10 * refine, 1 * refine, warpFunc=warpFunc)
        conn = {"quad": conn}

        # create model
        model = fp.FEMpyModel(con, nodeCoords=nodeCoords, connectivity=conn, options={"outputFormat": ".dat"})
        prob = model.addProblem("Static")

        # Find the right, left and top edges
        rightEdgeNodeInds = np.argwhere(nodeCoords[:, 0] == 1.0).flatten()
        topEdgeNodeInds = np.argwhere(nodeCoords[:, 1] == np.max(nodeCoords[:, 1])).flatten()
        leftEdgeNodeInds = np.argwhere(nodeCoords[:, 0] == 0.0).flatten()

        # apply BCs
        prob.addFixedBCToNodes("Fixed", leftEdgeNodeInds, dof=[0, 1], value=0.0)
        prob.addLoadToNodes("Load", topEdgeNodeInds, dof=[1], value=-(10**3), totalLoad=True)

        # Calculate analytic displacement
        momInertia = 1 / 12 * 0.02**3
        dmax_analytic = 10**3 / (8 * 70e9 * momInertia)

        # solve the problem
        prob.solve()
        prob.writeSolution(baseName="mesh%s.dat" % refine)

        # average the displacement from numerical model and compute the error
        averageDisplacement = np.average(prob.states[rightEdgeNodeInds, 1])
        Error.append(np.abs(-averageDisplacement - dmax_analytic))
        numDOF.append(prob.numDOF)

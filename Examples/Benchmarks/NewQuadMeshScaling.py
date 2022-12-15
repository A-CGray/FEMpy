"""
==============================================================================
FEMpy 2D Quad Mesh benchmark case
==============================================================================
@File    :   QuadMeshScaling.py
@Date    :   2021/05/04
@Author  :   Jasmin Lim
             Modeled by script written by: Alasdair Christison Gray
@Description : This file contains a simple 2D case using quad elements that is
used to benchmark the performance of FEMpy as part of a CI job. The case uses a
nxn mesh of 2D quad elements, comparing the run time for each
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import time
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import FEMpy as fp
from numba import njit

import matplotlib.pyplot as plt
import niceplots

niceplots.setRCParams()
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
    conn[:, 2] = conn[:, 1] + nx + 1  # upper right
    conn[:, 3] = conn[:, 2] - 1  # upper left

    return np.array([xNodes, yNodes]).T, conn


def solve_problem(n):
    con = fp.Constitutive.IsoPlaneStrain(E=70e9, nu=0.3, t=1.0, rho=2700.0)
    nodeCoords, conn = createGridMesh(n, n, warpFunc=warpFunc)
    conn = {"quad": conn}
    model = fp.FEMpyModel(con, nodeCoords=nodeCoords, connectivity=conn)
    prob = model.addProblem("Static", options={"printTiming": True})
    rightEdgeNodeInds = np.argwhere(nodeCoords[:, 0] == 2.0).flatten()
    leftEdgeNodeInds = np.argwhere(nodeCoords[:, 0] == 0.0).flatten()
    prob.addFixedBCToNodes("Fixed", leftEdgeNodeInds, dof=[0, 1], value=0.0)
    prob.addLoadToNodes("Load", rightEdgeNodeInds, dof=[0], value=1e4)

    # This is just to make sure everything is JIT compiled before the timing starts
    times = prob.solve()
    prob.reset()

    return nodeCoords, times


if __name__ == "__main__":
    numEl = [2, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    forceIntTimeList = []
    assemblyTimeList = []
    solveTimeList = []
    totalTimeList = []
    numDOFList = []

    # run problem
    for i in range(len(numEl)):
        for j in range(2):
            nodeCoords, times = solve_problem(numEl[i])

            numNodes = np.shape(nodeCoords)[0]
            if j == 0:
                numDOFList.append(2 * numNodes)

            resAssemblyTime = times["ResAssembled"] - times["Start"]
            matAssemblyTime = times["JacAssembled"] - times["ResAssembled"]
            solveTime = times["Solved"] - times["JacAssembled"]
            totalTime = resAssemblyTime + matAssemblyTime + solveTime

            if j == 0:
                forceIntTimeList.append(resAssemblyTime)
                assemblyTimeList.append(matAssemblyTime)
                solveTimeList.append(solveTime)
                totalTimeList.append(totalTime)

            if j == 1:
                forceIntTimeList[-1] = min(forceIntTimeList[-1], resAssemblyTime)
                assemblyTimeList[-1] = min(assemblyTimeList[-1], matAssemblyTime)
                solveTimeList[-1] = min(solveTimeList[-1], solveTime)
                totalTimeList[-1] = min(totalTimeList[-1], totalTime)

    # plot results
    plotVars_old = np.genfromtxt("QuadMeshScaling.csv", delimiter=",")  # get old FEMpy

    plotVars_new = [forceIntTimeList, assemblyTimeList, solveTimeList, totalTimeList]
    plotVarNames = ["Residual Assembly", "Matrix Assembly", "Linear Solution", "Total"]
    plotFEMpy = ["Old FEMpy", "New FEMpy"]
    markers = ["-o", "--o"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i, plotVars in enumerate([plotVars_old, plotVars_new]):
        if i == 1:
            ax.set_xlabel("DOF")
            ax.set_xscale("log")
            ax.set_ylabel("Time\n(s)", rotation="horizontal", ha="right")
            ax.set_yscale("log")
            # ax.set_title(plotFEMpy[i])
            niceplots.adjust_spines(ax, outward=True)

            for v, name in zip(plotVars, plotVarNames):
                ax.plot(numDOFList[1:], v[1:], "-o", markeredgecolor="w", label=name, clip_on=False)

            ax.set_xticks(numDOFList[1:])
            ax.set_xticklabels(numDOFList[1:])
            ax.legend(labelcolor="linecolor")

    fig.savefig("../../docs/docs/Images/NewQuadElScaling.png", dpi=400)
    # plt.show()

    np.savetxt("NewQuadMeshScaling.csv", plotVars, delimiter=",")

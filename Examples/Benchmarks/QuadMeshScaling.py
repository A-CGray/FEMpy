"""
==============================================================================
FEMpy scaling test
==============================================================================
@File    :   QuadMeshScaling.py
@Date    :   2021/05/04
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import time

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
try:
    from pypardiso import spsolve
    solverArgs = {"factorize":False}
except ModuleNotFoundError:
    from scipy.sparse.linalg import spsolve
    solverArgs = {}
import FEMpy as fp
from numba import jit
import matplotlib.pyplot as plt
import niceplots

niceplots.setRCParams()
# ==============================================================================
# Extension modules
# ==============================================================================


@jit(nopython=True, cache=True)
def tractionForce(x):
    ft = np.zeros_like(x)
    ft[:, 0] = 1e6
    return ft


@jit(nopython=True, cache=True)
def bodyForce(x):
    fb = np.zeros_like(x)
    fb[:, 1] = 1e6
    return fb


@jit(nopython=True, cache=True)
def warpFunc(x, y):
    return 2.0 * x, (2.0 - x) * y


# @jit(nopython=True, cache=True)
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
    conn[:, 2] = conn[:, 0] + nx + 1  # upper left
    conn[:, 3] = conn[:, 2] + 1  # upper right

    return np.array([xNodes, yNodes]).T, conn


El = fp.QuadElement()
con = fp.isoPlaneStrain(E=70e9, nu=0.3, t=1.0)

numEl = [2, 5, 10, 20, 40, 80, 160, 320]
forceIntTimeList = []
assemblyTimeList = []
solveTimeList = []
totalTimeList = []
numDOFList = []

for e in numEl:
    minforceIntTime = np.inf
    minassemblyTime = np.inf
    minsolveTime = np.inf

    nodeCoords, conn = createGridMesh(e, e, warpFunc=warpFunc)
    nodeEls = fp.makeNodeElsMat(conn)
    numNodes = np.shape(nodeCoords)[0]
    numDOFList.append(2 * numNodes)

    leftEdgeNodes = np.argwhere(nodeCoords[:, 0] == 0.0).flatten()
    leftEdgeDisp = 0.0

    knownDisp = np.empty(numNodes * 2)
    knownDisp[:] = np.nan
    knownDisp[leftEdgeNodes * 2] = leftEdgeDisp
    knownDisp[leftEdgeNodes * 2 + 1] = leftEdgeDisp

    for _ in range(3):

        startTime = time.time()

        # --- Get elements and edge numbers associated with right hand side traction---
        rightEdgeNodes = np.argwhere(nodeCoords[:, 0] == 2.0).flatten()
        edgeInds = [[0, 1], [1, 3], [3, 2], [2, 0]]
        TractElems, TractEdges = fp.getEdgesfromNodes(rightEdgeNodes, conn, nodeEls, edgeInds)

        FTract = fp.assembleTractions(nodeCoords, conn, El, con, TractElems, TractEdges, tractionForce, knownDisp)
        FBody = fp.assembleBodyForce(nodeCoords, conn, El, con, bodyForce, knownDisp, n=2)
        forceIntTime = time.time() - startTime

        K, RHS = fp.assembleMatrix(nodeCoords, conn, El, con, knownDisp)

        assemblyTime = time.time() - forceIntTime - startTime

        u = spsolve(K, RHS + FTract + FBody, **solverArgs)
        # u = u.reshape(len(u) // 2, 2)

        solveTime = time.time() - startTime - assemblyTime - forceIntTime

        minforceIntTime = min(minforceIntTime, forceIntTime)
        minassemblyTime = min(minassemblyTime, assemblyTime)
        minsolveTime = min(minsolveTime, solveTime)

        print("\n-------------------")
        print(f"{e**2} Elements")
        print(f"{numNodes} Nodes")
        print(f"{numNodes*2} DOF")
        print(f"Force integration: {forceIntTime:03e} s")
        print(f"Matrix assembly: {assemblyTime:03e} s")
        print(f"Linear System solution: {solveTime:03e} s")
        print("-------------------")
    forceIntTimeList.append(minforceIntTime)
    assemblyTimeList.append(minassemblyTime)
    solveTimeList.append(minsolveTime)
    totalTimeList.append(minforceIntTime + minassemblyTime + minsolveTime)

fig, ax = plt.subplots()
ax.set_xlabel("DOF")
ax.set_xscale("log")
ax.set_ylabel("Time\n(s)", rotation="horizontal", ha="right")
ax.set_yscale("log")
niceplots.adjust_spines(ax, outward=True)

plotVars = [forceIntTimeList, assemblyTimeList, solveTimeList, totalTimeList]
plotVarNames = ["Force Assembly", "Matrix Assembly", "Linear Solution", "Total"]

for v, name in zip(plotVars, plotVarNames):
    ax.plot(numDOFList, v, "-o", markeredgecolor="w", label=name, clip_on=False)

ax.set_xticks(numDOFList)
ax.set_xticklabels(numDOFList)
ax.legend(labelcolor="linecolor")
fig.savefig("../../docs/docs/Images/QuadElScaling.png", dpi=400)
plt.show()

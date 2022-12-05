"""
==============================================================================
Testing 9 node quad element
==============================================================================
@File    :   NineNodeQuadTest.py
@Date    :   2022/12/05
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Extension modules
# ==============================================================================
import FEMpy as fp


"""
The element looks like this
3-----------------6-----------------2
|                 |                 |
|                 |                 |
7-----------------8-----------------5
|                 |                 |
|                 |                 |
0 ----------------4-----------------1

Shape func node ordering:
6-----------------7-----------------8
|                 |                 |
|                 |                 |
3-----------------4-----------------5
|                 |                 |
|                 |                 |
0 ----------------1-----------------2

Node 0 is fixed in Y
Node 1 is fixed in X and Y
There is a downward vertical force at node 6
"""

nodeCoords = np.array(
    [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 1.0], [1.0, 1.0]]
)
connectivity = {"quad9": np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])}

E = 71.7e9
nu = 0.33
rho = 2780

# This thickness value is a design variable, by default all elements will use this value, but we can change it later if
# we want
t = 5e-3
constitutiveModel = fp.Constitutive.IsoPlaneStress(E, nu, rho, t)

model = fp.FEMpyModel(constitutiveModel, nodeCoords=nodeCoords, connectivity=connectivity)
problem = model.addProblem("Test")

problem.addFixedBCToNodes(name="YFixed", nodeInds=[0, 1], dof=1, value=0.0)
problem.addFixedBCToNodes(name="XFixed", nodeInds=[0, 1], dof=0, value=0.0)

problem.addLoadToNodes(name="Force", nodeInds=6, dof=1, value=-1e3)

problem.solve()

# Plot the deformed shape
plt.plot(nodeCoords[:, 0], nodeCoords[:, 1], "o")
plt.quiver(nodeCoords[:, 0], nodeCoords[:, 1], problem.states[:, 0], problem.states[:, 1])

for nodeNum in range(9):
    print(f"Node {nodeNum} displacements: {problem.states[nodeNum]}")
    plt.annotate(f"{nodeNum}", (nodeCoords[nodeNum, 0], nodeCoords[nodeNum, 1]))

plt.show()

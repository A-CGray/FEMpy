import numpy as np
import FEMpy as fp
import matplotlib.pyplot as plt
import niceplots as nice

nice.setRCParams()

E = 71.7e9
rho = 2700
nu = 0.33
constitutiveModel = fp.Constitutive.Iso3D(E, nu, rho)

# ==============================================================================
# Create the FEMpy model by loading in a mesh
# ==============================================================================
options = {"outputDir": "HydrofoilCase", "outputFormat": ".dat", "outputFunctions": ["Von-Mises-Stress"]}
model = fp.FEMpyModel(constitutiveModel, meshFileName="Meshes/Hydrofoil.bdf", options=options)

# --- Define a boundary condition that will be applied in all problems, fixing the top edge of the bracket in x and y ---
nodeCoords = model.getCoordinates()
fixedNodeInds = np.argwhere(nodeCoords[:, 1] < 0.001).flatten()
model.addFixedBCToNodes(name="Fixed", nodeInds=fixedNodeInds, dof=[0, 1, 2], value=0.0)

# ==============================================================================
# Create problem
# ==============================================================================
problem = model.addProblem("TipLoad", options={"printTiming": True})
tipNodeInds = np.argwhere(nodeCoords[:, 1] >= 0.85).flatten()
problem.addLoadToNodes("TipLoad", nodeInds=tipNodeInds, dof=2, value=1e4, totalLoad=True)

# ==============================================================================
# Solve the problem
# ==============================================================================
problem.solve()
problem.writeSolution()

# ==============================================================================
# Plot stiffness matrix sparsity
# ==============================================================================
plt.spy(problem.Jacobian, markersize=0.1, markeredgewidth=0.0)
plt.savefig("HydrofoilCase/StiffnessMatrixSparsity.png", dpi=300)
plt.show()

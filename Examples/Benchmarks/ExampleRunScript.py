"""
==============================================================================
FEMpy example runscript
==============================================================================
@File    :   ExampleRunScript.py
@Date    :   2022/11/21
@Author  :   Alasdair Christison Gray and Jasmin Lim
@Description : This is an example of how I imagine a FEMpy runscript might look
The model used is an L shaped bracket that is fixed at the top and has loads
applied at the right hand edge
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import FEMpy as fp
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


# --- Create constitutive model, 7000 series Aluminium ---
E = 71.7e9
nu = 0.33
rho = 2

# This thickness value is a design variable, by default all elements will use this value, but we can change it later if
# we want
t = 5e-3
constitutiveModel = fp.Constitutive.IsoPlaneStress(E, nu, rho, t)

# ==============================================================================
# Create the FEMpy model by loading in a mesh
# ==============================================================================
options = {"outputDir": "ExampleOutput"}
model = fp.FEMpyModel("../Meshes/LBracket.msh", constitutiveModel, options=options)

# --- Define a boundary condition that will be applied in all problems, fixing the top edge of the bracket in x and y ---
# For now I will just manually find the nodes that are on the top edge of the mesh, maybe we can add better functionality for doing things like this in future
nodeCoords = model.getCoordinates()
topEdgeNodeInds = np.argwhere(nodeCoords[:, 1] == np.max(nodeCoords[:, 1])).flatten()
model.addGlobalFixedBC(name="Fixed", nodeInds=topEdgeNodeInds, dof=[0, 1], value=0.0)

# ==============================================================================
# Setup FEMpy problems
# ==============================================================================
# --- Create 2 different problems for 2 different loading scenarios ---
verticalLoadCase = model.addProblem("Vertical-Load")
horizontalLoadCase = model.addProblem("Horizontal-Load")

# --- Setup the vertical load loadcase ---
# Again I will just manually find the nodes that are on the right edge of the mesh
rightEdgeNodeInds = np.argwhere(nodeCoords[:, 0] == np.max(nodeCoords[:, 0])).flatten()

# In the first case, we will add a total vertical force of 1000N to the right edge of the bracket
verticalLoadCase.addLoadToNodes(nodeInds=rightEdgeNodeInds, dof=1, value=-1e3, totalLoad=True)

# And we will fix the right edge in the x-direction, so it can only move up and down
verticalLoadCase.addFixedBC(name="RightEdgeRoller", nodeInds=rightEdgeNodeInds, dof=0, value=0.0)

# Now add gravity as a body force
rho = 2780
verticalLoadCase.addBodyLoad(name="Gravity", loadingFunction=[0.0, -9.81 * rho])

# --- Add loads to the horizontal loadcase ---
# Add a total horizontal force of 1000N to the right edge of the bracket
horizontalLoadCase.addLoadToNodes(nodeInds=rightEdgeNodeInds, dof=0, value=-1e3, totalLoad=True)

# add gravity again
horizontalLoadCase.addBodyLoad(name="Gravity", loadingFunction=[0.0, -9.81 * rho])

# ==============================================================================
# Solve the problems
# ==============================================================================

for problem in model.problems:
    problem.solve()
    problem.writeSolution(format="plt")

# ==============================================================================
# Change the thickness design variables and solve again
# ==============================================================================
DVs = model.getDVs()
DVs["thickness"] = np.random.rand(len(DVs["thickness"])) * 1e-2 + 1e-3
model.setDVs(DVs)

for problem in model.problems:
    problem.solve()
    problem.writeSolution(format="plt")

# ==============================================================================
# Compute function
# ==============================================================================
for problem in model.problems:
    value = problem.computeFunction("Pressure", elementReductionType=None, globalReductionType=None)
    print(value)

# TODO: Add something related to computing functions of interest
# TODO: How should we set DV bounds?

"""
==============================================================================
FEMpy example runscript
==============================================================================
@File    :   ExampleRunScript.py
@Date    :   2022/11/21
@Author  :   Alasdair Christison Gray
@Description : This is an example of how I imagine a FEMpy runscript might look
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
t = 5e-3
constitutiveModel = fp.Constitutive.isoPlaneStress(E, nu, t)

# --- Create the FEMpy model by loading in a mesh ---
model = fp.FEMpyModel("../Meshes/LBracket.msh", constitutiveModel)

# --- Define a boundary condition that will be applied in all problems, fixing the top edge of the bracket ---
nodeCoords = model.getCoordinates()
fixedBCinds = np.argwhere(nodeCoords[:, 1] == np.max(nodeCoords[:, 1])).flatten()
model.addGlobalBC(name="Fixed", nodeInds=fixedBCinds, dof=0, value=0.0)

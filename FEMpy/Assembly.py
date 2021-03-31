"""
==============================================================================
Matrix and Vector Assembly Procedures
==============================================================================
@File    :   Assembly.py
@Date    :   2021/03/30
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
from scipy import sparse as sp
from numba import jit

# ==============================================================================
# Extension modules
# ==============================================================================


def assembleMatrix(nodeCoords, conn, element, constitutive, knownStates, matType="stiffness"):

    numNodes = np.shape(nodeCoords)[0]
    numEl = np.shape(conn)[0]
    numDisp = element.numDisp

    MatRows = []
    MatColumns = []
    MatEntries = []
    RHSRows = []
    RHSEntries = []

    localMat = np.zeros((element.numDOF, element.numDOF))
    RHSLocal = np.zeros(element.numDOF)
    RHS = np.zeros(element.numDOF * numEl)

    if matType.lower() == "stiffness":
        matFunc = element.getStiffnessMat
    elif matType.lower() == "mass":
        matFunc = element.getMassMat

    for e in range(numEl):
        elNodes = conn[e]
        elDOF = (np.array([numDisp * elNodes]).T + np.arange(numDisp)).flatten()
        elNodeCoords = nodeCoords[elNodes]

        # --- Compute the local  matrix ---
        localMat[:] = matFunc(elNodeCoords, constitutive)

        RHSLocal[:] = 0.0

        # ==============================================================================
        # Apply state BC's
        # ==============================================================================

        # Find the local indices of any DOF in the current element that are restrained
        localConstrainedDOF = np.argwhere(np.isfinite(knownStates[elDOF])).flatten()
        localFreeDOF = np.argwhere(np.isnan(knownStates[elDOF])).flatten()
        nonzeroConstrainedDOF = localConstrainedDOF[
            np.argwhere(knownStates[elDOF[localConstrainedDOF]] != 0.0).flatten()
        ]

        # Add the force contribution of any nonzero fixed displacements
        RHSLocal[localFreeDOF] = -np.sum(
            knownStates[elDOF[nonzeroConstrainedDOF]] * localMat[np.ix_(localFreeDOF, nonzeroConstrainedDOF)], axis=-1
        )

        # For constrained DOF, set RHS equal to the fixed disp value
        RHSLocal[nonzeroConstrainedDOF] = knownStates[elDOF[nonzeroConstrainedDOF]]

        # set all rows and cols of K corresponding to fixed DOF to zero, except diagonal entries which are 1
        localMat[:, localConstrainedDOF] = 0.0
        localMat[localConstrainedDOF, :] = 0.0
        localMat[localConstrainedDOF, localConstrainedDOF] = 1.0

        # ==============================================================================
        # Add local terms to arrays used to create global mat and RHS
        # ==============================================================================
        # For sparse matrix creation we need to create 3 lists which store, the row index, cloumn index and value of every nonzero term in the local stiffness matrix
        nonzeroKInds = localMat.nonzero()

        MatRows += elDOF[nonzeroKInds[0]].tolist()
        MatColumns += elDOF[nonzeroKInds[1]].tolist()
        MatEntries += localMat[nonzeroKInds].tolist()

        RHSRows += elDOF[RHSLocal.nonzero()].tolist()
        RHSEntries += RHSLocal[RHSLocal.nonzero()].tolist()

    # ==============================================================================
    # Create global stiffness mat and RHS
    # ==============================================================================
    Mat = sp.coo_matrix((MatEntries, (MatRows, MatColumns)), shape=(numNodes * numDisp, numNodes * numDisp)).tocsc()
    RHS = sp.coo_matrix((RHSEntries, (np.zeros_like(RHSRows), RHSRows)), shape=(numNodes * numDisp, 1)).tocsc()

    return Mat, RHS


def assembleTractions(nodeCoords, conn, element, constitutive, tractElems, tractEdges, tractFunc, knownStates):
    numNodes = np.shape(nodeCoords)[0]
    numDisp = element.numDisp

    FRows = []
    FEntries = []
    FLocal = np.zeros(element.numDOF)

    for i in range(len(tractElems)):
        e = tractElems[i]
        elNodes = conn[e]
        elDOF = (np.array([numDisp * elNodes]).T + np.arange(numDisp)).flatten()
        elNodeCoords = nodeCoords[elNodes]
        FLocal[:] = element.integrateTraction(tractFunc, elNodeCoords, constitutive, edges=tractEdges[i], n=2).flatten()

        # Add traction force entries that aren't zero and aren't at nodes where displacement is known
        usefulDOF = np.argwhere(np.logical_and(np.isnan(knownStates[elDOF]), FLocal != 0.0)).flatten()
        FRows += elDOF[usefulDOF].tolist()
        FEntries += FLocal[usefulDOF].tolist()

    FTract = sp.coo_matrix((FEntries, (np.zeros_like(FRows), FRows)), shape=(numNodes * numDisp, 1)).tocsc()
    return FTract


def assembleBodyForce(nodeCoords, conn, element, constitutive, forceFunc, knownStates, n=2):
    numNodes = np.shape(nodeCoords)[0]
    numEl = np.shape(conn)[0]
    numDisp = element.numDisp

    FRows = []
    FEntries = []
    FLocal = np.zeros(element.numDOF)
    for e in range(numEl):
        elNodes = conn[e]
        elDOF = (np.array([numDisp * elNodes]).T + np.arange(numDisp)).flatten()
        elNodeCoords = nodeCoords[elNodes]
        FLocal[:] = element.integrateBodyForce(forceFunc, elNodeCoords, constitutive, n=n).flatten()

        # Add traction force entries that aren't zero and aren't at nodes where displacement is known
        usefulDOF = np.argwhere(np.logical_and(np.isnan(knownStates[elDOF]), FLocal != 0.0)).flatten()
        FRows += elDOF[usefulDOF].tolist()
        FEntries += FLocal[usefulDOF].tolist()

    FBody = sp.coo_matrix((FEntries, (FRows, np.zeros_like(FRows))), shape=(numNodes * numDisp, 1)).tocsc()
    return FBody
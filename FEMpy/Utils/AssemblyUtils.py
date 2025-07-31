"""
==============================================================================
FEMpy Assembly utility functions
==============================================================================
@File    :   AssemblyUtils.py
@Date    :   2022/11/23
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
from numba import njit
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


def applyBCsToMatrix(rows, cols, values, bcDOF):
    """Given the COO data for a sparse matrix, modify the data to apply a set of fixed state boundary conditions

    In the original matrix, the ith row of the linear system looks something like:

        ``[A_i1, ..., A_ii, ..., A_in] @ [du_1, ..., du_i, ..., du_n].T = [b_i]``

    To enforce the boundary condition ``u_i = c``, we alter the ith row of the matrix to represent this equation:

        ``[0, ..., 1, ..., 0] @ [du_1, ..., du_i, ..., du_n].T = [u_i - c]``

    This function performs the modification of the matrix data, the modification of the right hand side vector is
    performed by the function :func:`applyBCsToVector`.

    Parameters
    ----------
    rows : array of int
        Row indicies of the matrix entries
    cols : array of int
        Column indicies of the matrix entries
    values : array of float
        Values of the matrix entries
    bcDOF : array of int
        Degrees of freedom to apply boundary conditions to

    Returns
    -------
    rowInds :
    """
    indsToDelete = np.nonzero(np.in1d(rows, bcDOF))[0]
    values[indsToDelete] = 0.0

    rows = np.append(rows, bcDOF)
    cols = np.append(cols, bcDOF)
    values = np.append(values, np.ones(len(bcDOF)))

    return rows, cols, values


def applyBCsToVector(vector, state, bcDOF, bcValues):
    """Given the current state and RHS vectors, modify the data to apply a set of fixed state boundary conditions

    Take a linear system that we are solving to find an update in the state: K @ du = b
    In this system of equation, the ith row looks something like:

        ``[A_i1, ..., A_ii, ..., A_in] @ [du_1, ..., du_i, ..., du_n].T = [b_i]``

    To enforce the boundary condition ``u_i = c``, we alter the ith row of the matrix like this:

        ``[0, ..., 1, ..., 0] @ [du_1, ..., du_i, ..., du_n].T = [c - u_i]``

    This function performs the modification of the RHS vector, the modification of the matrix data is performed by the
    function :func:`applyBCsToMatrix`.

    Parameters
    ----------
    vector : numpy array
        Vector to apply boundary conditions to
    state : numpy array
        The current state vector
    bcDOF : array of int
        Degrees of freedom to apply boundary conditions to
    vector : numpy array
        Vector to apply boundary conditions to
    bcValues : array of float
        Values to fix degrees of freedom at
    """
    vector[bcDOF] = state[bcDOF] - bcValues
    return vector


def convertBCDictToLists(bcDict):
    """Convert a dictionary of boundary conditions to lists of DOF and values

    The boundary condition dictionary stores boundary conditions in this form::

        BCDict = {
            "BC1Name": {"DOF": [0, 1, 2], "Value": [0, 0, 0]},
            "BC2Name": {"DOF": [13, 46, 1385], "Value": [1.0, 1.0, -1.0]},
            "BC3Name": {"DOF": [837, 25], "Value": [1.0, 1.0]},
        }


    Parameters
    ----------
    bcDict : dict
        Dictionary of boundary conditions

    Returns
    -------
    bcDOF : array of int
        Degrees of freedom to apply boundary conditions to
    bcValues : array of float
        Values to fix degrees of freedom at

    Examples
    --------
    >>> BCDict = {
    ...     "BC1Name": {"DOF": [0, 1, 2], "Value": [0, 0, 0]},
    ...     "BC2Name": {"DOF": [13, 46, 1385], "Value": [1.0, 1.0, -1.0]},
    ...     "BC3Name": {"DOF": [837, 25], "Value": [1.0, 1.0]},
    ... }
    ... convertBCDictToLists(BCDict)
    ([0, 1, 2, 13, 46, 1385, 837, 25], [0, 0, 0, 1.0, 1.0, -1.0, 1.0, 1.0])

    """
    bcDOF = []
    bcValues = []
    for key in bcDict:
        bcDOF += bcDict[key]["DOF"]
        bcValues += bcDict[key]["Value"]
    return bcDOF, bcValues


def convertLoadsDictToVector(loadsDict, numDOF: int):
    """Convert a dictionary of loads to a global load vector

    The loads dictionary stores loads in this form::

        loadsDict = {
            "Load1Name": {"DOF": [0, 1, 2], "Value": [0, 0, 0]},
            "Load2Name": {"DOF": [13, 46, 1385], "Value": [1.0, 1.0, -1.0]},
            "Load3Name": {"DOF": [837, 25], "Value": [1.0, 1.0]},
        }


    Parameters
    ----------
    loadsDict : dict
        Dictionary of loads

    Returns
    -------
    loads : array of float
        global load vector, with loads applied to the correct DOF
    """
    loadForce = np.zeros(numDOF)
    for key in loadsDict:
        loadForce[loadsDict[key]["DOF"]] += loadsDict[key]["Value"]
    return loadForce


@njit(cache=True)
def localMatricesToCOOArrays(localMats, localDOF):
    """Convert a set of local matrices for a set of elements to COO format data for a global matrix

    Parameters
    ----------
    localMats : numElements x n x n array
        Local matrices for each element
    localDOF : numElement x n array
        Global DOF indices for each element

    Returns
    -------
    rows : list of int
        Row indicies of the global matrix entries
    cols : list of int
        Column indicies of the global matrix entries
    values : list of float
        Values of the global matrix entries

    """
    numElements = localMats.shape[0]
    dofPerElement = localDOF.shape[1]
    rows = np.zeros(numElements * dofPerElement**2, dtype=np.int64)
    cols = np.zeros(numElements * dofPerElement**2, dtype=np.int64)
    values = np.zeros(numElements * dofPerElement**2)

    for ii in range(numElements):
        # Figure out where in the COO arrays to write the entires from this element
        startInd = ii * dofPerElement**2

        for rowInd in range(dofPerElement):
            rowStartInd = startInd + rowInd * dofPerElement
            for colInd in range(dofPerElement):
                flatInd = rowStartInd + colInd
                rows[flatInd] = localDOF[ii, rowInd]
                cols[flatInd] = localDOF[ii, colInd]
                values[flatInd] = localMats[ii, rowInd, colInd]

    # Now we have all the data, return only the non-zero entries
    nonzeroEntries = np.flatnonzero(values)

    rows = rows[nonzeroEntries]
    cols = cols[nonzeroEntries]
    values = values[nonzeroEntries]

    return rows, cols, values


@njit(cache=True)
def scatterLocalResiduals(localResiduals, connectivity, globalResidual):
    """Scatter local element residuals into the global residual vector

    Parameters
    ----------
    localResiduals : numElements x numNodes x numStates array
        Element local residuals
    connectivity : numElements x numNodes array
        Element connectivity matrix, each row contains the nodeIDs for an element
    globalResidual : numpy array
        The global residual vector
    """
    numStates = localResiduals.shape[2]
    numElements = localResiduals.shape[0]
    numNodes = localResiduals.shape[1]

    for elNum in range(numElements):
        for localNodeID in range(numNodes):
            nodeID = connectivity[elNum, localNodeID]
            for stateNum in range(numStates):
                globalResidual[nodeID * numStates + stateNum] += localResiduals[elNum, localNodeID, stateNum]


if __name__ == "__main__":
    rows = np.array([0, 0, 1, 2], dtype=np.int64)
    cols = np.array([1, 2, 2, 0], dtype=np.int64)
    values = np.array([1, 0.5, 2, 3])
    bcDOF = np.array([1, 2], dtype=np.int64)
    bcValues = np.array([10, 15])
    rows, cols, values = applyBCsToMatrix(rows, cols, values, bcDOF)
    print(rows, cols, values)

    BCDict = {
        "BC1Name": {"DOF": [0, 1, 2], "Value": [0, 0, 0]},
        "BC2Name": {"DOF": [13, 46, 1385], "Value": [1.0, 1.0, -1.0]},
        "BC3Name": {"DOF": [837, 25], "Value": [1.0, 1.0]},
    }
    bcDOF, bcValues = convertBCDictToLists(BCDict)

    print(bcDOF, bcValues)

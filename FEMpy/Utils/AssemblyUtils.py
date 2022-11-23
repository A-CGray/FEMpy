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


def applyBCsToMatrix(self, rows, cols, values, bcDOF, bcValues):
    """Given the COO data for a sparse matrix, modify the data to apply a set of fixed state boundary conditions

    In the original matrix, the ith row of the linear system looks something like:

        ``[A_i1, ..., A_ii, ..., A_in] @ [u_1, ..., u_i, ..., u_n].T = [b_i]``

    To enforce the boundary condition ``u_i = c``, we alter the ith row of the matrix to represent this equation:

        ``[0, ..., 1, ..., 0] @ [u_1, ..., u_i, ..., u_n].T = [c]``

    This function modifies the supplied row, column and value arrays in-place, so returns nothing

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
    bcValues : array of float
        Values to fix degrees of freedom at
    """
    # TODO: Implement this
    return None


def applyBCsToVector(self, vector, state, bcDOF, bcValues):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    rows : _type_
        _description_
    values : _type_
        _description_
    bcDOF : _type_
        _description_
    bcValues : _type_
        _description_
    """
    return None


def convertBCDictToLists(bcDict, numStates):
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
    ...        "BC1Name": {"DOF": [0, 1, 2], "Value": [0, 0, 0]},
    ...        "BC2Name": {"DOF": [13, 46, 1385], "Value": [1.0, 1.0, -1.0]},
    ...        "BC3Name": {"DOF": [837, 25], "Value": [1.0, 1.0]},
    ...    }
    ... convertBCDictToLists(BCDict)
    ([0, 1, 2, 13, 46, 1385, 837, 25], [0, 0, 0, 1.0, 1.0, -1.0, 1.0, 1.0])

    """
    bcDOF = []
    bcValues = []
    # TODO: Implement this
    return bcDOF, bcValues


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
    rows : array of int
        Row indicies of the global matrix entries
    cols : array of int
        Column indicies of the global matrix entries
    values : array of float
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
        endInd = (ii + 1) * dofPerElement**2

        # Now we need to flatten the local matrix and write it to the COO arrays
        rowInds = np.repeat(localDOF[ii, :], dofPerElement)  # [0,0,0,..., 1,1,1,..., 2,2,2,...]
        colInds = np.tile(localDOF[ii, :], dofPerElement)  # [0,1,2,..., 0,1,2,..., 0,1,2,...]
        rows[startInd:endInd] = rowInds
        cols[startInd:endInd] = colInds
        values[startInd:endInd] = localMats[ii].flatten()

    # Now we have all the data, return only the non-zero entries
    nonzeroEntries = np.flatnonzero(values)

    rows = rows[nonzeroEntries]
    cols = cols[nonzeroEntries]
    values = values[nonzeroEntries]

    return rows, cols, values

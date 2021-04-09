"""
==============================================================================
Smoothing
==============================================================================
@File    :   Smoothing.py
@Date    :   2021/04/08
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
from scipy.sparse.linalg import factorized
from scipy.sparse import csc_matrix

# ==============================================================================
# Extension modules
# ==============================================================================


def getSmoother(intPointParamCoords, element, conn):
    """Generate a function which converts integration point values to nodal values using global smoothing

    [extended_summary]

    Parameters
    ----------
    intPointParamCoords : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    numIntPoints = np.shape(intPointParamCoords)[0]
    numNode = np.max(conn) + 1
    numEl = np.shape(conn)[0]
    # Compute the shape function values at the integration points
    N = element.getShapeFunctions(intPointParamCoords)

    entries = []
    rows = []
    columns = []

    for e in range(numEl):
        elNodes = conn[e].tolist()
        for ni in range(numIntPoints):
            entries += N[ni].tolist()
            rows += numIntPoints * [numIntPoints * e + ni]
            columns += elNodes

    # --- Somewhat confusingly, actually compute the transpose of M as it saves some transpose operations later ---
    MT = csc_matrix((entries, (columns, rows)), shape=(numNode, numIntPoints * numEl))

    # --- Create the matrix for the least squares solution and then factorise it ---
    MTM = MT @ MT.T
    smoothingSolver = factorized(MTM)

    def smoother(x):
        return smoothingSolver(MT @ x)

    return smoother, MTM, MT

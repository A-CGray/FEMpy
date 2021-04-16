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

    Given integration point values f_int, the smoother solves an over/underdetermined system of equations to find the
    nodal values which would best recover f_int through interpolation:

    M * f_node = f_int

    The least squares solution is:

    f_node = (M^T * M)^-1 * M^T * F_int

    Parameters
    ----------
    intPointParamCoords : [type]
        [description]

    Returns
    -------
    smoother : function
        A function which performs integration point to nodal value smoothing using a pre-factorized least-squares solution (a.k.a should be very very fast)
    MTM : Scipy CSC sparse matrix
        The matrix M^T * M which is used to compute the least squares solution, this output is not required
    MT : Scipy CSC sparse matrix
        The matrix M^T which is used to compute the least squares solution, this output is not required
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

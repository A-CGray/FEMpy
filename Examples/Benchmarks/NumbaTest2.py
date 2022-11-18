"""
==============================================================================
Numba nested function call test
==============================================================================
@File    :   NumbaTest.py
@Date    :   2022/11/16
@Author  :   Alasdair Christison Gray
@Description : Seeing if numba functions can handle being passed another numba function as an input
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
from numba import guvectorize, float64, prange
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


@guvectorize(
    [(float64[:, :, ::1], float64[:, :, ::1], float64[:, :, :, ::1])],
    "(p,d,n),(e,n,d)->(e,p,d,d)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
    target="parallel",
)
def computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jac):
    """This function computes a nasty product of two multidimensional array required when computing element mapping Jacobians

    Given the shape function derivatives at each point NPrimeParam (a numPoints x numDim x numNodes array), and the node
    coordinates at each element (a numElements x numNodes x numDim array), we want to compute:
    NPrimeParam x nodeCoords[ii]
    For each element, this function does this, but is jit'ed by numba to make it fast.

    Parameters
    ----------
    NPrimeParam : _type_
        _description_
    nodeCoords : _type_
        _description_
    """
    numElements = Jac.shape[0]
    numPoints = Jac.shape[1]
    for ii in prange(numElements):
        for jj in range(numPoints):
            Jac[ii, jj] = NPrimeParam[jj] @ nodeCoords[ii]


def computeNPrimeCoordProductNonjit(NPrimeParam, nodeCoords, Jac):
    """This function computes a nasty product of two multidimensional array required when computing element mapping Jacobians

    Given the shape function derivatives at each point NPrimeParam (a numPoints x numDim x numNodes array), and the node
    coordinates at each element (a numElements x numNodes x numDim array), we want to compute:
    NPrimeParam x nodeCoords[ii]
    For each element, this function does this, but is jit'ed by numba to make it fast.

    Parameters
    ----------
    NPrimeParam : _type_
        _description_
    nodeCoords : _type_
        _description_
    """
    numElements = Jac.shape[0]
    for ii in range(numElements):
        Jac[ii] = NPrimeParam @ nodeCoords[ii]


if __name__ == "__main__":
    import time

    numEl = 10
    numPoints = 8
    numNode = 4
    numDim = 3

    NPrimeParam = np.random.rand(numPoints, numDim, numNode)
    nodeCoords = np.random.rand(2 * numEl, numNode, numDim)[::2]
    Jac = np.zeros((numEl, numPoints, numDim, numDim))
    print(f"Is nodeCoords contiguous: {nodeCoords.flags['C_CONTIGUOUS']}")
    print(f"Is NPrimeParam contiguous: {NPrimeParam.flags['C_CONTIGUOUS']}")
    print(f"Is Jac contiguous: {Jac.flags['C_CONTIGUOUS']}")

    # Call once with small input to make it compile
    computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jac)

    # Now test the speed with a bigger array
    numEl = 10000000

    NPrimeParam = np.random.rand(numPoints, numDim, numNode)
    nodeCoords = np.random.rand(numEl, numNode, numDim)
    Jac = np.zeros((numEl, numPoints, numDim, numDim))

    start = time.time()
    nodeCoords = np.ascontiguousarray(nodeCoords)
    end = time.time()
    print(f"Time taken to make nodeCoords contiguous: {(end - start):e} s")

    start = time.time()
    computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jac)
    end = time.time()
    print(f"Time taken with Numba: {(end - start):e} s")

    start = time.time()
    computeNPrimeCoordProductNonjit(NPrimeParam, nodeCoords, Jac)
    end = time.time()
    print(f"Time taken with numpy: {(end - start):e} s")

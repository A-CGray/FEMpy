import numpy as np
from numba import guvectorize, float64
import time


def nastyProduct(dUPrimedq, weakRes, result):
    """Compute a nasty product of high dimensional arrays to compute integration point residuals

    _extended_summary_

    Parameters
    ----------
    dUPrimedq : numElements x numPoints x numDim x numNodes array
        Sensitivity of the state gradients to nodal DOFs
    weakRes : numElements x numPoints x numStates x numDim array
        Weak residual values
    result : numElements x numPoints x numNodes x numStates array
        Integration point residual contributions
    """

    result[:] = np.einsum("epdn,epds->epns", dUPrimedq, weakRes)

    # numElements = dUPrimedq.shape[0]
    # numPoints = dUPrimedq.shape[1]

    # for ii in range(numElements):
    #     for jj in range(numPoints):
    #         result[ii, jj] = dUPrimedq[ii, jj].T @ weakRes[ii, jj]


@guvectorize(
    [(float64[:, :, :, ::1], float64[:, :, :, ::1], float64[:, :, :, ::1])],
    "(e,p,d,n),(e,p,d,s)->(e,p,n,s)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def nastyProductVectorize(dUPrimedq, weakRes, result):
    """Compute a nasty product of high dimensional arrays to compute integration point residuals

    _extended_summary_

    Parameters
    ----------
    dUPrimedq : numElements x numPoints x numDim x numNodes array
        Sensitivity of the state gradients to nodal DOFs
    weakRes : numElements x numPoints x numStates x numDim array
        Weak residual values
    result : numElements x numPoints x numNodes x numStates array
        Integration point residual contributions
    """
    numElements = dUPrimedq.shape[0]
    numPoints = dUPrimedq.shape[1]

    for ii in range(numElements):
        for jj in range(numPoints):
            result[ii, jj] = dUPrimedq[ii, jj].T @ weakRes[ii, jj]


numElements = 100000
numPoints = 17
numNodes = 16
numDim = 3
numStates = 4
numStrains = 6

np.random.seed(1)
dUPrimedq = np.random.rand(numElements, numPoints, numDim, numNodes)
weakRes = np.random.rand(numElements, numPoints, numDim, numStates)
result = np.zeros((numElements, numPoints, numNodes, numStates))
result1 = np.zeros((numElements, numPoints, numNodes, numStates))
result2 = np.zeros((numElements, numPoints, numNodes, numStates))


start = time.time()
nastyProduct(dUPrimedq, weakRes, result)
end = time.time()

print(f"Time for python: {(end - start):e} s")


nastyProductVectorize(dUPrimedq, weakRes, result1)
start = time.time()
nastyProductVectorize(dUPrimedq, weakRes, result2)
end = time.time()
print(f"Time for Numba: {(end - start):e} s")

print(f"maxDiff between results = {np.max(np.abs(result - result1))}")

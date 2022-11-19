import numpy as np
from numba import guvectorize, float64
import time


def nastyProduct(JacInv, NPrimeParam, nodeStates, result):
    numElements = JacInv.shape[0]
    numPoints = JacInv.shape[1]
    for ii in range(numElements):
        for jj in range(numPoints):
            # result[ii, jj] = np.linalg.multi_dot([JacInv[ii, jj], NPrimeParam[jj], nodeStates[ii]]).T
            result[ii, jj] = np.einsum("df,fn,ns->sd", JacInv[ii, jj], NPrimeParam[jj], nodeStates[ii])


@guvectorize(
    [(float64[:, :, :, ::1], float64[:, :, ::1], float64[:, :, ::1], float64[:, :, :, ::1])],
    "(e,p,d,d),(p,d,n),(e,n,s)->(e,p,s,d)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def nastyProductJIT(JacInv, NPrimeParam, nodeStates, result):
    numElements = JacInv.shape[0]
    numPoints = JacInv.shape[1]
    for ii in range(numElements):
        for jj in range(numPoints):
            result[ii, jj] = (JacInv[ii, jj] @ NPrimeParam[jj] @ nodeStates[ii]).T


numElements = 100000
numPoints = 6
numNodes = 4
numDim = 3
numStates = 2


NPrimeParam = np.random.rand(numPoints, numDim, numNodes)
nodeStates = np.random.rand(numElements, numNodes, numStates)
Jac = np.random.rand(numElements, numPoints, numDim, numDim)
result1 = np.zeros((numElements, numPoints, numStates, numDim))
result2 = np.zeros((numElements, numPoints, numStates, numDim))

start = time.time()
nastyProduct(Jac, NPrimeParam, nodeStates, result1)
end = time.time()

print(f"Time for python: {(end - start):e} s")


nastyProductJIT(Jac, NPrimeParam, nodeStates, result1)
start = time.time()
nastyProductJIT(Jac, NPrimeParam, nodeStates, result2)
end = time.time()
print(f"Time for Numba: {(end - start):e} s")

print(f"maxDiff between results = {np.max(np.abs(result1 - result2))}")

import numpy as np
from numba import guvectorize, float64
import time


def nastyProduct(dUPrimedq, weakResJac, result):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    dUPrimedq : numPoints x numDim x numNodes array
        Sensitivity of the state gradients to nodal DOFs
    weakResJac : numPoints x numDim x numStates x numStates x numDim array
        _description_
    result : _type_
        _description_
    """
    np.einsum(
        "pdn,pdsSD,pDN->pnsNS", dUPrimedq, weakResJac, dUPrimedq, optimize=["einsum_path", (0, 1), (0, 1)], out=result
    )


# @guvectorize(
#     [(float64[:, :, ::1], float64[:, :, :, :, ::1], float64[:, :, ::1], float64[:, :, :, :, ::1])],
#     "(p,d,n),(p,d,s,s,d),(p,d,n)->(p,n,s,s,n)",
#     nopython=True,
#     cache=True,
#     fastmath=True,
#     boundscheck=False,
# )
# def nastyProductVectorize(a, weakRes, b, result):
#     numPoints = a.shape[0]

#     for ii in range(numPoints):
#         result[ii] = a[ii].T @ weakRes[ii] @ b[ii]


numElements = 100000
numPoints = 170000
numNodes = 16
numDim = 3
numStates = 4
numStrains = 6

np.random.seed(1)
dUPrimedq = np.random.rand(numPoints, numDim, numNodes)
weakResJac = np.random.rand(numPoints, numDim, numStates, numStates, numDim)
result = np.zeros((numPoints, numNodes, numStates, numStates, numNodes))
result1 = np.zeros((numPoints, numNodes, numStates, numStates, numNodes))
# result1 = np.zeros((numElements, numPoints, numNodes, numStates))
# result2 = np.zeros((numElements, numPoints, numNodes, numStates))


start = time.time()
nastyProduct(dUPrimedq, weakResJac, result)
end = time.time()

print(f"Time for python: {(end - start):e} s")


# nastyProductVectorize(dUPrimedq, weakResJac, dUPrimedq, result1)
# start = time.time()
# nastyProductVectorize(dUPrimedq, weakResJac, dUPrimedq, result1)
# end = time.time()
# print(f"Time for Numba: {(end - start):e} s")

# print(f"maxDiff between results = {np.max(np.abs(result - result1))}")

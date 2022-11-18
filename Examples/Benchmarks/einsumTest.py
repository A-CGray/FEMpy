import opt_einsum as oe
import numpy as np

numElements = 100000
numPoints = 6
numNodes = 4
numDim = 3
numStates = 2


NPrimeParam = np.random.rand(numPoints, numDim, numNodes)
nodeStates = np.random.rand(numElements, numNodes, numStates)
Jac = np.random.rand(numElements, numPoints, numDim, numDim)

einsumResult = np.einsum("epdf,pdn,ens->epsf", Jac, NPrimeParam, nodeStates)
forLoopResult = np.zeros((numElements, numPoints, numStates, numDim))
for ii in range(numElements):
    for jj in range(numPoints):
        forLoopResult[ii, jj] = Jac[ii, jj] @ NPrimeParam[jj] @ nodeStates[ii]

print(f"Max diff between einsum and for loop result is: {(np.max(np.abs(einsumResult-forLoopResult))):e}")

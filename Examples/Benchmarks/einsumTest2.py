import numpy as np
from numba import guvectorize, float64, njit, prange
import time


def nastyProduct(dStraindUPrime, stress, volScaling, result):
    """Compute a nasty product of high dimensional arrays to compute the weak residual

    _extended_summary_

    Parameters
    ----------
    dStraindUPrime : numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
    stress : numPoints x numStrains array
        Stresses at each point
    volScaling : numPoints array
        Volume scaling at each point
    result : numPoints x numDim x numStates array
        _description_
    """
    numPoints = dStraindUPrime.shape[0]

    for ii in range(numPoints):
        result[ii] += dStraindUPrime[ii].T @ stress[ii] * volScaling[ii]


@njit(cache=True, fastmath=True)
def nastyProductJIT(dStraindUPrime, stress, volScaling, result):
    """Compute a nasty product of high dimensional arrays to compute the weak residual

    _extended_summary_

    Parameters
    ----------
    dStraindUPrime : numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
    stress : numPoints x numStrains array
        Stresses at each point
    volScaling : numPoints array
        Volume scaling at each point
    result : numPoints x numDim x numStates array
        _description_
    """
    numPoints = dStraindUPrime.shape[0]
    numStrains = dStraindUPrime.shape[1]

    for ii in range(numPoints):
        for jj in range(numStrains):
            result[ii] += dStraindUPrime[ii, jj].T * stress[ii, jj] * volScaling[ii]


@guvectorize(
    [(float64[:, :, :, ::1], float64[:, ::1], float64[::1], float64[:, :, ::1])],
    "(p,e,s,d),(p,e),(p)->(p,d,s)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def nastyProductVectorize(dStraindUPrime, stress, volScaling, result):
    """Compute a nasty product of high dimensional arrays to compute the weak residual

    _extended_summary_

    Parameters
    ----------
    dStraindUPrime : numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
    stress : numPoints x numStrains array
        Stresses at each point
    volScaling : numPoints array
        Volume scaling at each point
    result : numPoints x numDim x numStates array
        _description_
    """
    numPoints = dStraindUPrime.shape[0]
    numStrains = dStraindUPrime.shape[1]

    for ii in prange(numPoints):
        for jj in range(numStrains):
            result[ii] += dStraindUPrime[ii, jj].T * stress[ii, jj] * volScaling[ii]


numPoints = 1000000
numNodes = 4
numDim = 3
numStates = 1
numStrains = 3

np.random.seed(1)
dStraindUPrime = np.random.rand(numPoints, numStrains, numStates, numDim)
stress = np.random.rand(numPoints, numStrains)
volScaling = np.random.rand(numPoints)
result = np.zeros((numPoints, numDim, numStates))
result1 = np.zeros((numPoints, numDim, numStates))
result2 = np.zeros((numPoints, numDim, numStates))


start = time.time()
nastyProduct(dStraindUPrime, stress, volScaling, result)
end = time.time()

print(f"Time for python: {(end - start):e} s")


nastyProductVectorize(dStraindUPrime, stress, volScaling, result1)
start = time.time()
nastyProductVectorize(dStraindUPrime, stress, volScaling, result2)
end = time.time()
print(f"Time for Numba: {(end - start):e} s")

print(f"maxDiff between results = {np.max(np.abs(result - result1))}")

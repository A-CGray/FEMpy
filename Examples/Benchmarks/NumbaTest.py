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
from numba import njit, prange
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


@njit(cache=True, fastmath=True)
def innerFunc(x, a):
    for ii in range(1, len(x)):
        x[ii] = a * np.tanh(np.sin(np.sqrt(x[ii - 1])) + np.cos(np.sqrt(x[ii]))) ** 2


@njit(cache=True, fastmath=True, parallel=True)
def outerFuncParallel(mat, innerFunc):
    n = mat.shape[0]
    for ii in prange(n):
        innerFunc(mat[ii])
    return np.sum(mat, axis=1)


@njit(cache=True, fastmath=True)
def outerFuncSerial(mat, innerFunc):
    n = mat.shape[0]
    for ii in range(n):
        innerFunc(mat[ii])
    return np.sum(mat, axis=1)


def genInnerFunc(a):
    @njit(cache=True, fastmath=True)
    def f(x):
        innerFunc(x, a)

    return f


class innerFuncClass:
    def __init__(self, a):
        self.a = a
        self.innerFunc = genInnerFunc(a)

    def getInnerFunc(self):
        return genInnerFunc(self.a)

    @staticmethod
    def genInnerFunc(a):
        @njit(cache=True, fastmath=True)
        def f(x):
            innerFunc(x, a)

        return f


if __name__ == "__main__":
    import time

    c = innerFuncClass(0.5)
    classInnerFunc = c.getInnerFunc()
    classInnerFunc2 = c.innerFunc

    @njit(cache=True, fastmath=True)
    def newInnerFunc(x):
        innerFunc(x, 0.5)

    mat = np.ones((5, 5))
    outerFuncSerial(mat, classInnerFunc2)
    outerFuncParallel(mat, classInnerFunc2)

    mat = np.random.rand(10000, 10000)
    start = time.time()
    outerFuncSerial(mat, classInnerFunc2)
    end = time.time()
    print(f"Big function call took {(end - start):e} seconds in serial")

    start = time.time()
    outerFuncParallel(mat, classInnerFunc2)
    end = time.time()
    print(f"Big function call took {(end - start):e} seconds in parallel")
    # outerFunc.parallel_diagnostics(level=4)

"""
==============================================================================
Linear Algebra speed test
==============================================================================
@File    :   detTest.py
@Date    :   2021/11/14
@Author  :   Alasdair Christison Gray
@Description : Testing speed of computing determinants and inverses of 2x2 and 3x3 matrices with numpy's generic det function vs
manual implementations for 2x2 nd 3x3 matrices
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import timeit

# ==============================================================================
# Extension modules
# ==============================================================================

numMats = 80000
setupString = f"""
import numpy as np
from FEMpy.LinAlg import det2, det3, inv2, inv3

A2 = np.random.rand({numMats}, 2, 2)
A3 = np.random.rand({numMats}, 3, 3)
det2(A2)
det3(A3)
inv2(A2)
inv3(A3)
"""
N = 50


def benchmark_LinAlg():
    for func in ["det", "inv"]:
        for size in [2, 3]:
            print("\n")
            for method in ["numpy", "numba"]:
                if method == "numpy":
                    test = f"np.linalg.{func}(A{size})"
                else:
                    test = f"{func}{size}(A{size})"
                t = timeit.timeit(test, setup=setupString, number=N) / N
                print(f"Time to compute {numMats} {size} x {size} {func}s with {method} = {t:7.11e} s")

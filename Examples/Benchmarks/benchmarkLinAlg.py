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
import unittest
import parameterized
import time

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from FEMpy.LinAlg import det2, det3, inv2, inv3

# ==============================================================================
# Extension modules
# ==============================================================================

npFuncMap = {"det": np.linalg.det, "inv": np.linalg.inv}
numbaFuncMap = {"det": {2: det2, 3: det3}, "inv": {2: inv2, 3: inv3}}

test_params = []
for func in ["det", "inv"]:
    for size in [2, 3]:
        for method in ["numpy", "numba"]:
            test_params.append({"func": func, "size": size, "method": method})


def nameFunc(testcase_func, param_num, params):
    return f"BenchmarkLinAlg-{params['method']}-{params['func']}-{params['size']}"


@parameterized.parameterized_class(test_params, class_name_func=nameFunc)
class BenchmarkLinalg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Call numba methods with a dummy array to compile them"""
        A2 = np.random.rand(10, 2, 2)
        A3 = np.random.rand(10, 3, 3)

        det2(A2)
        det3(A3)
        inv2(A2)
        inv3(A3)

    def setUp(self):
        # For some unknown reason, when running benchmarks, the non parameterized version of this class gets called,
        # with none of the test parameters defined, in this case we just skip everything as we don't want to use this
        # version of the class to test anything
        if self.__class__.__name__ == "BenchmarkLinalg":
            self.skip = True
            return
        else:
            self.skip = False
        self.numMats = int(1e5)
        self.mats = np.random.rand(self.numMats, self.size, self.size)
        self.N = 50

        if self.method == "numpy":
            self.testFunc = npFuncMap[self.func]
        else:
            self.testFunc = numbaFuncMap[self.func][self.size]

    def benchmark_LinAlg(self):
        if not self.skip:
            start = time.time()
            for _ in range(self.N):
                self.testFunc(self.mats)
            end = time.time()
            runTime = (end - start) / self.N
            print(
                f"\nTime to compute {self.numMats} {self.size} x {self.size} {self.func}s with {self.method} = {runTime:7.11e} s"
            )


if __name__ == "__main__":
    unittest.main()

"""
==============================================================================
Numerical Integration speed test
==============================================================================
@File    :   IntegrationSpeedTest.py
@Date    :   2021/09/24
@Author  :   Alasdair Christison Gray
@Description : Testing the speed of numerical integration with numpy, numba and cython
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from numba import njit
import timeit

# ==============================================================================
# Extension modules
# ==============================================================================

version = ["numpy", "numba", "cython", "optimised cython"]
names = ["numpy_integrate", "numba_integrate", "cython_integrate", "cython_opt_integrate"]

for n in range(2, 20, 2):
    print(f"\n{n} Points:")
    print("--------------")
    for i in range(len(names)):
        setup = f"from {names[i]} import integrate\n"
        setup += "import numpy as np\n"
        setup += f"f = np.random.rand({n})\n"
        setup += f"weights = np.random.rand({n})\n"
        time = timeit.timeit(setup=setup, stmt="integrate(weights, f)", number=1)
        print(f"{version[i]} code time = {time:.04e} s")

"""
==============================================================================
FEMpy Utilities
==============================================================================
@File    :   Utils.py
@Date    :   2021/04/16
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from numba import njit

# ==============================================================================
# Extension modules
# ==============================================================================


@njit(cache=True)
def ksAgg(g, rho=100.0):
    """Compute a smooth approximation to the maximum of a set of values us KS aggregation

    Parameters
    ----------
    g : 1d array
        Values to approximate the maximum of
    rho : float, optional
        KS Weight parameter, larger values give a closer but less smooth approximation of the maximum, by default 100.0

    Returns
    -------
    float
        The KS agregated value
    """
    ng = len(g)
    maxg = np.max(g)
    return maxg + 1.0 / rho * np.log(1.0 / ng * np.sum(np.exp(rho * (g - maxg))))

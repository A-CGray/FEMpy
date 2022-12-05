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
def ksAgg(g, ksType, rho=100.0):
    """Compute a smooth approximation to the maximum of a set of values us KS agregation



    Parameters
    ----------
    g : 1d array
        Values to perform KS aggregation
    ksType : string 
        determines which type of KS aggregation is performed ["min", "max"]
    rho : float, optional
        KS Weight parameter, larger values give a closer but less smooth approximation of the maximum, by default 100.0

    Returns
    -------
    float
        The KS agregated value
    """
    assert ksType.lower() in ["min", "max"], "KS aggregation type not valid"
    
    ng = len(g)
    if ksType.lower() == "min":
        g = -1*g
        
    maxg = np.max(g)
    ks = maxg + 1.0 / rho * np.log(1.0 / ng * np.sum(np.exp(rho * (g - maxg))))

    if ksType.lower() == "max":
        return ks
        
    if ksType.lower() == "min":
        return -1*ks
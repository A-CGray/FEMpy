"""
==============================================================================
Triangular quadrature rules
==============================================================================
@File    :   TriQuad.py
@Date    :   2022/12/04
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
from functools import lru_cache

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


@lru_cache(maxsize=None)
def getTriQuadPoints(order: int) -> np.ndarray:
    """Get integration points for a triangular element

    Only 1st 2nd and 3rd order quadrature are supported

    Parameters
    ----------
    order : int
        Order of the quadrature rule

    Returns
    -------
    numIntPoints x numDim array
        Gauss quadrature point parametric coordinates

    Raises
    ------
    ValueError
        If requested order is not supported
    """
    if order == 1:
        return np.array([[1 / 3, 1 / 3]])
    elif order == 2:
        return np.array([[0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    elif order == 3:
        return np.array([[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]])
    else:
        raise ValueError(f"Triangular quadrature of order {order} not supported")


@lru_cache(maxsize=None)
def getTriQuadWeights(order: int) -> np.ndarray:
    """Get integration point weightss for a triangular element

    Only 1st 2nd and 3rd order quadrature are supported

    Parameters
    ----------
    order : int
        Order of the quadrature rule

    Returns
    -------
    array of numIntPoints length
        Gauss quadrature point weights

    Raises
    ------
    ValueError
        If requested order is not supported
    """
    if order == 1:
        w = np.array([1.0])
    elif order == 2:
        w = np.ones(3) / 3
    elif order == 3:
        w = np.array([-27 / 28, 25 / 48, 25 / 48, 25 / 48])
    else:
        raise ValueError(f"Triangular quadrature of order {order} not supported")
    return 0.5 * w

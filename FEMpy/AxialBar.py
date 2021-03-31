"""
==============================================================================
Axial Bar Constitutive Class
==============================================================================
@File    :   AxialBar.py
@Date    :   2021/03/30
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

# ==============================================================================
# Extension modules
# ==============================================================================
from .Constitutive import Constitutive
from .LagrangePoly import LagrangePoly1d, LagrangePoly1dDeriv


class AxialBar(Constitutive):
    def __init__(self, E, A, rho):
        super().__init__(numStrain=1, numDisp=1, numStress=1)
        self.DMat = np.array([[E]])
        self.LMats[0] = np.array([[1]])
        self.E = E
        self.A = A
        self.rho = rho

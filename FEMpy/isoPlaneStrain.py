"""
==============================================================================
Isotropic Plane Strain Constitutive Class
==============================================================================
@File    :   isoPlaneStress.py
@Date    :   2021/03/19
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


class isoPlaneStrain(Constitutive):
    def __init__(self, E, nu, t) -> None:
        super().__init__(numStrain=3, numDisp=2, numStress=3)
        self.DMat = (
            E
            / ((1.0 - 2.0 * nu) * (1.0 + nu))
            * np.array([[1.0 - nu, nu, 0.0], [nu, 1.0 - nu, 0.0], [0.0, 0.0, 0.5 * (1.0 - 2.0 * nu)]])
        )
        self.LMats[0] = np.array([[1, 0], [0, 0], [0, 1]])
        self.LMats[1] = np.array([[0, 0], [0, 1], [1, 0]])
        self.E = E
        self.nu = nu
        self.t = t
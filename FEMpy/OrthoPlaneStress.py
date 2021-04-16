"""
==============================================================================
Orthotropic Plane Stress Constitutive Class
==============================================================================
@File    :   isoPlaneStress.py
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
import pyComposite as pc

# ==============================================================================
# Extension modules
# ==============================================================================
from .Constitutive import Constitutive


class orthoPlaneStress(Constitutive):
    def __init__(self, E1, E2, nu12, G12, t, theta) -> None:
        """Initialise an orthotropic plane stress constitutive object

        This class models a thin, orthotropic material, e.g a single composite lamina

        Parameters
        ----------
        E1 : float
            Longitudinal modulus
        E2 : float
            Transverse Modulus
        nu12 : float
            Major Poisson's ratio
        G12 : float
            Shear modulus
        t : float
            Thickness
        theta : float
            Material orientation, in radians, theta = 0 means the material 1 axis is aligned with x
        """
        super().__init__(numStrain=3, numDisp=2, numStress=3)
        self.LMats[0] = np.array([[1, 0], [0, 0], [0, 1]])
        self.LMats[1] = np.array([[0, 0], [0, 1], [1, 0]])
        self.matProps = {
            "E1": E1,
            "E2": E2,
            "G12": G12,
            "v12": nu12,
        }
        self.t = t
        self.theta = theta
        self.lamina = pc.lamina(self.matProps)

    @property
    def DMat(self):
        return self.lamina.getQBar(self.theta)


if __name__ == "__main__":
    OPS = orthoPlaneStress(E1=145.88e6, E2=13.312e6, G12=4.386e6, nu12=0.263, t=1.0, theta=0.0)
    print(OPS.DMat)
    OPS.theta = np.deg2rad(45.0)
    print(OPS.DMat)

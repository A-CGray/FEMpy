"""
==============================================================================
2D Isotropic plane stress constitutive model
==============================================================================
@File    :   NewIsoPlaneStress.py
@Date    :   2022/11/30
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy.Constitutive.StrainModels import Planar2DStrain
from FEMpy.Constitutive import ConstitutiveModel


class IsoPlaneStress(ConstitutiveModel):
    def __init__(self, E, nu, t, linear=True):
        """Create an isotropic plane stress constitutive model

        _extended_summary_

        Parameters
        ----------
        E : float
            Elastic Modulus
        nu : float
            Poisson's ratio
        t : float
            Thickness, this will be used as the initial thickness value for all elements but can be changed later by calling setDesignVariables in the model
        linear : bool, optional
            Whether to use the linear kinematic equations for strains, by default True
        """
        dvs = {}
        dvs["Thickness"] = {"Def"}
        self.super.__init__()

"""
==============================================================================
1D Timoshenko beam constitutive model
==============================================================================
@File    :   TimoshenkoBeam.py
@Date    :   2023/01/28
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
from FEMpy.Constitutive import ConstitutiveModel
from FEMpy.Constitutive.StrainModels import timoshenkoStrain, timoshenkoStrainUPrimeSens, timoshenkoStrainUSens
from FEMpy.Constitutive.StressModels import timoshenkoStress, timoshenkoStressStrainSens


class TimoshenkoBeam(ConstitutiveModel):
    """Constitutive model for a 1D Timoshenko beam

    Inherits
    ----------
    ConstitutiveModel : FEMpy.Constitutive.ConstitutiveModel
        The base class for FEMpy constitutive models
    """

    def __init__(self, E: float, rho: float, I: float, A: float, G: float, k: float) -> None:
        """Create a Timoshenko beam constitutive model

        Parameters
        ----------
        E : float
            Material elastic modulus
        rho : float
            Density
        I : numPoints array
            Second moment of area at each points
        A : numPoints array
            Cross-sectional area at each points
        G : float
            Material shear modulus
        k : float
            Shear correction factor
        """
        # --- Design variables ---
        designVars = {}
        designVars["Area"] = {"defaultValue": A}
        designVars["I"] = {"defaultValue": I}

        # --- States ---
        stateNames = ["Displacement", "Rotation"]

        # --- Strains ---
        strainNames = ["phi", "gamma"]

        # --- Stresses ---
        stressNames = ["M_11", "Q_1"]

        # --- Functions ---
        functionNames = ["Mass"]

        # --- Material properties ---
        self.E = E
        self.rho = rho
        self.G = G
        self.k = k

        numDim = 1

        super().__init__(numDim, stateNames, strainNames, stressNames, designVars, functionNames, linear=True)

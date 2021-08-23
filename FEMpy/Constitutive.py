"""
==============================================================================
Constitutive Relation Class
==============================================================================
@File    :   Constitutive.py
@Date    :   2021/03/12
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


class Constitutive:
    def __init__(self, numStrain, numDisp, numStress=None, numDim=None) -> None:
        self.numStrain = numStrain
        self.numStress = numStress if numStress is not None else numStrain
        self.numDim = numDim if numDim is not None else numDisp
        self.numDisp = numDisp
        # self.DMat = np.zeros((self.numStrain, self.numStress))
        self.LMats = np.zeros((self.numDim, self.numStrain, self.numDisp))

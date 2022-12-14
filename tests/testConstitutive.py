"""
==============================================================================
Constitutive class unit tests
==============================================================================
@File    :   testConstitutive.py
@Date    :   2022/12/08
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
from parameterized import parameterized_class
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy.Constitutive import IsoPlaneStrain, IsoPlaneStress, Iso3D


testParams = []

for cm in [IsoPlaneStrain, IsoPlaneStress, Iso3D]:
    for linear in [True, False]:
        if cm in [IsoPlaneStrain, IsoPlaneStress]:
            constitutiveModel = cm(E=70e9, nu=0.3, rho=2700, t=1.0, linear=linear)
        else:
            constitutiveModel = cm(E=70e9, nu=0.3, rho=2700, linear=linear)
        testParams.append({"ConstitutiveModel": constitutiveModel})


@parameterized_class(testParams)
class ElementUnitTest(unittest.TestCase):
    def setUp(self) -> None:

        self.tol = 1e-10
        np.random.seed(0)

        self.numPoints = 5

        self.numPerts = 3
        self.stepSize = 1e-200
        self.relCSTol = 1e-10
        self.absCSTol = 1e-10

        self.numStates = self.ConstitutiveModel.numStates
        self.numDim = self.ConstitutiveModel.numDim
        self.numStrains = self.ConstitutiveModel.numStrains
        self.u = np.random.rand(self.numPoints, self.numStates)
        self.coords = np.random.rand(self.numPoints, self.numDim)
        self.dudx = np.random.rand(self.numPoints, self.numStates, self.numDim)

        self.dvs = {}
        for dvName in self.ConstitutiveModel.designVariables:
            self.dvs[dvName] = (
                np.random.rand(self.numPoints) * 2 * self.ConstitutiveModel.designVariables[dvName]["defaultValue"]
            )

    def testStrainSens(self):
        """Validate the strain sensitivity calculation against complex-step"""
        strainSens = self.ConstitutiveModel.computeStrainStateGradSens(self.u, self.dudx, self.coords, self.dvs)

        for ii in range(self.numStates):
            for jj in range(self.numDim):
                dudx = self.dudx.astype("complex")
                dudx[:, ii, jj] += self.stepSize * 1j
                strainPert = self.ConstitutiveModel.computeStrains(self.u, dudx, self.coords, self.dvs)

                csSens = np.imag(strainPert) / self.stepSize
                np.testing.assert_allclose(csSens, strainSens[:, :, ii, jj], rtol=self.relCSTol, atol=self.absCSTol)

    def testStressStrainSens(self):
        """Validate the stress-strain sensitivity calculation against complex-step"""
        strain = self.ConstitutiveModel.computeStrains(self.u, self.dudx, self.coords, self.dvs)
        stressSens = self.ConstitutiveModel.computeStressStrainSens(strain, self.dvs)

        for ii in range(self.numStrains):
            strainPert = strain.astype("complex")
            strainPert[:, ii] += self.stepSize * 1j
            stressPert = self.ConstitutiveModel.computeStresses(strainPert, self.dvs)

            csSens = np.imag(stressPert) / self.stepSize
            np.testing.assert_allclose(csSens, stressSens[:, :, ii], rtol=self.relCSTol, atol=self.absCSTol)

    def testGetFunction(self):
        """Check that, for each function the constitutive model says it has, it provides a function that has the correct signature"""
        states = self.u
        stateGrad = self.dudx
        dvs = self.dvs
        coords = self.coords
        for funcName in self.ConstitutiveModel.functionNames:
            f = self.ConstitutiveModel.getFunction(funcName)
            values = f(states, stateGrad, coords, dvs)
            np.testing.assert_equal(values.shape, (self.numPoints,))

    def testGetWrongFunction(self):
        """Check that asking for a function that doesn't exist raises an error"""
        with self.assertRaises(ValueError):
            self.ConstitutiveModel.getFunction("wrongFunction")


if __name__ == "__main__":
    unittest.main()

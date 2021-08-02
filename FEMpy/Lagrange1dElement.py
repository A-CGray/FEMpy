"""
==============================================================================
1D Lagrangian Element
==============================================================================
@File    :   Lagrange1dElement.py
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
from FEMpy.Element import Element
from .LagrangePoly import LagrangePoly1d, LagrangePoly1dDeriv


class Lagrange1dElement(Element):
    """An arbitrary order 1D finite element using Lagrange polynomial shape functions"""

    def __init__(self, order):
        """Initialise an arbitrary order 1D finite element

        Parameters
        ----------
        order : int
            Shape function polynomial order
        """
        self.order = order
        super().__init__(numNodes=order + 1, numDimensions=1)
        self.name = f"Order{self.order}-Lagrange1D"

    def getShapeFunctions(self, paramCoords):
        """Compute shape function values at a set of parametric coordinates

        This function returns a zero array and should be re-implemented in any child classes

        Parameters
        ----------
        paramCoords : n x nDim array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        N : n x numNode array
            Shape function values, N[i][j] is the value of the jth shape function at the ith point
        """
        return LagrangePoly1d(paramCoords, n=self.order + 1)

    def getShapeFunctionDerivs(self, paramCoords):
        """Compute shape function derivatives at a set of parametric coordinates

        These are the derivatives of the shape functions with respect to the parametric coordinates (si, eta, gamma)

        This function returns a zero array and should be re-implemented in any child classes

        Parameters
        ----------
        paramCoords : n x nD array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        NPrime : n x numDim x numNode array
            Shape function values, N[i][j] is the value of the jth shape function at the ith point
        """
        NPrime = np.zeros((np.shape(paramCoords)[0], np.shape(paramCoords)[1], self.numNodes))
        NPrime[:, 0, :] = LagrangePoly1dDeriv(paramCoords, n=self.order + 1)
        return NPrime

    def getStiffnessMat(self, nodeCoords, constitutive, n=None):
        return super().getStiffnessMat(nodeCoords, constitutive, n=n) * constitutive.A

    def getMassMat(self, nodeCoords, constitutive, n=None):
        return super().getMassMat(nodeCoords, constitutive, n=n) * constitutive.A

    def _getRandomNodeCoords(self):
        """Generate a random, but valid, set of node coordinates for an element

        For a 1D element, we simply create evenly spaced points from 0 to one, add some slight random noise and then
        apply a random scaling factor

        Returns
        -------
        nodeCoords : numNode x numDim array
            Node coordinates
        """
        nodeCoords = np.random.rand(self.numNodes, 1) * 0.1
        nodeCoords[:, 0] += np.linspace(0, 1, self.numNodes)
        return nodeCoords * np.random.rand(1)

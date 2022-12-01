"""
==============================================================================
FEMpy: 2d Quad Element
==============================================================================
@File    :   newQuadElement.py
@Date    :   2022/11/27
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
from .NewElement import Element
from FEMpy.Basis import LagrangePoly as LP
from FEMpy.Quadrature import getGaussQuadWeights, getGaussQuadPoints


class QuadElement2D(Element):
    """An "arbitrary order" 2d quadrilateral finite element

    Arbitrary order is in quotes at the moment because although the shape functions can in theory be computed for an arbitrary with the current LagrangePoly implementation, I have not figured out how to element the node reordering required to reorder the shape functions into the node ordering used by MeshIO yet for anything more than 3rd order elements.

    Inherits from
    -------------
    Element : FEMpy.Element
        The FEMpy element base class
    """

    def __init__(self, order=1, numStates=None, quadratureOrder=None):
        self.order = order
        numNodes = (order + 1) ** 2
        if quadratureOrder is None:
            shapeFuncOrder = 2 * order
            quadratureOrder = int(np.ceil((shapeFuncOrder + 1) / 2))
        super().__init__(numNodes, numDimensions=2, numStates=numStates, quadratureOrder=quadratureOrder)

        self.name = f"Order{self.order}-LagrangeQuad"

        self.shapeFuncToNodeOrder = self._getNodeReordering(self.order)

    # ==============================================================================
    # Public methods
    # ==============================================================================

    def computeShapeFunctions(self, paramcoords):
        """Compute the shape function values at a given set of parametric coordinates

        Parameters
        ----------
        paramCoords : numPoint x numDim array
            Array of parametric point coordinates to evaluate shape functions at

        Returns
        -------
        N: numPoint x numNodes array
            Array of shape function values at the given parametric coordinates, N[i][j] is the value of the jth shape function at the ith parametric point
        """
        N = LP.LagrangePoly2d(paramcoords[:, 0], paramcoords[:, 1], self.order + 1)
        return np.ascontiguousarray(N[:, self.shapeFuncToNodeOrder])

    def computeShapeFunctionGradients(self, paramCoords):
        """Compute the derivatives of the shape functions with respect to the parametric coordinates at a given set of parametric coordinates

        _extended_summary_

        Parameters
        ----------
        paramCoords : numPoint x numDim array
            Array of parametric point coordinates to evaluate shape function gradients at

        Returns
        -------
        NGrad: numPoint x numDim x numNodes array
            Shape function gradient values, NGrad[i][j][k] is the value of the kth shape function at the ith point w.r.t the kth
            parametric coordinate
        """
        NPrimeParam = LP.LagrangePoly2dDeriv(paramCoords[:, 0], paramCoords[:, 1], self.order + 1)
        return np.ascontiguousarray(NPrimeParam[:, :, self.shapeFuncToNodeOrder])

    def getIntegrationPointWeights(self, order=None):
        """Compute the integration point weights for a given quadrature order on this element

        Parameters
        ----------
        order : int
            Integration order

        Returns
        -------
        array of length numIntpoint
            Integration point weights
        """
        if order is None:
            order = self.quadratureOrder
        return getGaussQuadWeights(self.numDim, order)

    def getIntegrationPointCoords(self, order=None):
        """Compute the integration point parameteric coordinates for a given quadrature order on this element

        Parameters
        ----------
        order : int
            Integration order

        Returns
        -------
        numIntpoint x numDim array
            Integration point coordinates
        """
        if order is None:
            order = self.quadratureOrder
        return getGaussQuadPoints(self.numDim, order)

    def getReferenceElementCoordinates(self):
        """Get the node coordinates for the reference element, a.k.a the element on which the shape functions are defined

        For the quad element, the nodes of the reference element are simply in a order+1 x order+1 grid over the range [-1, 1] in both x and y, reordered as described by _getNodeReordering

        Returns
        -------
        numNodes x numDim array
            Element node coordinates
        """
        x = np.tile(np.linspace(-1, 1, self.order + 1), self.order + 1)
        y = np.repeat(np.linspace(-1, 1, self.order + 1), self.order + 1)
        return np.vstack((x[self.shapeFuncToNodeOrder], y[self.shapeFuncToNodeOrder])).T

    # ==============================================================================
    # Private methods
    # ==============================================================================

    @staticmethod
    def _getNodeReordering(order):
        """Compute the reordering required between shape functions and nodes

        The 2d lagrange polynomial shape functions are ordered left to right, bottom to top, but the node ordering is defined differently, (e.g for a four node element it is bottom left, bottom right, top left, top right). This method computes the reordering required to map the shape functions to the correct node ordering. As of now I have simply manually implemented this for the first few orders, but it should be possible to compute this for any order with some sort of recursion.

        Parameters
        ----------
        order : int
            Quad element order

        Returns
        -------
        np.array
            Reordering array, array[i] = j indicates that the ith shape function should be reordered to the jth node
        """
        if order == 1:
            return np.array([0, 1, 3, 2])
        if order == 2:
            return np.array([0, 4, 1, 7, 8, 5, 3, 6, 2])
        if order == 3:
            return np.array([0, 4, 5, 1, 11, 12, 13, 6, 10, 15, 14, 7, 3, 9, 8, 2])

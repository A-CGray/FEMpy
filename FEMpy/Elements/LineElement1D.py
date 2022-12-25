"""
==============================================================================
FEMpy: 1d Line Element
==============================================================================
@File    :   LineElement1D.py
@Date    :   2022/12/20
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
from FEMpy.Elements import Element
from FEMpy.Basis import LagrangePoly as LP
from FEMpy.Quadrature import getGaussQuadWeights, getGaussQuadPoints


class LineElement1D(Element):
    """An abitrary order 1d line finite element

    The node ordering for the nth order line element looks like: ::

        0 --- 2 --- 3 ... n-1 --- n --- 1

    Inherits from
    -------------
    Element : FEMpy.Elements.Element
        The FEMpy element base class
    """

    def __init__(self, order=1, numStates=None, quadratureOrder=None):
        """Create a new 1d line element object

        Parameters
        ----------
        order : int, optional
            _description_, by default 1
        numStates : _type_, optional
            _description_, by default None
        quadratureOrder : _type_, optional
            _description_, by default None
        """
        if order < 1:
            raise ValueError("Order must be greater than 0")
        self.order = order
        numNodes = order + 1
        if quadratureOrder is None:
            # Compute quadrature order necessary to exactly intergrate polynomials of the same order as this element's
            # shape functions
            quadratureOrder = int(np.ceil((order + 1) / 2))

        super().__init__(numNodes, numDim=1, quadratureOrder=quadratureOrder, numStates=numStates)

        self.name = f"Order{self.order}-LagrangeLine"

        self.shapeFuncToNodeOrder = self._getNodeReordering(self.order)

    # ==============================================================================
    # Public methods
    # ==============================================================================
    def computeShapeFunctions(self, paramCoords):
        """Compute the shape function values at a given set of parametric coordinates

        Parameters
        ----------
        paramCoords : numPoint x numDim array
            Array of parametric point coordinates to evaluate shape functions at

        Returns
        -------
        N: numPoint x numNodes array
            Array of shape function values at the given parametric coordinates, N[i][j] is the value of the jth shape
            function at the ith parametric point
        """
        N = LP.LagrangePoly1d(paramCoords, self.order + 1)
        return np.ascontiguousarray(N[:, self.shapeFuncToNodeOrder])

    def computeShapeFunctionGradients(self, paramCoords):
        """Compute the derivatives of the shape functions with respect to the parametric coordinates at a given set of parametric coordinates



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
        NPrimeParam = np.zeros((paramCoords.shape[0], self.numDim, self.numNodes))
        NPrimeParam[:, 0, :] = LP.LagrangePoly1dDeriv(paramCoords, self.order + 1)
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
        x = np.atleast_2d(np.linspace(-1, 1, self.numNodes)).T
        return x[self.shapeFuncToNodeOrder]

    # ==============================================================================
    # Private methods
    # ==============================================================================

    @staticmethod
    def _getNodeReordering(order):
        """Compute the reordering required between shape functions and nodes

        The 1d lagrange polynomial shape functions are ordered left to right, but the node ordering is defined
        differently, (e.g for a four node element it is 0-2-3-1). This method computes the reordering required to map
        the shape functions to the correct node ordering.

        Parameters
        ----------
        order : int
            Line element order

        Returns
        -------
        np.array
            Reordering array, array[i] = j indicates that the ith shape function should be reordered to the jth node
        """
        ordering = np.arange(order + 1)
        ordering[-1] = 1
        ordering[1:-1] += 1
        return ordering

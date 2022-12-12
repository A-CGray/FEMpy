"""
==============================================================================

==============================================================================
@File    :   HexElement3D.py
@Date    :   2022/12/06
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
from FEMpy.Basis import LagrangePoly as LP
from FEMpy.Elements.Element import Element
from FEMpy.Quadrature import getGaussQuadWeights, getGaussQuadPoints


class HexElement3D(Element):
    """An `arbitrary order` 3d hexahedral finite element

    Like the QuadElement2D, the arbitrary order bit is in quotes because I have not figured out how to do the node reordering from the shape function ordering to the node ordering used by MeshIO for anything more than 3nd order hex elements yet

    Parameters
    ----------
    Element : _type_
        _description_
    """

    def __init__(self, order=1, numStates=None, quadratureOrder=None):
        """Create a new 3d hexahedral finite element object

        Parameters
        ----------
        order : int, optional
            Element order, a first order hex has 6 nodes, 2nd order 27 etc, currently only orders 1-3 are
            supported, by default 1
        numStates : int, optional
            Number of states in the underlying PDE, by default 3
        quadratureOrder : int, optional
            Quadrature order to use for numerical integration, by default None, in which case a valid order for the
            chosen element order is used

        Raises
        ------
        ValueError
            Raises error if order is not 1, 2 or 3
        """
        if order not in [1, 2]:
            raise ValueError("Hex elements only support orders 1 and 2")
        self.order = order
        numNodes = (order + 1) ** 3
        if quadratureOrder is None:
            shapeFuncOrder = 2 * order
            quadratureOrder = int(np.ceil((shapeFuncOrder + 1) / 2))
        super().__init__(numNodes, numDim=3, quadratureOrder=quadratureOrder, numStates=numStates)

        self.name = f"Order{self.order}-LagrangeHex"

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
            Array of shape function values at the given parametric coordinates, N[i][j] is the value of the jth shape function at the ith parametric point
        """
        N = LP.LagrangePoly3d(paramCoords[:, 0], paramCoords[:, 1], paramCoords[:, 2], self.order + 1)
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
        NPrimeParam = LP.LagrangePoly3dDeriv(paramCoords[:, 0], paramCoords[:, 1], paramCoords[:, 2], self.order + 1)
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
        numPoints = self.order + 1
        p = np.linspace(-1, 1, numPoints)
        x = np.tile(p, numPoints**2)
        y = np.tile(np.repeat(p, numPoints), numPoints)
        z = np.repeat(p, numPoints**2)
        return np.vstack((x[self.shapeFuncToNodeOrder], y[self.shapeFuncToNodeOrder], z[self.shapeFuncToNodeOrder])).T

    # ==============================================================================
    # Private methods
    # ==============================================================================

    @staticmethod
    def _getNodeReordering(order):
        """Compute the reordering required between shape functions and nodes

        The 23d lagrange polynomial shape functions are ordered left to right, front to back and then bottom to top, but the node ordering is defined differently in finite element meshes. This method computes the reordering required to map the shape functions to the correct node ordering. As of now I have simply manually implemented this for the first few orders, but it should be possible to compute this for any order with some sort of recursion.

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
            return np.array([0, 1, 3, 2, 4, 5, 7, 6])
        if order == 2:
            return np.array(
                [0, 2, 8, 6, 18, 20, 26, 24, 1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15, 12, 14, 10, 16, 4, 22, 13]
            )

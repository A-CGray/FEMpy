"""
==============================================================================
FEMpy 2D Tri element
==============================================================================
@File    :   TriElement2D.py
@Date    :   2022/12/04
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
from .Element import Element
from FEMpy.Basis import LagrangePoly as LP
from FEMpy.Quadrature import getTriQuadPoints, getTriQuadWeights


class TriElement2D(Element):
    """
    An up-to-3rd-order 2d triangular element with 3, 6 or 10 nodes respectively.

    The node numbering follows the meshio conventions shown below:

    1st order element::

        2
        |\\
        | \\
        |  \\
        |   \\
        |    \\
        0-----1

    2nd order element::

        2
        |\\
        | \\
        5  4
        |   \\
        |    \\
        0--3--1

    3rd Order::

        2
        |\\
        | \\
        7  6
        |   \\
        |    \\
        8  9  5
        |      \\
        |       \\
        0--3--4--1

    """

    def __init__(self, order=1, numStates=None, quadratureOrder=None):
        """Create a new 2d triangular finite element object

        Parameters
        ----------
        order : int, optional
            Element order, a first order quad has 4 nodes, 2nd order 9, 3rd order 16 etc, currently only orders 1-3 are
            supported, by default 1
        numStates : int, optional
            Number of states in the underlying PDE, by default 2
        quadratureOrder : int, optional
            Quadrature order to use for numerical integration, by default None, in which case a valid order for the
            chosen element order is used

        Raises
        ------
        ValueError
            Raises error if order is not 1, 2 or 3
        """
        if order not in [1, 2, 3]:
            raise ValueError("Triangular elements only support orders 1, 2 and 3")
        self.order = order
        numNodes = (order + 1) * (order + 2) // 2
        if quadratureOrder is None:
            shapeFuncOrder = order
            quadratureOrder = int(np.ceil((shapeFuncOrder + 1) / 2))

        super().__init__(numNodes, numDim=2, quadratureOrder=quadratureOrder, numStates=numStates)

        self.name = f"Order{self.order}-LagrangeTri"

        # --- Define parametric coordinate bounds ---
        self.paramCoordLowerBounds = -np.zeros(self.numDim)
        self.paramCoordLinearConstaintMat = np.array([1.0, 1.0])
        self.paramCoordLinearConstaintUpperBounds = 1.0
        self.paramCoordLinearConstaintLowerBounds = -np.inf

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
        return LP.LagrangePolyTri(paramCoords[:, 0], paramCoords[:, 1], self.order)

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
        return LP.LagrangePolyTriDeriv(paramCoords[:, 0], paramCoords[:, 1], self.order)

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
        return getTriQuadWeights(order)

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
        return getTriQuadPoints(order)

    def getReferenceElementCoordinates(self):
        """Get the node coordinates for the reference element, a.k.a the element on which the shape functions are defined

        Returns
        -------
        numNodes x numDim array
            Element node coordinates
        """
        if self.order == 1:
            return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        elif self.order == 2:
            return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
        elif self.order == 3:
            return np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1 / 3, 0.0],
                    [2 / 3, 0.0],
                    [2 / 3, 1 / 3],
                    [1 / 3, 2 / 3],
                    [0.0, 2 / 3],
                    [0.0, 1 / 3],
                    [1 / 3, 1 / 3],
                ]
            )

    def getRandParamCoord(self, n):
        """Generate a set of random parametric coordinates
        For a tri element we need u and v in range [0,1] and u + v <= 1, we can generate these points by generating
        random points in a square on the domain [0,1] and then reflecting any points outside the triangle to the inside.
        Parameters
        ----------
        n : int, optional
            number of points to generate, by default 1
        Returns
        -------
        paramCoords : n x numDim array
            isoparametric coordinates, one row for each point
        """
        coords = np.atleast_2d(np.random.rand(n, 2))
        for i in range(n):
            coordSum = coords[i, 0] + coords[i, 1]
            if coordSum > 1:
                coords[i] = 1 - coords[i]
        return coords

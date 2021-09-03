"""
==============================================================================
T3 Element
==============================================================================
@File    :   TriElement.py
@Date    :   2021/03/11
@Author  :   Alasdair Christison Gray
@Description : This file contains a class that implements a 2D, up to 3rd order, triangular finite element
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
from . import LagrangePoly as LP

# TODO: Implement gaussian quadrature integration fr=or triangles
# TODO: Alter general element class to integrate based on values returned from getIntegrationPoints and getIntegrationWeights methods
class TriElement(Element):
    def __init__(self, order=1, numDisplacements=2):
        """Instantiate a triangular element

        Parameters
        ----------
        order : int, optional
            Element order, by default 1
        numDisplacements : int, optional
            Number of variables at each node, by default 2
        """

        self.order = order
        nodes = (order + 1) * (order + 2) // 2
        super().__init__(numNodes=nodes, numDimensions=2, numDisplacements=numDisplacements)

        self.name = f"Order{self.order}-LagrangeTri"

    def getShapeFunctions(self, paramCoords):
        """Compute shape function values at a set of parametric coordinates


        Parameters
        ----------
        paramCoords : n x nDim array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        N : n x numNode array
            Shape function values, N[i][j] is the value of the jth shape function at the ith point
        """
        return LP.LagrangePolyTri(paramCoords[:, 0], paramCoords[:, 1], self.order)

    def getShapeFunctionDerivs(self, paramCoords):
        """Compute shape function derivatives at a set of parametric coordinates

        These are the derivatives of the shape functions with respect to the parametric coordinates (si, eta, gamma)

        Parameters
        ----------
        paramCoords : n x nD array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        NPrime : n x numDim x numNode array
            Shape function values, N[i][j][k] is the value of the kth shape function at the ith point w.r.t the kth
            parametric coordinate
        """
        return LP.LagrangePolyTriDeriv(paramCoords[:, 0], paramCoords[:, 1], self.order)

    def _getRandomNodeCoords(self):
        """Generate a set of random node coordinates

        First create 3 random points and put them in CCW order, then add in higher order nodes if required, then add noise
        then apply a random scaling and rotation
        """

        xy = np.zeros((3, 2))
        xyOrder = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
        nodeCoords = np.zeros((self.numNodes, 2))
        nodeCoords[:3] = xy[xyOrder]

        if self.order > 1:
            edgeFrac = 1.0 / (self.order)
            for i in range(self.order - 1):
                for j in range(3):
                    nodeCoords[3 + j * (self.order - 1)] = nodeCoords[j] + edgeFrac * i * (
                        nodeCoords[j % 3] - nodeCoords[j]
                    )
        if self.order == 3:
            nodeCoords[-1] = 1.0 / 9.0 * np.sum(nodeCoords[:-1], axis=0)
        for i in range(2):
            nodeCoords[:, i] += np.random.rand(self.numNodes) * 0.1
        nodeCoords *= np.random.rand(1)
        theta = np.random.rand(1) * np.pi
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])[:, :, 0]
        return (R @ nodeCoords.T).T

    def _getRandParamCoord(self, n=1):
        """Generate a set of random parametric coordinates

        For a tri element we need u and v in range [0,1] and u + v <= 1

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
        coords[:, 1] *= 1.0 - coords[:, 0]
        return coords

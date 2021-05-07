"""
==============================================================================
Q4 Element
==============================================================================
@File    :   QuadElement.py
@Date    :   2021/03/11
@Author  :   Alasdair Christison Gray
@Description : This file contains a class that implements a 2D, 4-Node quadrilateral finite element
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
from .GaussQuad import gaussQuad1d


class QuadElement(Element):
    def __init__(self, order=1, numDisplacements=2):
        """Instantiate an arbitrary order 2d quadrilateral finite element

        Note that this element does not use the typical CCW node ordering, to make it simpler to work with arbitrary
        element orders, the nodes are ordered left to right followed by bottom to top, so for a 1st order element the
        ordering is:

        1) Bottom left
        2) Bottom Right
        3) Top left
        4) Top right

        Parameters
        ----------
        order : int, optional
            Element order, by default 1
        numDisplacements : int, optional
            Number of variables at each node, by default 2
        """

        self.order = order
        nodes = (order + 1) ** 2
        super().__init__(numNodes=nodes, numDimensions=2, numDisplacements=numDisplacements)

        # bottom, right, top, left order,
        # 0 means psi varies along edge, 1 means eta varies along edge
        self.edgeFreeCoord = [0, 1, 0, 1]
        # value of the fixed coordinate on each edge
        self.edgeFixedCoord = [-1.0, 1.0, 1.0, -1.0]

    def getShapeFunctions(self, paramCoords):
        """Compute shape function values at a set of parametric coordinates

        [extended_summary]

        Parameters
        ----------
        paramCoords : n x nDim array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        N : n x numNode array
            Shape function values, N[i][j] is the value of the jth shape function at the ith point
        """
        return LP.LagrangePoly2d(paramCoords[:, 0], paramCoords[:, 1], self.order + 1)

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
            Shape function values, N[i][j] is the value of the jth shape function at the ith point
        """
        return LP.LagrangePoly2dDeriv(paramCoords[:, 0], paramCoords[:, 1], self.order + 1)

    def getStiffnessMat(self, nodeCoords, constitutive, n=None):
        # TODO: this function should not presume the constitutive object has a thickness, what if you want to use this
        # quad element for something else, like 2D heat transfer where there's no thickness
        return super().getStiffnessMat(nodeCoords, constitutive, n=n) * constitutive.t

    def integrateBodyForce(self, f, nodeCoords, constitutive, n=1):
        # TODO: this function should not presume the constitutive object has a thickness, what if you want to use this
        # quad element for something else, like 2D heat transfer where there's no thickness
        return super().integrateBodyForce(f, nodeCoords, n=n) * constitutive.t

    def integrateTraction(self, f, nodeCoords, constitutive, edges=[0, 1, 2, 3], n=1):
        """Compute equivalent nodal forces due to body forces through numerical integration

        [extended_summary]

        Parameters
        ----------
        f : Body force function
            Should accept an nP x numDim array as input and output a nP x numDisp array, ie f(x)[i] returns the body force components at the ith point queried
        nodeCoords : numNode x numDim array
            Element node real coordinates
        n : int, optional
            Number of integration points, can be a single value or a list with a value for each direction, by default 1
        constitutive : [type]
            [description]
        edges : list, optional
            [description], by default [1,2,3,4]

        Returns
        -------
        Fb : numNode x numDisp array
            Equivalent nodal loads due to traction forces force
        """
        if isinstance(edges, (int, np.integer)):
            edges = [edges]
        Ft = np.zeros((self.numNodes, self.numDisp))
        for e in edges:
            if self.edgeFreeCoord[e] == 0:
                func = lambda x1: self.tractionIntegrand(  # noqa: E731
                    f, np.array([x1, self.edgeFixedCoord[e] * np.ones_like(x1)]).T, nodeCoords, e
                )
            else:
                func = lambda x2: self.tractionIntegrand(  # noqa: E731
                    f, np.array([self.edgeFixedCoord[e] * np.ones_like(x2), x2]).T, nodeCoords, e
                )
            Ft += gaussQuad1d(func, n)
        return Ft * constitutive.t

    def tractionIntegrand(self, f, paramCoord, nodeCoords, edgeNum):
        # Compute shape functions and Jacobian determinant at parametric coordinates
        N = self.getShapeFunctions(paramCoord)
        J = self.getJacobian(paramCoord, nodeCoords)
        detJStar = np.linalg.norm(J[:, self.edgeFreeCoord[edgeNum], :], axis=-1)

        # Transform parametric to real coordinates in order to compute body force components
        realCoord = self.getRealCoord(paramCoord, nodeCoords)
        F = f(realCoord)

        # Compute N^T fb at each point, it's complicated because things are not the right shape
        nP = np.shape(F)[0]
        nD = np.shape(F)[1]
        nN = np.shape(N)[1]
        Fb = np.zeros((nP, nN, nD))
        for p in range(nP):
            for d in range(nD):
                Fb[p, :, d] = (F[p, d] * N[p]).T
        return (Fb.T * detJStar).T


if __name__ == "__main__":

    QuadElem = QuadElement()

    # Test that we get the expected output when the original element matches the isoparametric element
    nodecoords = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    uNodes = np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])
    paramCoords = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    print(QuadElem.getShapeFunctions(paramCoords), "\n")
    print(QuadElem.getRealCoord(paramCoords, nodecoords), "\n")
    print(QuadElem.getU(paramCoords, uNodes), "\n")
    print(QuadElem.getUPrime(paramCoords, nodecoords, uNodes), "\n")
    print(QuadElem.getJacobian(paramCoords, nodecoords), "\n\n\n")

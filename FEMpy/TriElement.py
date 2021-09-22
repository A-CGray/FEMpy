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
try:
    from .Element import Element
except ImportError:
    from FEMpy.Element import Element
try:
    from . import LagrangePoly as LP
except ImportError:
    import FEMpy.LagrangePoly as LP

# TODO: Implement gaussian quadrature integration fr=or triangles
# TODO: Alter general element class to integrate based on values returned from getIntegrationPoints and getIntegrationWeights methods
class TriElement(Element):
    """
    An up-to-3rd-order 2d triangular element with 3, 6 or 10 nodes respectively. The edges of the triangle share the
    number of the node opposite them, so edge 1 is opposite node 1 (between nodes 2 and 3) and so on. In the second and
    3rd order elements, the first 3 nodes are still the corners of the triangle, the remaining mid-edge nodes are
    numbered sequentially along each edge, starting at edge 1. In the 3rd order element the final node is at the
    element centroid. If this doesn't make sense to you, just look at my nice ascii art below.

    3
    |\
    | \
    6  5
    |   \
    |    \
    7  10 4
    |      \
    |       \
    1--8--9--2
    """

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

    def getParamCoord(self, realCoords, nodeCoords, maxIter=10, tol=1e-8):
        """Find the parametric coordinates within an element corresponding to a point in real space

        This function is only reimplemented here so we can pass a better starting guess

        Parameters
        ----------
        realCoords : array of length numDim
            Real coordinates to find the paranmetric coordinates of the desired point
        nodeCoords : numNode x numDim array
            Element node real coordinates
        maxIter : int, optional
            Maximum number of search iterations, by default 4

        Returns
        -------
        x : array of length numDim
            Parametric coordinates of the desired point
        """
        return super().getParamCoord(realCoords, nodeCoords, maxIter, tol, x0=0.333 * np.ones(2))

    def _getRandomNodeCoords(self):
        """Generate a set of random node coordinates

        First create 3 random points and put them in CCW order, then add in higher order nodes if required, then add noise
        then apply a random scaling and rotation
        """

        nodeCoords = np.zeros((self.numNodes, 2))
        theta = np.linspace(0, 2 * np.pi, 4)
        nodeCoords[:3, 0] = np.cos(theta[:-1] + np.random.rand(3) * np.pi / 8)
        nodeCoords[:3, 1] = np.sin(theta[:-1] + np.random.rand(3) * np.pi / 8)

        if self.order > 1:
            edgeFrac = 1.0 / (self.order)
            for e in range(3):
                for i in range(self.order - 1):
                    nodeCoords[3 + e * (self.order - 1) + i] = nodeCoords[(e + 1) % 3] + (i + 1) * edgeFrac * (
                        nodeCoords[(e + 2) % 3] - nodeCoords[(e + 1) % 3]
                    )
        if self.order == 3:
            nodeCoords[-1] = 1.0 / 9.0 * np.sum(nodeCoords[:-1], axis=0)
        # for i in range(2):
        #     nodeCoords[:, i] += np.random.rand(self.numNodes) * 0.02
        nodeCoords *= 0.2 + (2.0 - 0.2) * np.random.rand(1)
        theta = np.random.rand(1) * np.pi
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])[:, :, 0]
        return (R @ nodeCoords.T).T

    def _getRandParamCoord(self, n=1):
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
                coords[i] -= 1 - coordSum
        return coords


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import niceplots

    niceplots.setRCParams()

    el = TriElement(order=2)
    x = np.linspace(0, 1.0, 101)
    psi, eta = np.meshgrid(x, x)
    p = psi.flatten()
    e = eta.flatten()
    N = el.getShapeFunctions(np.array([p, e]).T)

    numSF = el.numNodes
    nrows = int(np.floor(np.sqrt(numSF)))
    ncols = numSF / nrows
    if ncols % 1 == 0:
        ncols = int(ncols)
    else:
        ncols = int(ncols) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten()
    NSum = np.zeros((101, 101))
    for i in range(len(axes)):
        if i < numSF:
            ax = axes[i]
            Ni = N[:, i].reshape((101, 101))
            NSum += Ni
            ax.contourf(psi, eta, np.where(psi + eta > 1.0, np.nan, Ni), cmap="coolwarm", levels=np.linspace(-1, 1, 21))
            niceplots.adjust_spines(ax, outward=True)
            ax.set_title(f"N{i+1}")
        elif i == numSF:
            ax = axes[i]
            ax.contourf(
                psi, eta, np.where(psi + eta > 1.0, np.nan, NSum), cmap="coolwarm", levels=np.linspace(-1, 1, 21)
            )
            niceplots.adjust_spines(ax, outward=True)
            ax.set_title("$\sum N_i$")
        else:
            axes[i].set_axis_off()

    plt.show()

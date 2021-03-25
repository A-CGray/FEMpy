"""
==============================================================================
Element Class
==============================================================================
@File    :   Element.py
@Date    :   2021/03/11
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
from numba import jit

# ==============================================================================
# Extension modules
# ==============================================================================
from .GaussQuad import gaussQuad1d, gaussQuad2d, gaussQuad3d


@jit(nopython=True, cache=True)
def makeBMat(NPrime, LMats, numStrain, numDim, numNodes):
    numPoints = np.shape(NPrime)[0]
    BMat = np.zeros((numPoints, numStrain, numDim * numNodes))
    for p in range(numPoints):
        for n in range(numNodes):
            for d in range(numDim):
                BMat[p, :, n * numDim : (n + 1) * numDim] += LMats[d] * NPrime[p, d, n]
    return BMat


@jit(nopython=True, cache=True)
def _bodyForceInt(F, N):
    # Compute N^T fb at each point, it's complicated because things are not the right shape
    nP = np.shape(F)[0]
    nD = np.shape(F)[1]
    nN = np.shape(N)[1]
    Fb = np.zeros((nP, nN, nD))
    for p in range(nP):
        for d in range(nD):
            Fb[p, :, d] = (F[p, d] * N[p]).T
    return Fb


class Element(object):
    def __init__(self, numNodes, numDimensions, numDisplacements=None):
        """Instantiate an Element object

        Parameters
        ----------
        numNodes : int
            Number of nodes in each element
        numDimensions : int
            Number of spatial dimensions the element models
        numDisplacements : int, optional
            Number of displacements at each node, by default this is set equal to the number of spatial dimensions
        """
        self.numNodes = numNodes
        self.numDim = numDimensions
        self.numDisp = numDimensions if numDisplacements is None else numDisplacements
        self.numDOF = numNodes * numDisplacements

    def getRealCoord(self, paramCoords, nodeCoords):
        """Compute the real coordinates of a point in isoparametric space

        [extended_summary]

        Parameters
        ----------
        paramCoords : n x numDim array
            isoparametric coordinates, one row for each point in isoparametric space to be converted
        nodeCoords : numNode x numDim array
            Element node real coordinates

        Returns
        -------
        coords : n x nD array
            Point coordinates in real space
        """
        N = self.getShapeFunctions(paramCoords)

        return N[:, : self.numNodes] @ nodeCoords

    def getParamCoord(self, realCoords, nodeCoords, maxIter=4):
        """Find the parametric coordinates within an element corresponding to a point in real space

        Note this function only currently works for finding the parametric coordinates of one point inside one element

        Parameters
        ----------
        realCoords : array of length numDim
            Real coordinates to find the paranmetric coordinates of
        nodeCoords : numNode x numDim array
            Element node real coordinates
        maxIter : int, optional
            Maximum number of search iterations, by default 4

        Returns
        -------
        x : array of length numDim
            Parametric coordinates of the desired point
        """
        x = np.zeros(self.numDim)
        for i in range(maxIter):
            res = realCoords - self.getRealCoord(np.array([x]), nodeCoords).flatten()
            if np.sum(res ** 2) < 1e-6:
                break
            else:
                jacT = self.getJacobian(np.array([x]), nodeCoords)[0].T
                x += np.linalg.solve(jacT, res)
        return x

    def getJacobian(self, paramCoords, nodeCoords):
        """Get the element Jacobians at a set of parametric coordinates

        [extended_summary]

        Parameters
        ----------
        paramCoords : n x nD array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at
        nodeCoords : numNode x numDim array
            Element node real coordinates

        Returns
        -------
        Jac : n x numDim x numDim array
            The Jacobians at each point
        """
        return self.getShapeFunctionDerivs(paramCoords) @ nodeCoords

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
        return np.zeros((np.shape(paramCoords)[0], self.numNodes))

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
        return np.zeros((np.shape(paramCoords)[0], self.numDim, self.numNodes))

    def getNPrime(self, paramCoords, nodeCoords):
        """Compute shape function derivatives at a set of parametric coordinates

        These are the derivatives of the shape functions with respect to the real coordinates (x,y,z)

        Parameters
        ----------
        paramCoords : n x nD array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at
        nodeCoords : numNode x numDim array
            Element node real coordinates

        Returns
        -------
        NPrime : n x numDim x numNode array
            [description]
        """
        NPrimeParam = self.getShapeFunctionDerivs(paramCoords)
        # The Jacobian is NPrimeParam * nodeCoords so we don't need to waste time recomputing NPrimeParam inside the
        # getJacobian function
        return np.linalg.inv(NPrimeParam @ nodeCoords) @ NPrimeParam

    def getBMat(self, paramCoords, nodeCoords, constitutive):
        """Compute the element B matrix at a set of parametric coordinates

        The B matrix is the matrix that converts nodal DOF's to strains

        strain = B*q

        K = int (B^T * D * B) dv

        Parameters
        ----------

        paramCoords : n x nD array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at
        nodeCoords : numNode x numDim array
            Element node real coordinates

        Returns
        -------
        B : n x numStrain x (numNode*numDim) array
            The B matrices, B[i] returns the 2D B matrix at the ith parametric point
        """
        NPrime = self.getNPrime(paramCoords, nodeCoords)
        return makeBMat(NPrime, constitutive.LMats, constitutive.numStrain, self.numDim, self.numNodes)
        # return np.zeros((np.shape(paramCoords)[0], self.nStrain, self.numNodes * self.numDim))

    def getStress(self, paramCoords, nodeCoords, constitutive, uNodes):
        BMat = self.getBMat(paramCoords, nodeCoords, constitutive)
        return constitutive.DMat @ BMat @ uNodes.flatten()

    def getU(self, paramCoords, uNodes):
        """Compute the displacements at a set of parametric coordinates

        [extended_summary]

        Parameters
        ----------
        paramCoords : n x numDim array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at
        uNodes : numNode x numDim array
            Nodal displacements

        Returns
        -------
        u : n x numDim array
            Array of displacement values, u[i,j] is the jth displacement component at the ith point
        """
        N = self.getShapeFunctions(paramCoords)
        return N @ uNodes

    def getUPrime(self, paramCoords, nodeCoords, uNodes):
        NPrime = self.getNPrime(paramCoords, nodeCoords)
        return NPrime @ uNodes

    def getStiffnessIntegrand(self, paramCoords, nodeCoords, constitutive):
        B = self.getBMat(paramCoords, nodeCoords, constitutive)
        J = self.getJacobian(paramCoords, nodeCoords)
        detJ = np.linalg.det(J)
        BDB = np.swapaxes(B, 1, 2) @ constitutive.DMat @ B
        return (BDB.T * detJ).T

    def getStiffnessMat(self, nodeCoords, constitutive, n=None):
        if n is None:
            n = self.order + 1
        if self.numDim == 1:
            f = lambda x1: self.getStiffnessIntegrand(np.array([x1]).T, nodeCoords, constitutive)
            return gaussQuad1d(f=f, n=n)
        if self.numDim == 2:
            f = lambda x1, x2: self.getStiffnessIntegrand(np.array([x1, x2]).T, nodeCoords, constitutive)
            return gaussQuad2d(f=f, n=n)
        if self.numDim == 3:
            f = lambda x1, x2, x3: self.getStiffnessIntegrand(np.array([x1, x2, x3]).T, nodeCoords, constitutive)
            return gaussQuad3d(f, n)

    def integrateBodyForce(self, f, nodeCoords, n=1):
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

        Returns
        -------
        Fb : numNode x numDisp array
            Equivalent nodal loads due to body force
        """
        if self.numDim == 1:
            bodyForceFunc = lambda x1: self.bodyForceIntegrad(f, np.array([x1]).T, nodeCoords)
            return gaussQuad1d(bodyForceFunc, n)
        if self.numDim == 2:
            bodyForceFunc = lambda x1, x2: self.bodyForceIntegrad(f, np.array([x1, x2]).T, nodeCoords)
            return gaussQuad2d(bodyForceFunc, n)
        if self.numDim == 3:
            bodyForceFunc = lambda x1, x2, x3: self.bodyForceIntegrad(f, np.array([x1, x2, x3]).T, nodeCoords)
            return gaussQuad3d(bodyForceFunc, n)

    def bodyForceIntegrad(self, f, paramCoord, nodeCoords):
        # Compute shape functions and Jacobian determinant at parametric coordinates
        N = self.getShapeFunctions(paramCoord)
        J = self.getJacobian(paramCoord, nodeCoords)
        detJ = np.linalg.det(J)
        # Transform parametric to real coordinates in order to compute body force components
        realCoord = self.getRealCoord(paramCoord, nodeCoords)
        F = f(realCoord)
        Fb = _bodyForceInt(F, N)
        return detJ * Fb


if __name__ == "__main__":
    QuadElem = Element(numNodes=4, numDimensions=2, numStrains=3)
    nodecoords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    uNodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    paramCoords = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    print(QuadElem.getShapeFunctions(paramCoords), "\n")
    print(QuadElem.getRealCoord(paramCoords, nodecoords), "\n")
    print(QuadElem.getU(paramCoords, uNodes), "\n")
    print(QuadElem.getUPrime(paramCoords, nodecoords, uNodes), "\n")

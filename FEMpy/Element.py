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
import abc

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from numba import njit
from scipy.optimize import root

# ==============================================================================
# Extension modules
# ==============================================================================
from .GaussQuad import gaussQuad1d, gaussQuad2d, gaussQuad3d
from .LinAlg import det1, det2, det3, inv1, inv2, inv3


@njit(cache=True)
def _makeBMat(NPrime, LMats, numStrain, numDim, numNodes):
    numPoints = np.shape(NPrime)[0]
    BMat = np.zeros((numPoints, numStrain, numDim * numNodes))
    for p in range(numPoints):
        for n in range(numNodes):
            for d in range(numDim):
                BMat[p, :, n * numDim : (n + 1) * numDim] += LMats[d] * NPrime[p, d, n]
    return BMat


@njit(cache=True)
def _makeNMat(N, numDim):
    s = np.shape(N)
    numPoints = s[0]
    numShapeFunc = s[1]
    NMat = np.zeros((numPoints, numDim, numDim * numShapeFunc))
    for p in range(numPoints):
        for n in range(numShapeFunc):
            NMat[p, :, numDim * n : numDim * (n + 1)] = N[p, n] * np.eye(numDim)
    return NMat


@njit(cache=True)
def _computeNTFProduct(F, N):
    # Compute N^T fb at each point, it's complicated because things are not the right shape
    nP = np.shape(F)[0]
    nD = np.shape(F)[1]
    nN = np.shape(N)[1]
    Fb = np.zeros((nP, nN, nD))
    for p in range(nP):
        for d in range(nD):
            Fb[p, :, d] = (F[p, d] * N[p]).T
    return Fb


class Element:
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
        self.numDOF = numNodes * self.numDisp
        self.name = f"{self.numNodes}Node-{self.numDisp}Disp-{self.numDim}D-Element"

        # --- Define fast jacobian determinant function based on number of dimensions ---
        if self.numDim == 1:
            self.jacDet = det1
            self.jacInv = inv1
        elif self.numDim == 2:
            self.jacDet = det2
            self.jacInv = inv2
        elif self.numDim == 3:
            self.jacDet = det3
            self.jacInv = inv3

    def getRealCoord(self, paramCoords, nodeCoords):
        """Compute the real coordinates of a point in isoparametric space


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

    def getParamCoord(self, realCoords, nodeCoords, maxIter=10, tol=1e-8):
        """Find the parametric coordinates within an element corresponding to a point in real space

        Note this function only currently works for finding the parametric coordinates of one point inside one element

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
        # x = np.zeros(self.numDim)
        # for i in range(maxIter):
        #     res = realCoords - self.getRealCoord(np.array([x]), nodeCoords).flatten()
        #     if np.max(np.abs(res)) < tol:
        #         break
        #     else:
        #         jacT = self.getJacobian(np.array([x]), nodeCoords)[0].T
        #         x += np.linalg.solve(jacT, res)
        # return x

        def resFunc(x):
            return realCoords - self.getRealCoord(np.array([x]), nodeCoords).flatten()

        sol = root(resFunc, np.zeros(self.numDim), method="krylov", tol=tol)
        return sol.x

    def getJacobian(self, paramCoords, nodeCoords):
        """Get the element Jacobians at a set of parametric coordinates


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

    @abc.abstractmethod
    def getShapeFunctions(self, paramCoords):
        """Compute shape function values at a set of parametric coordinates

        This function must be implemented in any child classes

        Parameters
        ----------
        paramCoords : n x nDim array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        N : n x numNode array
            Shape function values, N[i][j] is the value of the jth shape function at the ith point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getShapeFunctionDerivs(self, paramCoords):
        """Compute shape function derivatives at a set of parametric coordinates

        These are the derivatives of the shape functions with respect to the parametric coordinates (si, eta, gamma)

        This function must be implemented in any child classes

        Parameters
        ----------
        paramCoords : n x nD array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at

        Returns
        -------
        NPrimeParam : n x numDim x numNode array
            Shape function values, N[i][j][k] is the value of the kth shape function at the ith point w.r.t the kth
            parametric coordinate
        """
        raise NotImplementedError

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
        return self.jacInv(NPrimeParam @ nodeCoords) @ NPrimeParam

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
        return self._makeBMat(NPrime, constitutive)

    def getStrain(self, paramCoords, nodeCoords, constitutive, uNodes):
        BMat = self.getBMat(paramCoords, nodeCoords, constitutive)
        return BMat @ uNodes.flatten()

    def getStress(self, paramCoords, nodeCoords, constitutive, uNodes):
        return self.getStrain(paramCoords, nodeCoords, constitutive, uNodes) @ constitutive.DMat

    def getU(self, paramCoords, uNodes):
        """Compute the displacements at a set of parametric coordinates


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
        """Compute the displacement derivatives at a set of parametric coordinates

        Parameters
        ----------
        paramCoords : n x numDim array
            isoparametric coordinates, one row for each point in isoparametric space to compute the Jacobian at
        nodeCoords : numNode x numDim array
            Element node real coordinates
        uNodes : numNode x numDim array
            Nodal displacements

        Returns
        -------
        NPrime : n x numDim x numNode array
            [description]
        """
        NPrime = self.getNPrime(paramCoords, nodeCoords)
        return NPrime @ uNodes

    def getStiffnessIntegrand(self, paramCoords, nodeCoords, constitutive):
        B = self.getBMat(paramCoords, nodeCoords, constitutive)
        J = self.getJacobian(paramCoords, nodeCoords)
        detJ = self.jacDet(J)
        BDB = np.swapaxes(B, 1, 2) @ constitutive.DMat @ B
        return (BDB.T * detJ).T

    def getStiffnessMat(self, nodeCoords, constitutive, n=None):
        if n is None:
            n = self.order + 1
        if self.numDim == 1:
            f = lambda x1: self.getStiffnessIntegrand(np.array([x1]).T, nodeCoords, constitutive)  # noqa: E731
            return gaussQuad1d(f=f, n=n)
        if self.numDim == 2:
            f = lambda x1, x2: self.getStiffnessIntegrand(np.array([x1, x2]).T, nodeCoords, constitutive)  # noqa: E731
            return gaussQuad2d(f=f, n=n)
        if self.numDim == 3:
            f = lambda x1, x2, x3: self.getStiffnessIntegrand(  # noqa: E731
                np.array([x1, x2, x3]).T, nodeCoords, constitutive
            )
            return gaussQuad3d(f, n)

    def getMassMat(self, nodeCoords, constitutive, n=None):
        if n is None:
            n = self.order + 1
        if self.numDim == 1:
            f = lambda x1: self.getMassIntegrand(np.array([x1]).T, nodeCoords, constitutive)  # noqa: E731
            return gaussQuad1d(f=f, n=n)
        if self.numDim == 2:
            f = lambda x1, x2: self.getMassIntegrand(np.array([x1, x2]).T, nodeCoords, constitutive)  # noqa: E731
            return gaussQuad2d(f=f, n=n)
        if self.numDim == 3:
            f = lambda x1, x2, x3: self.getMassIntegrand(  # noqa: E731
                np.array([x1, x2, x3]).T, nodeCoords, constitutive
            )
            return gaussQuad3d(f, n)

    def getMassIntegrand(self, paramCoords, nodeCoords, constitutive):
        N = self.getShapeFunctions(paramCoords)
        NMat = self._makeNMat(N)
        J = self.getJacobian(paramCoords, nodeCoords)
        detJ = self.jacDet(J)
        NTN = np.swapaxes(NMat, 1, 2) @ NMat * constitutive.rho
        return (NTN.T * detJ).T

    def integrateBodyForce(self, f, nodeCoords, n=1):
        """Compute equivalent nodal forces due to body forces through numerical integration


        Parameters
        ----------
        f : Body force function
            Should accept an nP x numDim array as input and output a nP x numDisp array, ie f(x)[i] returns the body
            force components at the ith point queried
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
            bodyForceFunc = lambda x1: self.bodyForceIntegrad(f, np.array([x1]).T, nodeCoords)  # noqa: E731
            return gaussQuad1d(bodyForceFunc, n)
        if self.numDim == 2:
            bodyForceFunc = lambda x1, x2: self.bodyForceIntegrad(f, np.array([x1, x2]).T, nodeCoords)  # noqa: E731
            return gaussQuad2d(bodyForceFunc, n)
        if self.numDim == 3:
            bodyForceFunc = lambda x1, x2, x3: self.bodyForceIntegrad(  # noqa: E731
                f, np.array([x1, x2, x3]).T, nodeCoords
            )
            return gaussQuad3d(bodyForceFunc, n)

    def _computeNTFProduct(self, F, N):
        """A basic wrapper for the jit compiled function _computeNTFProduct"""
        return _computeNTFProduct(F, N)

    def _makeNMat(self, N):
        """A basic wrapper for the jit compiled function _makeNMat"""
        return _makeNMat(N, self.numDim)

    def _makeBMat(self, NPrime, constitutive):
        """A basic wrapper for the jit compiled function _makeBMat"""
        return _makeBMat(
            NPrime,
            constitutive.LMats,
            constitutive.numStrain,
            self.numDim,
            self.numNodes,
        )

    def bodyForceIntegrad(self, f, paramCoord, nodeCoords):
        # Compute shape functions and Jacobian determinant at parametric coordinates
        N = self.getShapeFunctions(paramCoord)
        J = self.getJacobian(paramCoord, nodeCoords)
        detJ = self.jacDet(J)
        # Transform parametric to real coordinates in order to compute body force components
        realCoord = self.getRealCoord(paramCoord, nodeCoords)
        F = f(realCoord)
        Fb = self._computeNTFProduct(F, N)
        return (Fb.T * detJ).T

    # ==============================================================================
    # Functions for testing element implementations
    # ==============================================================================

    def getRandParamCoord(self, n=1):
        """Generate a set of random parametric coordinates

        By default this method assumes that the valid range for all parametric coordinates is [-1, 1].
        For elements where this is not the case, this method should be reimplemented.

        Parameters
        ----------
        n : int, optional
            number of points to generate, by default 1

        Returns
        -------
        paramCoords : n x numDim array
            isoparametric coordinates, one row for each point
        """
        return np.atleast_2d(np.random.rand(n, self.numDim))

    @abc.abstractmethod
    def getRandomNodeCoords(self):
        """Generate a random, but valid, set of node coordinates for an element

        This method should be implemented for each element.

        Returns
        -------
        nodeCoords : numNode x numDim array
            Node coordinates
        """
        raise NotImplementedError

    def testGetParamCoord(self, n=10, maxIter=40, tol=1e-10):
        """Test the getParamCoord method

        This test works by generating a set of random parametric coordinates, converting them to real coordinates, and
        then checking that the parametric coordinates returned by getParamCoord match the original random values.

        Parameters
        ----------
        n : int, optional
            Number of random coordinates to generate, by default 10
        """
        paramCoord = self.getRandParamCoord(n)
        nodeCoords = self.getRandomNodeCoords()
        realCoords = self.getRealCoord(paramCoord, nodeCoords)
        error = np.zeros_like(realCoords)
        for i in range(n):
            error[i] = paramCoord[i] - self.getParamCoord(realCoords[i], nodeCoords, maxIter=maxIter, tol=tol)
        return error

    def testShapeFunctionDerivatives(self, n=10):
        """Test the implementation of the shape function derivatives using the complex-step method

        Parameters
        ----------
        n : int, optional
            Number of random coordinates to generate, by default 10
        """
        paramCoords = self.getRandParamCoord(n)
        coordPert = np.zeros_like(paramCoords, dtype="complex128")
        dN = self.getShapeFunctionDerivs(paramCoords)
        dNApprox = np.zeros_like(dN)
        for i in range(self.numDim):
            np.copyto(coordPert, paramCoords)
            coordPert[:, i] += 1e-200 * 1j
            dNApprox[:, i, :] = 1e200 * np.imag(self.getShapeFunctions(coordPert))
        return dN - dNApprox

    def testShapeFunctionSum(self, n=10):
        """Test the basic property that shape function values should sum to 1 everywhere within an element

        Parameters
        ----------
        n : int, optional
            Number of points to test at, by default 10
        """
        paramCoords = self.getRandParamCoord(n)
        N = self.getShapeFunctions(paramCoords)
        return np.sum(N, axis=1)

    # TODO: Tests to add
    # - Complex step validation of jacobian
    # - Validate stiffness matrix against resdiual (would need to implement a residual assembly method)

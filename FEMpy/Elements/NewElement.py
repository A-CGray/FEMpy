"""
==============================================================================
FEMpy - New Element Class
==============================================================================
@File    :   NewElement.py
@Date    :   2022/11/17
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
from numba import guvectorize, float64, njit
from scipy.optimize import minimize, bounds, LinearConstraint
from scipy.spatial.transform import Rotation


# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy.LinAlg import det1, det2, det3, inv1, inv2, inv3


class Element:
    """_summary_

    ## What do we need an element to do?:
    - Given nodal DOF, compute state at given parametric coordinates within the element `computeState`
    - Given nodal DOF, compute state gradients at given parametric coordinates within the element `computeStateGradients`
    - Given nodal coordinates, compute coordinates at given parametric coordinates within the element `computeCoordinates`
    - Given real coordinates, find the parametric coordinates of the closest point on the element to that point
    - Given a function that can depend on true coordinates, the state, state gradients and some design variables, compute the value of that function over the element
    - Given a function that can depend on true coordinates, the state, state gradients and some design variables, integrate that function over the element
    - Given state and design variable values, and a constitutive model, compute a residual
    - Given state and design variable values, and a constitutive model, compute a residual Jacobian
    """

    def __init__(self, numNodes, numDimensions, numStates=None, quadratureOrder=2):
        """Instantiate an element object

        _extended_summary_

        Parameters
        ----------
        numNodes : int
            Number of nodes in the element
        numDimensions : int
            Number of spatial dimensions the element lives in
        numStates : int, optional
            Number of states in the underlying PDE, a.k.a the number of DOF per node, by default uses numDimensions
        quadratureOrder : int, optional
            Integration quadrature order, by default 2
        """
        self.numNodes = numNodes
        self.numDim = numDimensions
        self.numStates = numStates if numStates is not None else numDimensions
        self.numDOF = self.numNodes * self.numStates
        self.quadOrder = quadratureOrder
        self.name = f"{self.numNodes}Node-{self.numStates}Disp-{self.numDim}D-Element"

        # --- Parametric coordinate bounds ---
        # By default it is assumed that the parametric coordinates are in the range [-1, 1] in each dimension, for
        # elements where this is not true (e.g a 2d triangular element), these attributes should be overwritten
        self.paramCoordLowerBounds = -np.ones(self.numDim)
        self.paramCoordUpperBounds = np.ones(self.numDim)
        self.paramCoordLinearConstaintMat = None
        self.paramCoordLinearConstaintUBounds = None
        self.paramCoordLinearConstaintLBounds = None

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

        if self.numDim not in [1, 2, 3]:
            raise ValueError(f"Sorry, FEMpy doesn't support {self.numDim}-dimensional problems")

    # ==============================================================================
    # Abstract methods: Must be implemented by derived classes
    # ==============================================================================

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def computeShapeFunctionGradients(self, paramCoords):
        """Compute the derivatives of the shape functions with respect to the parametric coordinates at a given set of parametric coordinates

        _extended_summary_

        Parameters
        ----------
        paramCoords : numPoint x numDim array
            Array of parametric point coordinates to evaluate shape function gradients at

        Returns
        -------
        NGrad: numPoint x numNodes x numDim array
            Shape function gradient values, NGrad[i][j][k] is the value of the kth shape function at the ith point w.r.t the kth
            parametric coordinate
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getIntegrationPointWeights(self, order):
        """Compute the integration point weights for a given quadrature order on this element

        Parameters
        ----------
        order : int
            Integration order
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getIntegrationPointCoords(self, order):
        """Compute the integration point parameteric coordinates for a given quadrature order on this element

        Parameters
        ----------
        order : int
            Integration order
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getReferenceElementCoordinates(self):
        """Get the node coordinates for the reference element, a.k.a the elementon which the shape functions are defined

        _extended_summary_

        Returns
        -------
        numNodes x numDim array
            Element node coordinates
        """
        raise NotImplementedError

    # ==============================================================================
    # Implemented methods
    # ==============================================================================
    def computeResidual(self, nodeStates, nodeCoords, designVars, constitutiveModel, intOrder=None):
        """Compute the local residual for a series of elements

        Parameters
        ----------
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element
        nodeStates : numElements x numNodes x numStates array
            State values at the nodes of each element
        dvs : numElements x numDVs array
            Design variable values for each element
        constitutiveModel : FEMpy constitutive model object
            The constitutive model of the element

        Returns
        -------
        numElement x (numNodes * numStates) array
            The local residual for each element
        """
        numElements = nodeCoords.shape[0]
        nodeCoords = np.ascontiguousarray(nodeCoords)
        # - Get integration point parametric coordinates and weights (same for all elements of same type)
        intOrder = self.defaultIntOrder if intOrder is None else intOrder
        intPointWeights = self.getIntPointWeights(intOrder)  # NumElements x numIntPoints
        intPointParamCoords = self.getPointParamCoords(intOrder)  # NumElements x numIntPoints x numDim
        numIntPoints = len(intPointWeights)

        # - Get shape functions N (du/dq) and their gradients in parametric coordinates at integration points
        # (same for all elements of same type)
        N = self.computeShapeFunctions(intPointParamCoords)  # numIntPoints x numNodes
        NPrimeParam = self.computeShapeFunctionGradients(intPointParamCoords)  # numIntPoints x numDim x numNodes

        # - Compute real coordinates at integration points (different for each element)
        self._interpolationProduct(N[:, : self.numDim], nodeCoords)  # numElements x numIntPoints x numDim

        # - Compute states at integration points (different for each element)
        self._interpolationProduct(N, nodeStates)  # numElements x numIntPoints x numStates

        # - Compute Jacobians, their inverses, and their determinants at integration points (different for each element)
        Jacs = np.zeros(numElements, numIntPoints, self.numDim, self.numDim)
        _computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jacs)
        # JacInvs = self.jacInv(Jacs)
        # JacDets = self.jacDet(Jacs)

        # - Compute du'/dq at integration points (different for each element)

        # - Compute u' at integration points (different for each element)
        # - Compute function f(x_real, dvs, u, u') at integration points (different for each constitutive model)
        # - Compute r = du'/dq^T * f
        # - Compute R, weighted sum of r * detJ over each set of integration points

    # def integrate(self, integrand, nodeCoords, uNodes, dvs, intOrder=None):
    #     """Integrate a function over a set of elements

    #     _extended_summary_

    #     Parameters
    #     ----------
    #     integrand : function
    #         function to be integrated, should have the signature integrand(x, u, u', dvs)
    #     nodeCoords : numElements x numNodes x numDim array
    #         Node coordinates for each element,
    #     uNodes : NumElements x numDOF array
    #         The nodal state values for each element
    #     dvs : NumElements x numDV array
    #         The design variable values for each element
    #     intOrder : int, optional
    #         The integration order to use, uses element default if not provided
    #     """
    #     intOrder = self.defaultIntOrder if intOrder is None else intOrder
    #     intPointWeights = self.getIntPointWeights(intOrder)  # NumElements x numIntPoints
    #     intPointParamCoords = self.getPointParamCoords(intOrder)  # NumElements x numIntPoints x numDim

    #     intPointRealCoords = self.getRealCoord(intPointParamCoords, nodeCoords)  # NumElements x numIntPoints x numDim
    #     intPointu = self.getU(intPointParamCoords, nodeCoords, uNodes)  # NumElements x numIntPoints x numStates
    #     intPointuPrime = self.getUPrime(
    #         intPointParamCoords, nodeCoords, uNodes
    #     )  # NumElements x numIntPoints x numStates x numDim
    #     integrandValues = integrand(intPointRealCoords, intPointu, intPointuPrime, dvs)
    #     intPointDetJ = self.jacDet(self.getJacobian(intPointParamCoords, nodeCoords))  # NumElements x numIntPoints

    #     # Integrate over each element by computing the sum of the integrand values from each integration point, weight
    #     # by the integration point weights and the determinant of the jacobian
    #     integratedValues = np.tensordot(integrandValues, intPointWeights * intPointDetJ, axes=([1], [0]))

    #     return integratedValues

    def computeStates(self, paramCoords, nodeStates):
        """Given nodal DOF, compute the state at given parametric coordinates within the element

        This function is vectorised both across multiple elements, and multiple points within each element,
        but the parametric coordinates are assumed to be the same across all elements

        Parameters
        ----------
        paramCoords : numPoint x numDim array
            Array of parametric point coordinates to compute state at
        nodeStates : numElements x numNodes x numStates array
            State values at the nodes of each element

        Returns
        -------
        states : numElements x numPoint x numStates array
        """

        # Compute shape functions at the given parametric coordinates
        N = self.computeShapeFunctions(paramCoords)

        # Then for each element, compute the states at the points, the einsum below is equivalent to:
        # product = np.zeros((numElements, numPoints, numStates))
        # for ii in range(numElements):
        #     product[ii] = N @ nodeStates[ii]
        return self._interpolationProduct(N, nodeStates)

    def computeCoordinates(self, paramCoords, nodeCoords):
        """Given nodal coordinates, compute the real coordinates at given parametric coordinates within the element

        Parameters
        ----------
        paramCoords : numPoint x numDim array
            Array of parametric point coordinates to compute real coordinates of
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element
        """
        # Compute shape functions at the given parametric coordinates
        N = self.computeShapeFunctions(paramCoords)

        # Then for each element, compute the states at the points, the einsum below is equivalent to:
        # product = np.zeros((numElements, numPoints, numStates))
        # for ii in range(numElements):
        #     product[ii] = N[:, : self.numNodes] @ nodeStates[ii]
        return self._interpolationProduct(N[:, : self.numNodes], nodeCoords)

    def computeJacobians(self, paramCoords, nodeCoords):
        """Compute the Jacobian at a set of parametric coordinates within a set of elements

        This function is vectorised both across multiple elements, and multiple points within each element,
        but the parametric coordinates are assumed to be the same across all elements

        Parameters
        ----------
        paramCoords : numPoint x numDim array
            Array of parametric point coordinates to compute Jacobians at
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element

        Returns
        -------
        Jac : numElements x numPoints x numDim x numDim array
            The Jacobians at each point in each element
        """
        NPrimeParam = self.computeShapeFunctionGradients(paramCoords)  # numPoints x numDim x numNodes
        numElements = nodeCoords.shape[0]
        numPoints = paramCoords.shape[0]
        Jac = np.zeros((numElements, numPoints, self.numDim, self.numDim))
        nodeCoords = np.ascontiguousarray(nodeCoords)

        # The function call below does the following:
        # for ii in range(numElements):
        #   for jj in range(numPoints):
        #     Jac[ii, jj] = NPrimeParam[jj] @ nodeCoords[ii]
        _computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jac)
        return Jac

    def computeStateGradients(self, NPrimeParam, nodeStates, nodeCoords):
        """Given nodal DOF, compute the gradient of the state at given parametric coordinates within the element

        The gradient of the state at each point in each element is a numStates x numDim array.

        This function is vectorised both across multiple elements, and multiple points within each element,
        but the parametric coordinates are assumed to be the same across all elements

        Parameters
        ----------
        paramCoords : numPoint x numDim array
            Array of parametric point coordinates to compute state at
        nodeStates : numElements x numNodes x numStates array
            State values at the nodes of each element
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element

        Returns
        -------
        stateGradients : numElements x numPoint x numStates x numDim array
        """
        # J = NPrimeParam * nodeCoords
        # u' = J^-1 * NPrimeParam * q

        numElements = nodeCoords.shape[0]
        numPoints = NPrimeParam.shape[0]
        Jac = np.zeros((numElements, numPoints, self.numDim, self.numDim))
        _computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jac)
        JacInv = self.jacInv(np.reshape(Jac, (numElements * numPoints, self.numDim, self.numDim)))
        JacInv = np.reshape(JacInv, (numElements, numPoints, self.numDim, self.numDim))
        UPrime = np.zeros((numElements, numPoints, self.numStates, self.numDim))
        # The function call below is equivalent to the following
        # for ii in range(numElements):
        #     for jj in range(numPoints):
        #         result[ii, jj] = (JacInv[ii, jj] @ NPrimeParam[jj] @ nodeStates[ii]).T
        _computeUPrimeProduct(JacInv, NPrimeParam, np.ascontiguousarray(nodeStates), UPrime)
        return UPrime

    # Given a function that can depend on true coordinates, the state, state gradients and some design variables, compute the value of that function over the element

    # - Given node coordinates and states, design variable values, and a constitutive model, compute a residual Jacobian
    def computeJacobian(self, nodeCoords, nodeStates, dvs, constitutiveModel):
        """Compute the local residual Jacobian (dR/dq) for a series of elements

        Parameters
        ----------
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element
        nodeStates : numElements x numNodes x numStates array
            State values at the nodes of each element
        dvs : numElements x numDVs array
            Design variable values for each element
        constitutiveModel : FEMpy constitutive model object
            The constitutive model of the element

        Returns
        -------
        numElement x (numNodes * numStates) x (numNodes * numStates) array
            The local jacobian for each element
        """
        return None

    def getClosestPoints(self, nodeCoords, point):
        """Given real coordinates of a point, find the parametric coordinates of the closest point on a series of
        elements to that point

        Computing the closest point is an optimization problem of the form:

        min ||X(x) - P||^2

        s.t Ax <= b
            lb <= x <= ub

        Where X are the real coordinates of a point in the element, x the parametric coordinates of that point, and P is
        the target point. lb <= x <= ub and Ax <= b are a set of bounds and linear constraints on the parametric
        coordinates that encode the bounds of the element.

        Parameters
        ----------
        nodeCoords : numElements x numNodes x numDim array
            The coordinates of the elements
        point : array of length numDim
            Target point coordinates

        Returns
        -------
        closestParamCoords : numElements x numDim array
            The parametric coordinates of the closest point in each element
        closestDistances : numElements array
            The distances from the closest point in each element to the target point
        """
        numElements = nodeCoords.shape[0]
        closestDistances = np.zeros(numElements)
        closestParamCoords = np.zeros((numElements, self.numDim))

        paramCoordBounds = bounds(lb=self.paramCoordLowerBounds, ub=self.paramCoordUpperBounds)
        if self.paramCoordLinearConstaintMatrix is not None:
            paramCoordLinearConstraints = LinearConstraint(
                self.paramCoordLinearConstaintMat,
                self.paramCoordLinearConstaintLowerBounds,
                self.paramCoordLinearConstaintUpperBounds,
                keep_feasible=True,
            )
        else:
            paramCoordLinearConstraints = None

        for ii in range(numElements):
            closestParamCoords[ii], closestDistances[ii] = self._getClosestPoint(
                nodeCoords[ii], point, paramCoordBounds, paramCoordLinearConstraints
            )

        return closestParamCoords, closestDistances

    def _getClosestPoint(self, nodeCoords, point, paramCoordBounds, paramCoordLinearConstraints):
        nodeCoordCopy = np.zeros((1, self.numNodes, self.numDim))
        nodeCoordCopy[0] = nodeCoords

        def r(xParam):
            xTrue = self.computeCoordinates(np.atleast_2d(xParam), nodeCoordCopy)
            return np.linalg.norm(xTrue.flatten() - point)

        def drdxParam(xParam):
            xTrue = self.computeCoordinates(np.atleast_2d(xParam), nodeCoordCopy)
            Jac = self.computeJacobians(np.atleast_2d(xParam), nodeCoordCopy)
            return 2 * (xTrue.flatten() - point) @ Jac[0, 0].T

        sol = minimize(
            r,
            np.zeros(self.numDim),
            jac=drdxParam,
            bounds=paramCoordBounds,
            constraints=paramCoordLinearConstraints,
            method="SLSQP",
            tol=1e-10,
        )

        return sol.x, sol.fun

    def getRandomElementCoordinates(self):
        """Compute random node coordinates for an element

        The random node coordinates are computed by taking the reference element coordinates and then applying:
        - Random perturbations to each node
        - Random translation in each dimension
        - Random scalings in each dimension
        - Random rotations around each available axis
        """
        rng = np.random.default_rng()
        coords = self.getReferenceElementCoordinates()  # numNodes x numDim array

        # Perturb coordinates by up to 10% of the maximum distance between any two nodes
        maxDistance, _ = _computeMaxMinDistance(coords)
        coords += rng.random(coords.shape) * 0.1 * maxDistance

        # Scale each dimension by a random factor between 0.1 and 10
        for dim in range(self.numDim):
            scalingPower = rng.random() * 2 - 1
            coords[:, dim] *= 10**scalingPower

        # Rotate the element around each axis by a random angle
        if self.numDim == 2:
            R = Rotation.from_rotvec(np.array([0, 0, 1]) * rng.random() * 2 * np.pi)
            coords = R.as_matrix()[:2, :2] @ coords

    @staticmethod
    def _interpolationProduct(N, nodeValues):
        """Compute the product of the interpolation matrix and a set of node values

        Parameters
        ----------
        N : numPoints x numNodes array
            Shape function values at each point
        nodeValues : numElements x numNodes x numValues array
            Nodal values for each element

        Returns
        -------
        numElements x numPoints x numStates array
            Interpolated values for each element
        """
        # the einsum below is equivalent to:
        # product = np.zeros((numElements, numPoints, numStates))
        # for ii in range(numElements):
        #     product[ii] = N @ nodeStates[ii]
        return np.einsum("pn,ens->eps", N, nodeValues)


@guvectorize(
    [(float64[:, :, ::1], float64[:, :, ::1], float64[:, :, :, ::1])],
    "(p,d,n),(e,n,d)->(e,p,d,d)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jac):
    """This function computes a nasty product of two 3d arrays that is required when computing element mapping Jacobians

    Given the shape function derivatives at each point NPrimeParam (a numPoints x numDim x numNodes array), and the node
    coordinates at each element (a numElements x numNodes x numDim array), we want to compute:

    `NPrimeParam[jj] @ nodeCoords[ii]`

    For each element ii and parametric point jj, this function does this, but is vectorized by numba to make it fast.

    NOTE: It's very important to make sure that the arrays passed into this function are memory contiguous,
    if you're not sure if they are then run numpy's `ascontiguousarray` function on them before passing them in

    Parameters
    ----------
    NPrimeParam : numPoints x numDim x numNodes array
        Shape function gradient values, NPrimeParam[i][j][k] is the derivative of the kth shape function at the ith point w.r.t the jth
            parametric coordinate
    nodeCoords : numElements x numNodes x numDim array
        Node coordinates for each element
    Jac : numElements x numPoints x numDim x numDim array
        The Jacobians at each point in each element
    """
    numElements = Jac.shape[0]
    numPoints = Jac.shape[1]
    for ii in range(numElements):
        for jj in range(numPoints):
            Jac[ii, jj] = NPrimeParam[jj] @ nodeCoords[ii]


@guvectorize(
    [(float64[:, :, :, ::1], float64[:, :, ::1], float64[:, :, ::1], float64[:, :, :, ::1])],
    "(e,p,d,d),(p,d,n),(e,n,s)->(e,p,s,d)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _computeUPrimeProduct(JacInv, NPrimeParam, nodeStates, result):
    """This function computes a nasty product of 3 and 4d arrays that is required when computing state gradients at multiple points within multiple elements

    Given the shape function derivatives in the parametric coordinates at each point, NPrimeParam (a numPoints x numDim x numNodes array), the inverses of the element mapping Jacobians at each point in each element, JacInv (a numElements x numPoints x numDim x numDim array) and the nodal state values for each element, nodeStates (a numElement x numNodes x numStates), we want to compute:

    `UPrime[ii, jj] = (JacInv[ii, jj] @ NPrimeParam[jj] @ nodeStates[ii]).T`

    For each element ii and parametric point jj, this function does this, but is vectorized by numba to make it fast.

    Parameters
    ----------
    JacInv : numElements x numPoints x numDim x numDim
        Inverse element mapping Jacobians at each point in each element
    NPrimeParam : numPoints x numDim x numNodes array
        Shape function gradients in the parametric coordinates at each point
    nodeStates : numElements x numNodes x numStates array
        Nodal state values for each element
    result : numElements x numPoints x numStates x numDim array
        The state gradient at each point in each element
    """
    numElements = JacInv.shape[0]
    numPoints = JacInv.shape[1]
    for ii in range(numElements):
        for jj in range(numPoints):
            result[ii, jj] = (JacInv[ii, jj] @ NPrimeParam[jj] @ nodeStates[ii]).T


@njit(cache=True, fastmath=True, boundscheck=False)
def _computeMaxMinDistance(coords):
    """Compute the maximum distance between any two points in a set of coordinates

    Parameters
    ----------
    coords : numPoints x numDim array
        The coordinates to compute the distances between

    Returns
    -------
    maxDistance : float
        The maximum distance between any two points in the set of coordinates
    minDistance : float
        The minimum distance between any two points in the set of coordinates
    """
    numPoints = coords.shape[0]
    maxDistance = 0.0
    minDistance = np.inf
    for ii in range(numPoints):
        for jj in range(ii + 1, numPoints):
            distance = np.linalg.norm(coords[ii] - coords[jj])
            maxDistance = max(maxDistance, distance)
            maxDistance = max(maxDistance, distance)
    return maxDistance, minDistance

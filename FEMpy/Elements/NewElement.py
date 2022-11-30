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
from scipy.optimize import minimize, Bounds, LinearConstraint
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
        NPrimeParam: numPoint x numDim x numNodes array
            Shape function gradient values, NPrimeParam[i][j][k] is the derivative of the kth shape function at the ith point w.r.t the jth parametric coordinate
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getIntegrationPointWeights(self, order=None):
        """Compute the integration point weights for a given quadrature order on this element

        Parameters
        ----------
        order : int, optional
            Integration order

        Returns
        -------
        array of length numIntpoint
            Integration point weights
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getIntegrationPointCoords(self, order=None):
        """Compute the integration point parameteric coordinates for a given quadrature order on this element

        Parameters
        ----------
        order : int, optional
            Integration order

        Returns
        -------
        numIntpoint x numDim array
            Integration point coordinates
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getReferenceElementCoordinates(self):
        """Get the node coordinates for the reference element, a.k.a the element on which the shape functions are defined

        Returns
        -------
        numNodes x numDim array
            Element node coordinates
        """
        raise NotImplementedError

    # ==============================================================================
    # Implemented methods
    # ==============================================================================
    def computeFunction(self, nodeCoords, nodeStates, elementDVs, function, elementReductionType):
        """Given a function that can depend on true coordinates, the state, state gradients and some design variables, compute the value of that function over the element

        _extended_summary_

        Parameters
        ----------
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element
        nodeStates : numElements x numNodes x numStates array
            State values at the nodes of each element
        dvs : numElements x numDVs array
            Design variable values for each element
        function : callable
            Function to evaluate at each point within each element, must have signature f(x, u, u', dvs), where:
                x is an n x numDim array of coordinates
                u is an n x numStates array of state values
                u' is an n x (numStates*numDim) array of state gradients
                dvs is an n x numDVs array of design variable values
        elementReductionType : _type_
            Type of reduction to do to get a single value for each element, can be:
                - 'sum' : sum all values
                - 'mean' : average all values
                - `integrate` : integrate the function over the element
                - 'max' : take the maximum value
                - 'min' : take the minimum value
                - 'ksmax' : Compute a smooth approximation of the maximum value using KS aggregation
                - 'ksmin' : Compute a smooth approximation of the minimum value using KS aggregation

        Returns
        -------
        values : numElements array
            Value of the function for each element
        """
        numElements = nodeCoords.shape[0]
        return np.random.rand(numElements)

    def computeResiduals(self, nodeStates, nodeCoords, designVars, constitutiveModel, intOrder=None):
        """Compute the local residual for a series of elements

        Parameters
        ----------
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element
        nodeStates : numElements x numNodes x numStates array
            State values at the nodes of each element
        designVars : dict of numElements arrays
            Design variable values for each element
        constitutiveModel : FEMpy constitutive model object
            The constitutive model of the element

        Returns
        -------
        numElement x numNodes x numStates array
            The local residuals for each element
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
        intPointCoords = self._interpolationProduct(
            N[:, : self.numDim], nodeCoords
        )  # numElements x numIntPoints x numDim

        # - Compute states at integration points (different for each element)
        intPointStates = self._interpolationProduct(N, nodeStates)  # numElements x numIntPoints x numStates

        # - Compute Jacobians, their inverses, and their determinants at integration points (different for each element)
        Jacs = np.zeros(numElements, numIntPoints, self.numDim, self.numDim)
        _computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jacs)
        JacInvs = self.jacInv(Jacs)
        JacDets = self.jacDet(Jacs)  # numElements x numIntPoints

        # - Compute du'/dq at integration points (different for each element)
        dUPrimedq = np.zeros(numElements, numIntPoints, self.numDim, self.numNodes)
        _computeDUPrimeDqProduct(JacInvs, NPrimeParam, dUPrimedq)

        # - Compute u' at integration points (different for each element)
        intPointStateGradients = np.zeros(numElements, numIntPoints, self.numStates, self.numDim)
        _computeUPrimeProduct(JacInvs, NPrimeParam, nodeStates, intPointStateGradients)

        # - Compute function f(x_real, dvs, u, u') at integration points (different for each constitutive model)
        # First, currently everything is in numElements x numIntPoints x ... arrays, but the constitutive model doesn't care about the distinction between different elements, so we need to flatten the first two dimensions
        numPointsTotal = numElements * numIntPoints
        intPointCoords = np.ascontiguousarray(np.reshape(intPointCoords, (numPointsTotal, self.numDim)))
        intPointStates = np.ascontiguousarray(np.reshape(intPointStates, (numPointsTotal, self.numStates)))
        intPointStateGradients = np.ascontiguousarray(
            np.reshape(intPointStateGradients, (numPointsTotal, self.numStates, self.numDim))
        )

        # For the DVs it's a bit different, we have one DV value per element, so we actually need to expand them so that we have one value per integration point
        intPointDVs = {}
        for dvName, dvValues in designVars.items():
            intPointDVs[dvName] = np.repeat(dvValues, numIntPoints)

        weakRes = constitutiveModel.computeWeakResiduals(
            intPointStates, intPointStateGradients, intPointCoords, intPointDVs
        )

        # Reshape the weak residuals back to numElements x numIntPoints x ...
        weakRes = np.ascontiguousarray(np.reshape(weakRes, (numElements, numIntPoints, self.numStates, self.numDim)))

        # - Compute r = du'/dq^T * f
        r = np.zeros(numElements, numIntPoints, self.numNodes, self.numStates)
        _transformResidual(dUPrimedq, weakRes, r)

        # - Compute R, weighted sum of w * r * detJ over each set of integration points
        R = np.einsum("epns,ep,p->ens", r, JacDets, intPointWeights)
        return R
        # R = np.tensordot(r * JacDets, intPointWeights, axes=([1], [0]))

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

    def computeStateGradients(self, paramCoords, nodeStates, nodeCoords):
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
        numPoints = paramCoords.shape[0]
        NPrimeParam = self.computeShapeFunctionGradients(paramCoords)
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

    def getClosestPoints(self, nodeCoords, point, **kwargs):
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

        paramCoordBounds = Bounds(lb=self.paramCoordLowerBounds, ub=self.paramCoordUpperBounds)
        if self.paramCoordLinearConstaintMat is not None:
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
                nodeCoords[ii], point, paramCoordBounds, paramCoordLinearConstraints, **kwargs
            )

        return closestParamCoords, closestDistances

    def _getClosestPoint(self, nodeCoords, point, paramCoordBounds, paramCoordLinearConstraints, **kwargs):
        nodeCoordCopy = np.zeros((1, self.numNodes, self.numDim))
        nodeCoordCopy[0] = nodeCoords

        def r(xParam):
            xTrue = self.computeCoordinates(np.atleast_2d(xParam), nodeCoordCopy).flatten()
            print(f"{xParam=}, {xTrue=}, {point=}")
            return np.linalg.norm(xTrue - point)

        def drdxParam(xParam):
            xTrue = self.computeCoordinates(np.atleast_2d(xParam), nodeCoordCopy).flatten()
            Jac = self.computeJacobians(np.atleast_2d(xParam), nodeCoordCopy)
            return 2 * (xTrue - point) @ Jac[0, 0].T

        if "tol" not in kwargs:
            kwargs["tol"] = 1e-10

        maxAttempts = 10
        closestPointFound = False
        for _ in range(maxAttempts):
            sol = minimize(
                r,
                np.random.rand(self.numDim),
                jac=drdxParam,
                bounds=paramCoordBounds,
                constraints=paramCoordLinearConstraints,
                method="trust-constr",
                **kwargs,
            )
            closestPointFound = sol.success
            if closestPointFound:
                break

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

        # Apply random translation
        for ii in range(self.numDim):
            translation = rng.random() * 2 * maxDistance - maxDistance
            coords[:, ii] += translation

        # Scale each dimension by a random factor between 0.1 and 10
        for dim in range(self.numDim):
            scalingPower = rng.random() * 2 - 1
            coords[:, dim] *= 10**scalingPower

        # Rotate the element around each axis by a random angle
        if self.numDim == 2:
            angle = rng.random() * 2 * np.pi
            print(f"{angle=}")
            c, s = np.cos(angle), np.sin(angle)
            R = np.array(((c, s), (-s, c)))
            coords = coords @ R.T
        elif self.numDim == 3:
            R = Rotation.random(random_state=rng)
            coords = coords @ R.as_matrix().T

        return coords

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

    # ==============================================================================
    # Testing methods
    # ==============================================================================

    def getRandParamCoord(self, n):
        """Get a random set of parametric coordinates within the element

        By default this method assumes the the valid parametric coordinates are between -1 and 1 in each direction. If this is not the case for a particular element then that element should reimplemnt this method.

        Parameters
        ----------
        n : int
            Number of points to generate
        """
        return np.random.rand(n, self.numDim) * 2 - 1

    def testShapeFunctionDerivatives(self, n=10):
        """Test the implementation of the shape function derivatives using the complex-step method

        Parameters
        ----------
        n : int, optional
            Number of random coordinates to generate, by default 10
        """
        paramCoords = self.getRandParamCoord(n)
        coordPert = np.zeros_like(paramCoords, dtype="complex128")
        dN = self.computeShapeFunctionGradients(paramCoords)
        dNApprox = np.zeros_like(dN)
        for i in range(self.numDim):
            np.copyto(coordPert, paramCoords)
            coordPert[:, i] += 1e-200 * 1j
            dNApprox[:, i, :] = 1e200 * np.imag(self.computeShapeFunctions(coordPert))
        return dN - dNApprox

    def testShapeFunctionSum(self, n=10):
        """Test the basic property that shape function values should sum to 1 everywhere within an element

        Parameters
        ----------
        n : int, optional
            Number of points to test at, by default 10
        """
        paramCoords = self.getRandParamCoord(n)
        N = self.computeShapeFunctions(paramCoords)
        return np.sum(N, axis=1)

    def testIdentityJacobian(self, n=10):
        """Validate that, when the element geometry matches the reference element exactly, the mapping Jacobian is the identity matrix everywhere.

        Parameters
        ----------
        n : int, optional
            Number of points to test at, by default 10
        """
        nodeCoords = np.zeros((1, self.numNodes, self.numDim))
        nodeCoords[0] = self.getReferenceElementCoordinates()
        paramCoords = self.getRandParamCoord(n)

        # The expected Jacobians are a stack of n identity matrices
        expectedJacs = np.tile(np.eye(self.numDim), (1, n, 1, 1))
        Jacs = self.computeJacobians(paramCoords, nodeCoords)
        return Jacs - expectedJacs

    def testStateGradient(self, n=10):
        """Test that the state gradient is correctly reconstructed within the element

        This test works by generating random node coordinates, then computing the states at each node using the
        following equation:

        u_i = a_i * x + b_i * y + c_i * z + d_i

        This field has a gradient, du/dx, of [a_i, b_i, c_i] everywhere in the element, which should be exactly reproduced by
        the state gradient computed by the element.

        Parameters
        ----------
        n : int, optional
            _description_, by default 10
        """
        nodeCoords = np.zeros((1, self.numNodes, self.numDim))
        nodeCoords[0] = self.getRandomElementCoordinates()
        paramCoords = self.getRandParamCoord(n)

        randStateGradient = np.random.rand(self.numStates, self.numDim)
        ExpectedStateGradients = np.tile(
            randStateGradient, (1, n, 1, 1)
        )  # np.ones((1, n, self.numStates, self.numDim))

        nodeStates = np.zeros((1, self.numNodes, self.numStates))
        for ii in range(self.numNodes):
            for jj in range(self.numStates):
                nodeStates[:, ii, jj] = np.dot(nodeCoords[0, ii], randStateGradient[jj])

        stateGradient = self.computeStateGradients(paramCoords, nodeStates, nodeCoords)

        return stateGradient - ExpectedStateGradients

    def testGetClosestPoints(self, n=10, tol=1e-10):
        """Test the getClosestPoints method

        This test works by generating a set of random parametric coordinates, converting them to real coordinates, and
        then checking that the parametric coordinates returned by getClosestPoints match the original random values.

        Parameters
        ----------
        n : int, optional
            Number of random coordinates to generate, by default 10
        """
        nodeCoords = np.zeros((1, self.numNodes, self.numDim))
        nodeCoords[0] = self.getRandomElementCoordinates()
        paramCoords = self.getRandParamCoord(n)
        realCoords = self.computeCoordinates(paramCoords, nodeCoords)
        error = np.zeros_like(realCoords)
        for i in range(n):
            coords, _ = self.getClosestPoints(nodeCoords, realCoords[0, i], tol=tol)
            error[0, i] = coords - paramCoords[i]
        return error


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
    """Compute the nasty product of 3 and 4d arrays required when computing state gradients at multiple points within multiple elements

    Given the shape function derivatives in the parametric coordinates at each point, NPrimeParam (a numPoints x numDim
    x numNodes array), the inverses of the element mapping Jacobians at each point in each element, JacInv
    (a numElements x numPoints x numDim x numDim array) and the nodal state values for each element, nodeStates
    (a numElement x numNodes x numStates), we want to compute:

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


@guvectorize(
    [(float64[:, :, :, ::1], float64[:, :, ::1], float64[:, :, :, ::1])],
    "(e,p,d,d),(p,d,n)->(e,p,d,n)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _computeDUPrimeDqProduct(JacInv, NPrimeParam, result):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    JacInv : numElements x numPoints x numDim x numDim
        Inverse element mapping Jacobians at each point in each element
    NPrimeParam : numPoints x numDim x numNodes array
        Shape function gradients in the parametric coordinates at each point
    result : numElements x numPoints x numDim x numNodes array
        result[i, j, k, l] contains the sensitvity of du/dx_k at the jth point in the ith element to the state at the lth node
    """
    numElements = JacInv.shape[0]
    numPoints = JacInv.shape[1]
    for ii in range(numElements):
        for jj in range(numPoints):
            result[ii, jj] = JacInv[ii, jj] @ NPrimeParam[jj]


@guvectorize(
    [(float64[:, :, :, ::1], float64[:, :, :, ::1], float64[:, :, :, ::1])],
    "(e,p,d,n),(e,p,d,s)->(e,p,n,s)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def _transformResidual(dUPrimedq, weakRes, result):
    """Compute a nasty product of high dimensional arrays to compute integration point residuals

    The weak residual computed by the constitutive model is essentially the derivative of the energy at each point with
    respect to the state gradients, this function transforms this to the derivative of the energy with respect to the
    nodal DOF, which is what we need to compute the element residual:

    r = dE/dq^T = dU'/dq^T @ dE/dU'^T

    Parameters
    ----------
    dUPrimedq : numElements x numPoints x numDim x numNodes array
        Sensitivity of the state gradients to nodal DOFs
    weakRes : numElements x numPoints x numStates x numDim array
        Weak residual values
    result : numElements x numPoints x numNodes x numStates array
        Integration point residual contributions
    """
    numElements = dUPrimedq.shape[0]
    numPoints = dUPrimedq.shape[1]

    for ii in range(numElements):
        for jj in range(numPoints):
            result[ii, jj] = dUPrimedq[ii, jj].T @ weakRes[ii, jj]


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

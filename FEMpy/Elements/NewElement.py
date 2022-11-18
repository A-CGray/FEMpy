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
from numba import guvectorize, float64


# ==============================================================================
# Extension modules
# ==============================================================================
from FEMpy.LinAlg import det1, det2, det3, inv1, inv2, inv3


class Element:
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

    # ==============================================================================
    # Implemented methods
    # ==============================================================================
    def integrate(self, integrand, nodeCoords, uNodes, dvs, intOrder=None):
        """Integrate a function over a set of elements

        _extended_summary_

        Parameters
        ----------
        integrand : function
            function to be integrated, should have the signature integrand(x, u, u', dvs)
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element,
        uNodes : NumElements x numDOF array
            The nodal state values for each element
        dvs : NumElements x numDV array
            The design variable values for each element
        intOrder : int, optional
            The integration order to use, uses element default if not provided
        """
        intOrder = self.defaultIntOrder if intOrder is None else intOrder
        intPointWeights = self.getIntPointWeights(intOrder)  # NumElements x numIntPoints
        intPointParamCoords = self.getPointParamCoords(intOrder)  # NumElements x numIntPoints x numDim

        intPointRealCoords = self.getRealCoord(intPointParamCoords, nodeCoords)  # NumElements x numIntPoints x numDim
        intPointu = self.getU(intPointParamCoords, nodeCoords, uNodes)  # NumElements x numIntPoints x numStates
        intPointuPrime = self.getUPrime(
            intPointParamCoords, nodeCoords, uNodes
        )  # NumElements x numIntPoints x numStates x numDim
        integrandValues = integrand(intPointRealCoords, intPointu, intPointuPrime, dvs)
        intPointDetJ = self.jacDet(self.getJacobian(intPointParamCoords, nodeCoords))  # NumElements x numIntPoints

        # Integrate over each element by computing the sum of the integrand values from each integration point, weight
        # by the integration point weights and the determinant of the jacobian
        integratedValues = np.tensordot(integrandValues, intPointWeights * intPointDetJ, axes=([1], [0]))

        return integratedValues

    def computeState(self, paramCoords, nodeStates):
        """Given nodal DOF, compute the state at given parametric coordinates within the element

        This function is vectorised both across multiple elements, and multiple points within each element,
        but the parametric coordinates are assumed to be the same across all elements

        Parameters
        ----------
        paramCoords : numPoint x numState array
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
        return np.einsum("pn,ens->eps", N, nodeStates)

    def computeJacobian(self, paramCoords, nodeCoords):
        """Compute the Jacobian at a set of parametric coordinates within a set of elements

        This function is vectorised both across multiple elements, and multiple points within each element,
        but the parametric coordinates are assumed to be the same across all elements

        Parameters
        ----------
        paramCoords : numPoint x numState array
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
        computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jac)
        return Jac

    def computeStateGradient(self, paramCoords, nodeStates, nodeCoords):
        """Given nodal DOF, compute the gradient of the state at given parametric coordinates within the element

        This function is vectorised both across multiple elements, and multiple points within each element,
        but the parametric coordinates are assumed to be the same across all elements

        Parameters
        ----------
        paramCoords : numPoint x numState array
            Array of parametric point coordinates to compute state at
        nodeStates : numElements x numNodes x numStates array
            State values at the nodes of each element
        nodeCoords : numElements x numNodes x numDim array
            Node coordinates for each element

        Returns
        -------
        stateGradients : numElements x numPoint x numStates x numDim array
        """


@guvectorize(
    [(float64[:, :, ::1], float64[:, :, ::1], float64[:, :, :, ::1])],
    "(p,d,n),(e,n,d)->(e,p,d,d)",
    nopython=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def computeNPrimeCoordProduct(NPrimeParam, nodeCoords, Jac):
    """This function computes a nasty product of two 3d arrays that is required when computing element mapping Jacobians

    Given the shape function derivatives at each point NPrimeParam (a numPoints x numDim x numNodes array), and the node
    coordinates at each element (a numElements x numNodes x numDim array), we want to compute:
    NPrimeParam[jj] x nodeCoords[ii]
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

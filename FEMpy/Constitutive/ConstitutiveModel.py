"""
==============================================================================
FEMpy Constitutive model base class
==============================================================================
@File    :   NewConstitutive.py
@Date    :   2022/11/28
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
from numba import njit, prange

# ==============================================================================
# Extension modules
# ==============================================================================


class ConstitutiveModel:
    """The base class for all FEMpy constitutive models

    The constitutive model defines the underlying PDE being solved. Currently, this base class is defined for
    solid mechanics problems, but in future it may be extended for other PDE types.

    It contains information on:

    - The number of spatial dimensions the model is valid for
    - The number and names of the PDE states
    - The number and names of the stresses and strains for this model
    - The number and names of the design variables associated with the PDE
    - The names of functions which can be computed for this constitutive model (e.g mass, Von Mises stress etc)

    And contains methods to:

    - Given the coordinates, state value, state gradient, and design variables at a point, compute:
        - The strain components
        - The sensitivities of the strain components
        - The stress components
        - The sensitivities of the stress components
        - The pointwise mass
        - The volume integral scaling parameter (e.g thickness for 2D plane models or $2 \pi r$ for 2D axisymmetric problems)
        - The weak form residual
        - The weak form residual Jacobian
        - Other arbitrary output values (e.g failure criterion)
    """

    def __init__(self, numDim, stateNames, strainNames, stressNames, designVars, functionNames, linear=True) -> None:
        """_summary_



        Parameters
        ----------
        numDim : int
            Number of spatial dimensions the model is valid for
        stateNames : list of str
            Names for each state variable
        strainNames : list of str
            Names for each strain component
        stressNames : list of str
            Names for each stress component
        designVars : dict
            A nested dictionary of design variables, with the key being the name of the design variable and the value
            being a dictionary that contains various pieces of information about that DV, including:
                - "defaultValue" : The default value of that DV
        functionNames : list of str
            The names of functions that can be computed with this constitutive model
        linear : bool, optional
            Whether the constitutive model is linear or not, a.k.a whether the weak residual is a linear function of the states/state gradients, by default True
        """
        self.numDim = numDim

        self.stateNames = stateNames
        self.numStates = len(stateNames)

        self.strainNames = strainNames
        self.numStrains = len(strainNames)
        if len(stressNames) != self.numStrains:
            raise ValueError("Number of strains must equal number of stresses")
        self.stressNames = stressNames

        self.designVariables = designVars
        self.numDesignVariables = len(designVars)

        self.functionNames = functionNames

        self.lowerCaseFuncNames = [func.lower() for func in self.functionNames]

        self.isLinear = linear

    # ==============================================================================
    # Abstract methods: To be implemented by derived classes
    # ==============================================================================
    @abc.abstractmethod
    def computeStrains(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the strains at each one



        Parameters
        ----------
        states : numPoints x numStates array
            State values at each point
        stateGradients : numPoints x numStates x numDim array
            State gradients at each point
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : dictionary of len(numpoints) array
            design variable values at each point

        Returns
        -------
        numPoints x numStrains array
            Strain components at each point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def computeStrainStateGradSens(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the
        sensitivity of the strains to the state gradient at each one



        Parameters
        ----------
        states : numPoints x numStates array
            State values at each point
        stateGradients : numPoints x numStates x numDim array
            State gradients at each point
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : dictionary of len(numpoints) array
            design variable values at each point

        Returns
        -------
        numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
        """
        raise NotImplementedError

    @abc.abstractmethod
    def computeStresses(self, strains, dvs):
        """Given the strains and design variables at a bunch of points, compute the stresses at each one



        Parameters
        ----------
        strains : numPoints x numStrains array
            Strain components at each point
        dvs : dictionary of len(numpoints) array
            design variable values at each point

        Returns
        -------
        numPoints x numStresses array
            Stress components at each point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def computeStressStrainSens(self, strains, dvs):
        """Given the strains and design variables at a bunch of points, compute the sensitivity of the stresses to the strains at each one



        Parameters
        ----------
        strains : numPoints x numStrains array
            Strain components at each point
        dvs : dictionary of len(numpoints) array
            design variable values at each point

        Returns
        -------
        sens : numPoints x numStrains x numStates x numDim array
            Strain sensitivities, sens[i,j,k,l] is the sensitivity of strain component j at point i to state gradient du_k/dx_l
        """
        raise NotImplementedError

    @abc.abstractmethod
    def computeVolumeScaling(self, coords, dvs):
        """Given the coordinates and design variables at a bunch of points, compute the volume scaling parameter at each one

        The volume scaling parameter is used to scale functions that are integrated over the element to get a true
        volume integral. For example, in a 2D plane stress model, we need to multiply by the thickness of the element
        to get a true volume integral. In a 2D axisymmetric model, we need to multiply by 2*pi*r to get a true volume
        integral.

        Parameters
        ----------
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : dictionary of len(numpoints) array
            design variable values at each point

        Returns
        -------
        numPoints length array
            Volume scaling parameter at each point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def getFunction(self, name):
        """Return a function that can be computed for this constitutive model

        Parameters
        ----------
        name : str
            Name of the function to compute

        Returns
        -------
        callable
            A function that can be called to compute the desired function at a bunch of points with the signature,
            `f(states, stateGradients, coords, dvs)`, where:

            - states is a numPoints x numStates array
            - stateGradients is a numPoints x numStates x numDim array
            - coords is a numPoints x numDim array
            - dvs is a dictionary of numPoints length arrays
        """

        raise NotImplementedError

    # ==============================================================================
    # Public methods
    # ==============================================================================
    def computeWeakResiduals(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the weak residual integrand

        For a solid mechanics problem, the weak residual, derived from the virtual work principle is:

        $R = \int r dV = \int (du'/dq)^T  (d\epsilon/du')^T  \sigma  \\theta d\Omega$

        Where:

        - $du'/dq$ is the sensitivity of the state gradient to the nodal state values, this is handled by the element
        - $d\epsilon/du'$ is the sensitivity of the strain to the state gradient
        - $\sigma$ are the stresses
        - $\\theta$ is the volume scaling parameter
        - $\Omega$ is the element

        This function computes $(de/du')^T * \sigma * \\theta$ at each point

        Parameters
        ----------
        states : numPoints x numStates array
            State values at each point
        stateGradients : numPoints x numStates x numDim array
            State gradients at each point
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : dictionary of len(numpoints) array
            design variable values at each point

        Returns
        -------
        residuals : numPoints x self.numDim x self.numStates array
            Weak residual integrand at each point
        """
        numPoints = coords.shape[0]
        strain = self.computeStrains(states, stateGradients, coords, dvs)
        stress = self.computeStresses(strain, dvs)
        scale = self.computeVolumeScaling(coords, dvs)

        strainSens = self.computeStrainStateGradSens(states, stateGradients, coords, dvs)

        residuals = np.zeros((numPoints, self.numDim, self.numStates))
        _computeWeakResidualProduct(strainSens, stress, scale, residuals)
        return residuals

        # return np.einsum("pesd,pe,p->pds", strainSens, stress, scale, optimize=["einsum_path", (0, 1), (0, 1)])

    def computeWeakResidualJacobian(self, states, stateGradients, coords, dvs):
        """Given the coordinates, state value, state gradient, and design variables at a bunch of points, compute the
        weak residual jacobian integrand

        $j = (d\epsilon/du')^T \\times d\sigma/d\epsilon \\times d\epsilon/du'$

        Where:

        - $d\epsilon/du'$ is the sensitivity of the strain to the state gradient
        - $d\sigma/d\epsilon$ the sensitivity of the stress to the strain gradient

        This function computes `de/du'^T * sigma * scale` at each point

        Parameters
        ----------
        states : numPoints x numStates array
            State values at each point
        stateGradients : numPoints x numStates x numDim array
            State gradients at each point
        coords : numPoints x numDim array
            Coordinates of each point
        dvs : dictionary of len(numpoints) array
            design variable values at each point

        Returns
        -------
        Jacobians : numPoints x numDim x numStates x numStates x numDim array
            The sensibility of the weak residual integrand components to the state gradients at each point
        """
        strainSens = self.computeStrainStateGradSens(states, stateGradients, coords, dvs)
        strain = self.computeStrains(states, stateGradients, coords, dvs)
        stressSens = self.computeStressStrainSens(strain, dvs)
        scale = self.computeVolumeScaling(coords, dvs)
        numPoints = states.shape[0]
        # strainSens = strainSens.reshape(numPoints, self.numStrains, self.numStates * self.numDim)
        # Jacobian = _computeWeakJacobianProduct(strainSens, stressSens, scale)
        # points = p
        # strains = e
        # stress = o
        # states = s
        # dim = d
        # Jacobians = np.einsum(
        #     "posd,poe,peSD,p->pdsSD",
        #     strainSens,
        #     stressSens,
        #     strainSens,
        #     scale,
        #     optimize=["einsum_path", (1, 3), (0, 2), (0, 1)],
        # )
        Jacobians = np.zeros((numPoints, self.numDim, self.numStates, self.numStates, self.numDim))
        _computeWeakJacobianProduct(strainSens, stressSens, scale, Jacobians)

        return Jacobians

    # ==============================================================================
    # Private methods
    # ==============================================================================


@njit(cache=True, fastmath=True, parallel=True)
def _computeWeakResidualProduct(strainSens, stress, scale, result):
    """_summary_

    Equivalent to ::

        np.einsum("pesd,pe,p->pds", strainSens, stress, scale, optimize=["einsum_path", (0, 1), (0, 1)])

      Complete contraction:  pesd,pe,p->pds
            Naive scaling:  4
        Optimized scaling:  4
        Naive FLOP count:  1.844e+8
    Optimized FLOP count:  1.298e+8
    Theoretical speedup:  1.421e+0
    Largest intermediate:  1.025e+7 elements
    --------------------------------------------------------------------------------
    scaling        BLAS                current                             remaining
    --------------------------------------------------------------------------------
    2              0               p,pe->pe                          pesd,pe->pds
    4              0           pe,pesd->pds                              pds->pds

    Parameters
    ----------
    strainSens : _type_
        _description_
    stress : _type_
        _description_
    scale : _type_
        _description_
    result : _type_
        _description_
    """
    numPoints = strainSens.shape[0]
    numStrains = strainSens.shape[1]
    numStates = strainSens.shape[2]
    numDim = strainSens.shape[3]
    for p in prange(numPoints):
        for d in range(numDim):
            for s in range(numStates):
                for e in range(numStrains):
                    result[p, d, s] += strainSens[p, e, s, d] * stress[p, e]
        result[p] *= scale[p]


@njit(parallel=True, cache=True, fastmath=True)
def _computeWeakJacobianProduct(strainSens, stressSens, scale, jacs):
    """Computes the following product which is necessary for compute the weak residual Jacobian:

    $$j = (d\epsilon/du')^T \\times d\sigma/d\epsilon \\times d\epsilon/du'$$

    Equivalent to::

        np.einsum(
            "posd,poe,peSD,p->pdsSD",
            strainSens,
            stressSens,
            strainSens,
            scale,
            optimize=["einsum_path", (1, 3), (0, 2), (0, 1)],
            out=jacs,
        )

    # points = p
    # strains = e
    # stress = o
    # states = s
    # dim = d

    Complete contraction:  posd,poe,peSD,p->pdsSD
            Naive scaling:  7
        Optimized scaling:  6
        Naive FLOP count:  1.866e+10
    Optimized FLOP count:  2.650e+9
    Theoretical speedup:  7.043e+0
    Largest intermediate:  1.296e+8 elements
    --------------------------------------------------------------------------------
    scaling        BLAS                current                             remaining
    --------------------------------------------------------------------------------
    3              0             p,poe->poe                  posd,peSD,poe->pdsSD
    5              0         poe,posd->pesd                      peSD,pesd->pdsSD
    6              0       pesd,peSD->pdsSD                          pdsSD->pdsSD

    Parameters
    ----------
    strainSens : _type_
        _description_
    stressSens : _type_
        _description_
    scale : _type_
        _description_
    jacs : _type_
        _description_
    """
    numPoints = jacs.shape[0]
    numDim = jacs.shape[1]
    numStrains = stressSens.shape[1]
    numStates = jacs.shape[2]

    # poe,posd,p->pesd
    intProd1 = np.zeros((numPoints, numStrains, numStates, numDim))
    for p in prange(numPoints):
        for e in range(numStrains):
            for s in range(numStates):
                for d in range(numDim):
                    for o in range(numStrains):
                        intProd1[p, e, s, d] += stressSens[p, o, e] * strainSens[p, o, s, d]
        intProd1[p] *= scale[p]

    # pesd,peSD->pdsSD
    for p in prange(numPoints):
        for d in range(numDim):
            for s in range(numStates):
                for S in range(numStates):
                    for D in range(numDim):
                        for e in range(numStrains):
                            jacs[p, d, s, S, D] += intProd1[p, e, s, d] * strainSens[p, e, S, D]

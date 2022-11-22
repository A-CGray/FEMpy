"""
==============================================================================
FEMpy Problem Class
==============================================================================
@File    :   Problem.py
@Date    :   2022/11/21
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
from typing import Iterable, Union, Callable

# ==============================================================================
# External Python modules
# ==============================================================================


# ==============================================================================
# Extension modules
# ==============================================================================


class FEMpyProblem:
    """The FEMpy problem class represents a single finite element problem, many such problems may be associated with a
    single FEMpy model to represent different loading and boundary conditions.

    The problem class contains:
    - The state vactor
    - The residual vector
    - The Jacobian matrix

    And contains methods for:
    - Assembling local vectors and matrices into global vectors and matrices
    - Solving the problem
    - Writing the solution to a file
    """

    def __init__(self) -> None:
        pass

    def addLoadToNodes(
        self,
        name: str,
        nodeInds: Iterable[int],
        dof: Union[int, Iterable[int]],
        value: Union[float, Iterable[float]],
        totalLoad: bool = False,
    ) -> None:
        """Add a load to a set of nodes

        Parameters
        ----------
        name : str
            Name for this load
        nodeInds : int or iterable of ints
            Indicies of nodes to apply the load to
        dof : int or iterable of ints
            Degrees of freedom to apply this load to at each node
        values : float or iterable of floats
            Load values, if a single value is supplied then this value is applied to all specified degrees of freedom
        totalLoad : bool, optional
            If true then the `values` are treated as total loads and split uniformly between the nodes, by default False, in which case the `values` are applied at each node
        """
        return None

    def addBodyLoad(self, name: str, loadingFunction: Union[Callable, Iterable[float]]) -> None:
        """Add a volumetric forcing term, commonly known as a "body force"

        _extended_summary_

        Parameters
        ----------
        name : str
            Name for the load
        loadingFunction : function or array of length numDimensions
            Pass an array to define a uniform field, otherwise, pass a function with the signature `F = loadingFunction(coord)` where coord is an n x numDimensions array of coordinates and F is an n x numDimensions array of loads at each point.
        """
        return None

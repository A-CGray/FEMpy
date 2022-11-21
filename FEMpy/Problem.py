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

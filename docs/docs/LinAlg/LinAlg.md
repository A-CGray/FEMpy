# Efficient Jacobian Linear Algebra Routines

When computing derivatives and integrals on elements we need to repeatedly compute the determinant or inverse of the element's Jacobian.
Since these Jacobians are small (max 3x3) FEMpy uses the manual implementations below for these inverse and determinant calculations. 
With the help of numba, these routines are approximately 1-2 orders of magnitude quicker than numpy's `linalg.det` and `linalg.inv` functions.

::: FEMpy.LinAlg
"""
==============================================================================
Dynamic Analysis Functions
==============================================================================
@File    :   Dynamics.py
@Date    :   2021/03/31
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
from scipy.sparse.linalg import factorized

# ==============================================================================
# Extension modules
# ==============================================================================


def NewmarkExplicit(MMat, KMat, Force, tStep, tf, t0=0.0, u0=None, uDot0=None):
    """Perform a transient analysis using the explicit central difference version of the Newmark-beta method

    At each timestep:
    u_n+1 = u_n + dt*uDot_n + 1/2*uDDot_n*dt**2
    uDDot_n+1 = M^-1(f_n+1 - K*u_n+1)
    uDot_n+1 = uDot_n + dt/2*(uDDot_n + uDDot_n+1)

    Stability condition is w*dt<=2, where w is the highest natural frequency of the structure.py

    For more details see `the wikipedia page for the method. <https://en.wikipedia.org/wiki/Newmark-beta_method>`_

    Parameters
    ----------
    MMat : numpy array or Scipy sparse matrix
        Mass matrix
    KMat : numpy array or Scipy sparse matrix
        Stiffness matrix
    Force : Function or vector
        Force vector, can provide a vector if force is constant or a function which takes a single time argument and
        returns a force array for a time varying force
    tStep : float
        Timestep size
    tf : float
        Final analysis time
    t0 : float, optional
        Initial analysis time, by default 0.
    u0 : array, optional
        Inital states, by default zero
    uDot0 : array, optional
        Initial velocities, by default zero

    Returns
    -------
    u : nState x nTimestep array
        State history
    uDot : nState x nTimestep array
        Velocity history
    uDDot : nState x nTimestep array
        Acceleration history
    t : array of length nTimestep
        Time history
    """

    # --- Convert force to function of time if it isn't ---
    if not callable(Force):
        ForceFunc = lambda t: Force
    else:
        ForceFunc = Force

    # --- Generate time steps ---
    t = np.concatenate((np.arange(t0, tf, tStep), np.array([tf])))
    dt = t[1:] - t[:-1]

    # --- Generate history matrices ---
    F0 = ForceFunc(t0)
    numStates = np.shape(F0)[0]
    u = np.zeros((numStates, len(t)))
    u[:, 0] = 0.0 if u0 is None else u0
    uDot = np.zeros((numStates, len(t)))
    uDot[:, 0] = 0.0 if uDot0 is None else uDot0
    uDDot = np.zeros((numStates, len(t)))

    # --- Factorise the mass matrix ---
    MSolve = factorized(MMat)

    # --- Compute the initial acceleration ---
    Res = F0 - KMat @ u[:, 0]
    uDDot[:, 0] = MSolve(Res)
    print(
        f"Iteration 0, t = 0, max U = {np.linalg.norm(u[:, 0], ord = np.inf):.04e}, max UDot = {np.linalg.norm(uDot[:, 0], ord = np.inf):.04e}, max UDDot = {np.linalg.norm(uDDot[:, 0], ord = np.inf):.04e}"
    )

    for i in range(len(dt)):
        u[:, i + 1] = u[:, i] + dt[i] * uDot[:, i] + 0.5 * dt[i] ** 2 * uDDot[:, i]
        Res[:] = ForceFunc(t[i + 1]) - KMat @ u[:, i + 1]
        uDDot[:, i + 1] = MSolve(Res)
        uDot[:, i + 1] = uDot[:, i] + 0.5 * dt[i] * (uDDot[:, i + 1] + uDDot[:, i])
        print(
            f"Iteration {i+1}, t = {t[i + 1]:.03e}, max U = {np.linalg.norm(u[:, i+1], ord = np.inf):.04e}, max UDot = {np.linalg.norm(uDot[:, i+1], ord = np.inf):.04e}, max UDDot = {np.linalg.norm(uDDot[:, i+1], ord = np.inf):.04e}"
        )

    return u, uDot, uDDot, t

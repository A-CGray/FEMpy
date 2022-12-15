# Finite Element Theory (or FEMpy's version of it at least)

## Isoparametric Elements

In FEMpy, elements defining the mapping between the discrete and continuous.
An element has a certain number of nodes which have coordinates in real space (reffered to as $\vec{x}$), and at which the state, $u$ is defined.
Depending on the PDE being modelled, the state can be a scalar or a vector, (a single temperature value, or a 3-component displacement).

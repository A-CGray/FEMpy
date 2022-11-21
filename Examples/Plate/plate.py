import meshio
from FEMpy.Mesh import readNastranSPCs

mesh = meshio.read("../Meshes/Order5Quad.msh")
BCdict = readNastranSPCs("../Meshes/Order5Quad.msh")

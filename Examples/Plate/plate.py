import meshio
from FEMpy.Mesh import readNastranSPCs

mesh = meshio.read("../Meshes/LBracket.msh")
BCdict = readNastranSPCs("../Meshes/Order5Quad.msh")

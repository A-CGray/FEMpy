import meshio
from FEMpy.Mesh import readNastranSPCs

mesh = meshio.read("../Meshes/GMSHTest.msh")
BCdict = readNastranSPCs("../Meshes/GMSHTest.msh")

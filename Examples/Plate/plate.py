import meshio
from FEMpy.Mesh import readNastranSPCs

mesh = meshio.read("../Meshes/Plate.bdf")
BCdict = readNastranSPCs("../Meshes/Plate.bdf")

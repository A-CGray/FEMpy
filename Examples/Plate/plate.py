import meshio
from FEMpy.mesh.nastranBCreader import readBC

mesh = meshio.read("../Meshes/Plate.bdf")
BCdict = readBC("../Meshes/Plate.bdf")

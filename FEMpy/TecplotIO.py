"""
==============================================================================
Tecplot I/O Functions
==============================================================================
@File    :   TecplotIO.py
@Date    :   2021/04/02
@Author  :   Alasdair Christison Gray
@Description : Functions for reading and writing tecplot files
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import re

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


def readTecplot(filename):
    """Read a text based tecplot file

    This function assumes the following file structure:

    TITLE = <Some title name>
    VARIABLES = "<var1>"  "<var2>"  "<var3>" ....
    ZONE N=<number of nodes>, E=<number of elements>, ....
    <var1 @ node 1> <var2 @ node 1> <var3 @ node 1> ...
    .
    .
    .
    <Element 1 node 1> <Element 1 node 2> <Element 1 node 3> <Element 1 node 4> ...
    .
    .
    .

    Parameters
    ----------
    filename : str
        File name

    Returns
    -------
    data : dictionary
        A dictionary containing a "title" field and a separate array for each each nodal variable
    conn : 2d array
        Element connectivity matrix
    """
    # --- Open file and read lines ---
    with open(filename, "r") as file:
        lines = file.readlines()

        data = {}

        # --- Read title ---
        TitleMatch = re.match(r"TITLE = (.*$)", lines[0], re.M | re.I)
        data["title"] = TitleMatch[1].rstrip()

        # --- Read number of nodes and elements ---
        nums = re.match(r"ZONE N=([0-9]*), E=([0-9]*)", lines[2], re.M | re.I)
        numNodes = int(nums[1])
        numEl = int(nums[2])

        # --- Read variable names ---
        varMatch = re.match(r"VARIABLES = (.*$)", lines[1], re.M | re.I)
        varNames = varMatch[1].rstrip().split()
        for i in range(len(varNames)):
            varNames[i] = varNames[i].replace('"', "")
            data[varNames[i]] = np.zeros(numNodes)

        # --- Read nodal data values ---
        for i in range(3, 3 + numNodes):
            line = lines[i].rstrip().split()
            for col in range(len(varNames)):
                data[varNames[col]][i - 3] = float(line[col])

        # --- Read connectivity data, for now I just assume all elements have same number of nodes ---
        connLine = lines[3 + numNodes].rstrip().split()
        numCon = len(connLine)
        conn = np.zeros((numEl, numCon), dtype=np.int)

        for i in range(3 + numNodes, len(lines)):
            line = lines[i].rstrip().split()
            for col in range(numCon):
                conn[i - (3 + numNodes), col] = int(line[col]) - 1

    return data, conn

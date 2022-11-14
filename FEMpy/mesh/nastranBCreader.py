from collections import OrderedDict


def readBC(filename):
    """
    This function takes the mesh file name as an input and output the BC data as a dictionary format.

    Parameters
    ----------
    filename : str
        Mesh file name

    Returns
    -------
    Dictionary
        It returns a dictionary of BC data.
    """

    # string to search in file
    word = "SPC"

    outputDict = OrderedDict()
    with open(filename, "r") as fp:
        # read all lines using readline()
        lines = fp.readlines()
        for row in lines:
            # check if string present on a current line

            # print(row.find(word))
            # find() method returns -1 if the value is not found,
            # if found it returns index of the first occurrence of the substring
            if row.find(word) != -1:
                line_vals = row.split()
                if line_vals[0] == word:
                    SID = line_vals[1]
                    Gi = int(line_vals[2]) - 1
                    Ci = line_vals[3]
                    Val = line_vals[4]

                    # print(row.split(), line_vals, line)
                    if SID not in outputDict:

                        outputDict[SID] = OrderedDict()
                    if Gi not in outputDict[SID]:
                        outputDict[SID][Gi] = OrderedDict()
                        outputDict[SID][Gi]["DOF"] = []
                        outputDict[SID][Gi]["Val"] = []

                    for iw in Ci:
                        outputDict[SID][Gi]["DOF"].append(int(iw) - 1)
                        outputDict[SID][Gi]["Val"].append(float(Val))

    return outputDict

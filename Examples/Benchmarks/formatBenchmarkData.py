"""
==============================================================================
Benchmark data formatting script
==============================================================================
@File    :   formatBenchmarkData.py
@Date    :   2022/12/11
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import csv

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================

print(
    """\n\n\n
FEMpy Benchmark Results:
==============================================================================================
|                       Case                       | Status |    Time (s)     |  Memory (MB) |
=============================================================================================="""
)

# 38
filename = "benchmark_data.csv"

with open(filename, "r") as file:
    reader = csv.reader(file, delimiter=",")
    for line in reader:
        Casename = line[1].split(":")[-1]
        Casename = Casename.replace("benchmark", "")
        Casename = Casename.replace("Benchmark", "")
        Casename = Casename.replace(".", " ")
        status = line[2]

        time = float(line[3])
        mem = float(line[4])
        print(f"| {Casename:<48} | {status:^6} |  {time:11.6e}   | {mem:11.6e} |")

print("==============================================================================================")

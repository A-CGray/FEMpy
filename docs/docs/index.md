# FEMpy
FEMpy is my attempt to implement a basic object-oriented finite element method in python

![Pretty Colours](Images/PrettyColours.png)

FEMpy uses scipy's sparse matrix implementation to enable scaling to problems with many (>100k) degrees of freedom.
Wherever possible, operations use numpy vectorisation or numba JIT compiling for speed, there's still plenty of room for improvement though!

![FEMpy can easily handle problems with 100,000 degrees of freedom](Images/QuadElScaling.png)

## How to install
Inside the FEMpy root directory run:
```shell
pip install .
```
Or, if you want to make changes to the code:
```shell
pip install -e .
```
If you want to build documentation locally, or run the unit tests, make sure to install the necessary dependencies:
```shell
pip install -e .[docs, dev]
```
And then run:
```shell
make build
```

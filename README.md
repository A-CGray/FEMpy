# FEMpy

[![Docs](https://github.com/A-CGray/FEMpy/actions/workflows/docs.yml/badge.svg)](https://A-CGray.github.io/FEMpy/)
[![Unit Tests](https://github.com/A-CGray/FEMpy/actions/workflows/Tests.yml/badge.svg)](https://A-CGray.github.io/FEMpy/)
[![Code Formatting](https://github.com/A-CGray/FEMpy/actions/workflows/Formatting.yml/badge.svg)](https://A-CGray.github.io/FEMpy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Test Coverage](https://api.codeclimate.com/v1/badges/ba60cf87e121a05dfe5d/test_coverage)](https://codeclimate.com/github/A-CGray/FEMpy/test_coverage)
[![Maintainability](https://api.codeclimate.com/v1/badges/ba60cf87e121a05dfe5d/maintainability)](https://codeclimate.com/github/A-CGray/FEMpy/maintainability)

FEMpy is my attempt to implement a basic object-oriented finite element method in python.

![Pretty Colours](docs/docs/Images/PrettyColours.png)

FEMpy uses scipy's sparse matrix implementation to enable scaling to problems with many (>100k) degrees of freedom.
Wherever possible, operations use numpy vectorisation or numba JIT compiling for speed, there's still plenty of room for improvement though!

![FEMpy can easily handle problems with 100,000 degrees of freedom](docs/docs/Images/QuadElScaling.png)

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

To build the docs locally, or:
```shell
testflo -n 1 -v .
```

To run unit tests.

## Documentation
View the documentation (still under construction) [here](https://A-CGray.github.io/FEMpy/)

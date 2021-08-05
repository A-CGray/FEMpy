# FEMpy

[![Docs](https://github.com/A-Gray-94/FEMpy/actions/workflows/docs.yml/badge.svg)](https://a-gray-94.github.io/FEMpy/)
[![Unit Tests](https://github.com/A-Gray-94/FEMpy/actions/workflows/Tests.yml/badge.svg)](https://a-gray-94.github.io/FEMpy/)
[![Code Formatting](https://github.com/A-Gray-94/FEMpy/actions/workflows/Formatting.yml/badge.svg)](https://a-gray-94.github.io/FEMpy/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/A-Gray-94/FEMpy/branch/main/graph/badge.svg?token=VLBE5HHP68)](https://codecov.io/gh/A-Gray-94/FEMpy)
[![Maintainability](https://api.codeclimate.com/v1/badges/38d025776dc6fc3e77c5/maintainability)](https://codeclimate.com/github/A-Gray-94/FEMpy/maintainability)

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

## Documentation
View the documentation (still under construction) [here](https://A-Gray-94.github.io/FEMpy/)

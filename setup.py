from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import re
import os
import subprocess

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("FEMpy/__init__.py").read(),
)[0]


setup(
    name="FEMpy",
    version=__version__,
    description="FEMpy is my attempt to implement a basic object oriented finite element method in python",
    keywords="Finite Element Method, FEM",
    author="Alasdair Christison Gray",
    author_email="",
    url="https://github.com/A-Gray-94/FEMpy",
    license="Apache License Version 2.0",
    packages=find_packages(where="."),
    install_requires=[
        "mdolab-baseclasses",
        "meshio",
        "numpy",
        "numba",
        "scipy>=1.8.0",
    ],
    extras_require={
        "docs": [
            "mkdocs-material",
            "mkdocstrings[python]",
            "pytkdocs[numpy-style]",
        ],
        "dev": ["parameterized", "testflo", "black==24.4.2", "flake8==7.0.0", "pre-commit"],
    },
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
)

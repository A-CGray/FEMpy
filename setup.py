from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
import re
import os
import subprocess

__version__ = re.findall(
    r"""__version__ = ["']+([0-9\.]*)["']+""",
    open("FEMpy/__init__.py").read(),
)[0]


def computeGaussQuadValues(n, outdir=None):
    from numpy.polynomial.legendre import leggauss
    import pickle

    if outdir is None:
        outdir = ""
    gaussWeights = {}
    gaussCoords = {}
    for i in range(1, n + 1):
        gaussCoords[i - 1], gaussWeights[i - 1] = leggauss(i)
    with open(os.path.join(outdir, "FEMpy/GaussQuadWeights.pkl"), "wb") as f:
        pickle.dump(gaussWeights, f, protocol=-1)
    with open(os.path.join(outdir, "FEMpy/GaussQuadCoords.pkl"), "wb") as f:
        pickle.dump(gaussCoords, f, protocol=-1)


class installWrapper(install):
    """wrapper around setuptools' install method that will run a post install script"""

    def run(self):
        from distutils.sysconfig import get_python_lib

        install.run(self)
        computeGaussQuadValues(64, outdir=get_python_lib())


class developWrapper(develop):
    """wrapper around setuptools' develop method that will run a post install script"""

    def run(self):
        develop.run(self)
        computeGaussQuadValues(64)
        subprocess.run("pre-commit install", shell=True)


setup(
    name="FEMpy",
    version=__version__,
    description="FEMpy is my attempt to implement a basic object oriented finite element method in python",
    keywords="Finite Element Method, FEM",
    author="Alasdair Christison Gray",
    author_email="",
    url="https://github.com/A-Gray-94/FEMpy",
    license="Apache License Version 2.0",
    packages=["FEMpy"],
    install_requires=[
        "numpy",
        "numba",
        "scipy",
        "pyComposite @ git+https://github.com/A-Gray-94/pyComposite.git",
    ],
    extras_require={
        "docs": [
            "mkdocs",
            "pymdown-extensions",
            "mkdocs-material",
            "mkdocstrings",
            "pytkdocs[numpy-style]",
            "Jinja2>=2.11,<4.0",
        ],
        "dev": ["parameterized", "testflo", "black==22.1.0", "flake8==3.9.2", "pre-commit"],
    },
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    cmdclass={"install": installWrapper, "develop": developWrapper},
)

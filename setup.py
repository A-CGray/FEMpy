from setuptools import setup
import re

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
    packages=[
        "FEMpy",
    ],
    install_requires=["numpy", "numba", "scipy"],
    extras_require={
        "docs": [
            "mkdocs",
            "pymdown-extensions",
            "mkdocs-material",
            "mkdocstrings",
            "pytkdocs[numpy-style]",
            "Jinja2<3.0,>=2.11",
        ]
    },
    classifiers=["Operating System :: OS Independent", "Programming Language :: Python"],
)

"""
The MF6RTM (Modflow 6 Reactive Transport Model) package is a python package
for reactive transport modeling via the MODFLOW 6 and PhreeqcRM APIs.
"""

# populate package namespace
from mf6rtm import (
    solver,
    mf6api,
    phreeqcbmi,
    mup3d,
    utils,
)

from mf6rtm.solver import run_cmd

__author__ = "Pablo Ortega"

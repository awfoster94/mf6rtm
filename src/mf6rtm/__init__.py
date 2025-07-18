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

from mf6rtm.solver import run_cmd, Mf6RTM
from mf6rtm.mf6api import Mf6API
from mf6rtm.phreeqcbmi import PhreeqcBMI

__author__ = "Pablo Ortega"

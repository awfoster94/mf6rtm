"""
The MF6RTM (Modflow 6 Reactive Transport Model) package is a python package
for reactive transport modeling via the MODFLOW 6 and PhreeqcRM APIs.
"""

# populate package namespace
from mf6rtm import (
    mf6rtm,
    mf6api,
    phreeqcbmi,
    mup3d,
    utils,
)

from mf6rtm.mf6rtm import run_cmd

__author__ = "Pablo Ortega"

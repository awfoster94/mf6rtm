# MF6RTM: Reactive Transport Model via the MODFLOW 6 and PHREEQCRM APIs
![Tests](https://github.com/p-ortega/mf6rtm/actions/workflows/tests_main.yml/badge.svg)
![Tests](https://github.com/p-ortega/mf6rtm/actions/workflows/tests_macos.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/p-ortega/mf6rtm/badge.svg?branch=develop)](https://coveralls.io/github/p-ortega/mf6rtm?branch=main)
[![PyPI License](https://img.shields.io/pypi/l/mf6rtm)](https://pypi.python.org/pypi/mf6rtm)
<!-- [![PyPI Status](https://img.shields.io/pypi/status/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm) -->
<!-- [![PyPI Format](https://img.shields.io/pypi/format/mf6rtm)](https://pypi.python.org/pypi/mf6rtm) -->
[![PyPI Version](https://img.shields.io/pypi/v/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm)
[![PyPI Versions](https://img.shields.io/pypi/pyversions/mf6rtm.png)](https://pypi.python.org/pypi/mf6rtm)
[![DOI](https://zenodo.org/badge/798559356.svg)](https://doi.org/10.5281/zenodo.17118951)

## Benchmarks
Benchmark comparing model results against PHT3D are in `benchmark/`. Each folder contains a Jupyter notebook to write and execute an MF6RTM model via the MUP3D class. Additionally, PHT3D files are provided in the corresponding `pht3d` directory for each example.

## Considerations
The current version is intended to work with structured grids (dis object in MF6), unstructured grids (disv) and one MF6 simulation that includes the flow and transport solutions. No support is currently provided for a 'flow then transport scheme,' meaning that advanced packages cannot be incorporated yet.

On the PHREEQC side, the following have been included:

- Solution
- Equilibrium phases
- Cation Exchange
- Surface Complexation
- Kinetic Phases

Most options for each phreeqc block can be passed by adding list with options. However, not all options had been tested, so please create an issue if any option is not working or crashing the model.

## Software requirements
All dependencies and executables are included in this repo. This package extensively uses [modflowapi](https://github.com/MODFLOW-USGS/modflowapi) and [phreeqcrm](https://github.com/usgs-coupled/phreeqcrm)

## Installation
The package can be installed via pip:

```commandline
pip install mf6rtm
```

We recommend creating a dedicated environment to manage all dependencies and executables required to run mf6rtm. For example, using mamba with an environment file:

```commandline
mamba env create -f env.yml
mamba activate mf6rtm-dev
```

After activating the environment, install the MODFLOW6 executables into the environment's bin directory:

```commandline
pip install modflow-devtools
get-modflow --subset mf6,libmf6,gridgen :python
```

Once installed, the executables in `envs/[env-name]/bin` will be automatically invoked whenever mf6rtm runs within the environment. These are typically the latest stable versions. 

If you need to use custom or older versions of mf6 (e.g., for running PESTPP on an HPC cluster), place them in a separate directory and use the provided utility to copy the executables to the appropriate directory.

```Python
from mf6rtm import utils

utils.prep_bins(model_dir, src_path=path_to_bins)

```
### Running the benchmark notebooks
We have provided some benchmarks in the form of Jupyter notebooks. We have also included the executables needed to run them out of the box. Nevertheless, they can also be run using the executables downloaded with modflow-devtools.

## Developing
We recommend forking and cloning a local version of this repo. A development environment is provided in the `env.yml` file, which should install all required dependencies to run tests and modify the package on the fly. To install the environment, use the following command:

```commandline
conda env create -f env.yml
```

The rest of the development dependencies, especially for the testing suite is located in the requirements-dev file. We have also provided some dependencies with flopy and pyemu inside the repo but feel free to use your own distro.

## Funding
The developing of mf6rtm was kindly funded and supported by [Intera, Inc](https://www.intera.com).

## Authors
Pablo Ortega (Portega)

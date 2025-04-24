"""
This module contains the Discretization functions for various ModFlow grid types.
"""

from mf6rtm.modflow import Mf6API


def total_cells_in_grid(modflow_api: Mf6API) -> int:
    grid_type = modflow_api.grid_type.upper()
    try:
        return __DISCRETIZATION_FUNCTIONS[grid_type](modflow_api)
    except KeyError or NotImplementedError:
        raise ValueError(f"Grid type '{grid_type}' is not yet supported.")


def __not_supported(*args, **kargs):
    raise NotImplementedError("This grid type is not supported.")


def __dis(api: Mf6API) -> int:
    """
    Returns the total number of grid.
    """
    simulation = api.sim
    discretization = simulation.get_model(simulation.model_names[0]).dis
    nlay = discretization.nlay.get_data()
    nrow = discretization.nrow.get_data()
    ncol = discretization.ncol.get_data()
    return nlay * nrow * ncol


def __disv(api: Mf6API) -> int:
    simulation = api.sim
    return simulation.nlay * simulation.ncpl


__DISCRETIZATION_FUNCTIONS = {
    "DIS": __dis,
    "DISV": __disv,
    "DISU": __not_supported,
    "DISV2D": __not_supported,
    "DIS3D": __not_supported,
    "DISV3D": __not_supported,
    "UNDEFINED": __not_supported,
}

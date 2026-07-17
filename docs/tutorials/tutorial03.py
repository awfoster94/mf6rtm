# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pyrite oxidation on an unstructured (DISV) grid
#
# The other tutorials use structured (DIS) column grids. This one runs
# **pyrite oxidation** on an **unstructured quadtree (DISV)** mesh: oxidising
# water is injected along one edge of a single-layer aquifer that contains
# pyrite, driving oxidation and releasing **sulfate**. It is a small,
# self-contained, single-layer version of the Dizon deep-well-injection case;
# the full 3-D voronoi model lives in the GMDSI ``rtm-tutorial`` workshop.
#
# It uses the **classic workflow** (build the :class:`~mf6rtm.mup3d.base.Mup3d`
# model, then the MODFLOW 6 flow + per-component transport models) on a grid
# built with FloPy's ``Gridgen`` quadtree utility.

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy
from flopy.discretization import StructuredGrid, VertexGrid
from flopy.utils.gridgen import Gridgen

from mf6rtm import utils, mup3d

BASE = "." if os.path.isdir("data") else os.path.join("docs", "tutorials")
DATA = os.path.join(BASE, "data")

# %% [markdown]
# ## 1. Build a quadtree DISV grid with Gridgen
#
# A 1 m x 1 m single-layer domain on a 20x20 base grid, refined one level in a
# central square. ``Gridgen`` needs the ``gridgen`` executable (fetched into the
# environment by the docs pipeline).

# %%
length_units = "meters"
time_units = "days"

Lx = Ly = 1.0
nlay, nrow, ncol = 1, 20, 20
delr = Lx / ncol
delc = Ly / nrow
top = 1.0
botm = [0.0]

sim_ws = os.path.join(BASE, "_tutorial03_run")
gridgen_ws = os.path.join(sim_ws, "gridgen")
os.makedirs(gridgen_ws, exist_ok=True)

base_grid = StructuredGrid(
    delr=np.full(ncol, delr), delc=np.full(nrow, delc),
    top=np.full((nrow, ncol), top),
    botm=np.full((nlay, nrow, ncol), botm[0]),
)
g = Gridgen(base_grid, model_ws=gridgen_ws, exe_name="gridgen")

refine_poly = [[[(0.35 * Lx, 0.35 * Ly), (0.65 * Lx, 0.35 * Ly),
                 (0.65 * Lx, 0.65 * Ly), (0.35 * Lx, 0.65 * Ly),
                 (0.35 * Lx, 0.35 * Ly)]]]
g.add_refinement_features(refine_poly, "polygon", 2, [0])
g.build(verbose=False)
disv_props = g.get_gridprops_disv()
ncpl = disv_props["ncpl"]
print(f"DISV grid: {ncpl} cells in {nlay} layer")

# identify inflow (left edge) and outflow (right edge) cells by x-coordinate
grid = VertexGrid(**g.get_gridprops_vertexgrid())
xc = np.array(grid.xcellcenters)
left = np.where(xc < delr)[0]
right = np.where(xc > Lx - delr)[0]

# %% [markdown]
# ## 2. Geochemistry — pyrite oxidation network
#
# The same ``mup3d`` chemistry classes as the pyrite column tutorial, applied
# uniformly across the mesh: an initial anoxic pore water (**Solutions** 1)
# containing exchangers, calcite (**EquilibriumPhases**), surface sites
# (**Surfaces**) and kinetic **pyrite + organic matter** (**KineticPhases**).
# On a DISV grid ``set_ic`` takes an integer (uniform) or a per-cell array of
# length ``ncpl``.

# %%
solutionsdf = pd.read_csv(os.path.join(DATA, "ex5_solutions.csv"),
                          comment="#", index_col=0)
solution = mup3d.Solutions(utils.solution_df_to_dict(solutionsdf))
solution.set_ic(1)

# cation exchanger (uniform zone 1)
excdf = pd.read_csv(os.path.join(DATA, "ex5_exchange.csv"), comment="#", index_col=0)
excdf.columns = [0, 1, 2, 3]
exchanger_dict = excdf.to_dict()
for _z, subdict in exchanger_dict.items():
    for key in subdict:
        subdict[key] = {"m0": subdict[key]}
exchanger = mup3d.ExchangePhases(exchanger_dict)
exchanger.set_ic(1)
exchanger.set_equilibrate_solutions([1, 1, 1, 1])

# kinetic pyrite + organic matter
kin_df = pd.read_csv(os.path.join(DATA, "ex5_kinetic_phases.csv"))
kin_phases = utils.parse_kinetics_dataframe(kin_df)
kin_phases[1]["Orgc_sed"]["formula"] = "Orgc_sed -1.0 C 1.0"
kinetics = mup3d.KineticPhases(kin_phases)
kinetics.set_ic(1)

# equilibrium phases + surfaces
eqp_df = pd.read_csv(os.path.join(DATA, "ex5_equilibrium_phases.csv"))
equilibriums = mup3d.EquilibriumPhases(utils.parse_equilibriums_dataframe(eqp_df))
equilibriums.set_ic(1)

surfaces = mup3d.Surfaces(utils.surfaces_csv_to_dict(
    os.path.join(DATA, "ex5_surfaces.csv")))
surfaces.set_ic(1)

# %% [markdown]
# ## 3. Build the `Mup3d` model and assign the injected chemistry
#
# The model is created with ``ncpl`` (DISV) instead of ``nrow``/``ncol``. The
# oxidising inflow water (solution 3) is mapped to every inflow cell along the
# left edge; ``set_chem_stress`` stores its per-component concentrations.

# %%
model = mup3d.Mup3d("rtm3d", solution, nlay=nlay, ncpl=ncpl)
model.set_wd(sim_ws)
model.set_database(os.path.join(DATA, "ex5.dat"))
model.set_initial_temp([7.0, 7.0, 7.0])
model.set_postfix(os.path.join(DATA, "ex5_postfix.phqr"))

model.set_exchange_phases(exchanger)
model.set_phases(kinetics)
model.set_phases(equilibriums)
model.set_phases(surfaces)

model.initialize()

# inject oxidising water (solution 3) along the whole left edge.
# In the classic workflow set_spd lists the injected solution per stress period
# (here a single period → [3]); the concentrations are applied to every cell of
# the package.
chin = mup3d.ChemStress("chdin")
chin.set_spd([3])
model.set_chem_stress(chin)

# %% [markdown]
# ## 4. Flow + per-component transport on the DISV grid
#
# Constant-head inflow (left, carrying the injected components) and outflow
# (right) drive flow across the mesh; one GWT model per component transports it.

# %%
nper = 1
tdis_rc = [(2.0, 40, 1.0)]

chdin = [[(0, int(i)), 1.0] for i in left]      # head; components appended below
chdout = [[(0, int(i)), 0.5] for i in right]


def build_model(model):
    sim = flopy.mf6.MFSimulation(sim_name=model.name, sim_ws=model.wd, exe_name="mf6")
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=tdis_rc, time_units=time_units)

    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True)
    imsgwf = flopy.mf6.ModflowIms(sim, complexity="complex", linear_acceleration="CG",
                                  filename="gwf.ims")
    sim.register_ims_package(imsgwf, [gwf.name])

    flopy.mf6.ModflowGwfdisv(gwf, length_units=length_units, **disv_props)
    flopy.mf6.ModflowGwfnpf(gwf, save_specific_discharge=True, save_saturation=True,
                            icelltype=0, k=1.0, filename="gwf.npf")
    flopy.mf6.ModflowGwfic(gwf, strt=1.0, filename="gwf.ic")

    # append the injected component concentrations (period 0) to every inflow CHD row
    for i in range(len(chdin)):
        chdin[i].extend(model.chdin.data[0])
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdin, auxiliary=model.components,
                            pname="chdin", filename="gwf.chdin.chd")
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chdout, pname="chdout",
                            filename="gwf.chdout.chd")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="gwf.hds", budget_filerecord="gwf.cbb",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])

    for c in model.components:
        gwt = flopy.mf6.MFModel(sim, model_type="gwt6", modelname=c,
                                model_nam_file=f"{c}.nam")
        imsgwt = flopy.mf6.ModflowIms(sim, complexity="complex",
                                      linear_acceleration="BICGSTAB", filename=f"{c}.ims")
        sim.register_ims_package(imsgwt, [gwt.name])

        flopy.mf6.ModflowGwtdisv(gwt, length_units=length_units, **disv_props)
        flopy.mf6.ModflowGwtic(gwt, strt=model.sconc[c], filename=f"{c}.ic")
        flopy.mf6.ModflowGwtssm(gwt, sources=[["chdin", "aux", c]], filename=f"{c}.ssm")
        flopy.mf6.ModflowGwtadv(gwt, scheme="tvd")
        flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=0.01, ath1=0.001,
                                filename=f"{c}.dsp")
        flopy.mf6.ModflowGwtmst(gwt, porosity=0.35, filename=f"{c}.mst")
        flopy.mf6.ModflowGwtoc(gwt, concentration_filerecord=f"{c}.ucn",
                               saverecord=[("CONCENTRATION", "ALL")])
        flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea="gwf",
                                exgmnameb=c, filename=f"{c}.gwfgwt")

    sim.write_simulation()
    return sim


sim = build_model(model)

# %% [markdown]
# ## 5. Run

# %%
# Stage the platform's MODFLOW 6 binaries into the run directory (pixi fetches
# them into ``benchmark/bin``) so the solver finds libmf6 locally.
utils.prep_bins(model.wd, src_path=os.path.join(BASE, "..", "..", "benchmark", "bin"))

model.run()

# %% [markdown]
# ## 6. Results: sulfate across the mesh
#
# Sulfate (``S(6)``) released by pyrite oxidation at the final time, shown on the
# unstructured quadtree grid — highest near the oxidising inflow edge.

# %%
sout = pd.read_csv(os.path.join(model.wd, "sout.csv"))
final = sout[sout["time_d"] == sout["time_d"].max()]

arr = np.full(ncpl, np.nan)
arr[final["cell"].astype(int).values] = final["S(6)"].values

fig, ax = plt.subplots(figsize=(6, 5))
pmv = flopy.plot.PlotMapView(modelgrid=grid, ax=ax)
so4 = pmv.plot_array(arr)
pmv.plot_grid(linewidth=0.3, color="0.5")
plt.colorbar(so4, ax=ax, shrink=0.8, label="S(6) (mol/L)")
ax.set_title("Sulfate from pyrite oxidation at t = %.2f d" % sout["time_d"].max())
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
fig.tight_layout()

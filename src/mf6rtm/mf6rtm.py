"""The mf6rtm module provides the Mf6RTM class that couples modflowapi and
phreeqcrm.
"""

from typing import Any
from pathlib import Path
import os
from os import PathLike
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
from PIL import Image
import pandas as pd
import numpy as np

# from time import sleep
from mf6rtm.mf6api import Mf6API
from mf6rtm.phreeqcbmi import PhreeqcBMI
from mf6rtm import utils
from mf6rtm.discretization import total_cells_in_grid

# global variables
DT_FMT = "%Y-%m-%d %H:%M:%S"

time_units_dict = {
    "seconds": 1,
    "minutes": 60,
    "hours": 3600,
    "days": 86400,
    "years": 31536000,
    "unknown": 1,  # if unknown assume seconds
}


def prep_to_run(wd: PathLike) -> tuple[PathLike, PathLike]:
    """
    Prepares the model to run by checking if the model directory (wd) contains the necessary files
    and returns the path to the yaml file (phreeqcrm) and the dll file (mf6 api)

    Parameters
    ----------
    wd : PathLike
        The path to the working directory of model directory
    Returns
    -------
    tuple[PathLike, PathLike]
        The path to the phreeqcrm model file (yaml) and the path to the MODFLOW 6 dll (associated with mf6api).
    """
    # check if wd exists
    assert os.path.exists(wd), f"Path {wd} not found"
    # check if file starting with libmf6 exists
    dll = [f for f in os.listdir(wd) if f.startswith("libmf6")]
    assert len(dll) == 1, "libmf6 dll not found in model directory"
    assert os.path.exists(
        os.path.join(wd, "mf6rtm.yaml")
    ), "mf6rtm.yaml not found in model directory"

    nam = [f for f in os.listdir(wd) if f.endswith(".nam")]
    assert "mfsim.nam" in nam, "mfsim.nam file not found in model directory"
    assert "gwf.nam" in nam, "gwf.nam file not found in model directory"
    dll = os.path.join(wd, "libmf6")
    yamlfile = os.path.join(wd, "mf6rtm.yaml")

    return yamlfile, dll


def solve(wd: PathLike, reactive: bool = True, nthread: int = 1) -> bool:
    """Wrapper to prepare and call solve functions"""

    mf6rtm = initialize_interfaces(wd, nthread=nthread)
    if not reactive:
        mf6rtm._set_reactive(reactive)
    success = mf6rtm._solve()
    return success


# TODO: we should maybe move this into the Mf6API as an alternative constructor
def initialize_interfaces(wd: PathLike, nthread: int = 1) -> Mf6API:
    """Function to initialize the interfaces for modflowapi and phreeqcrm and returns the mf6rtm object"""

    yamlfile, dll = prep_to_run(wd)

    if nthread > 1:
        # set nthreds to nthread
        set_nthread_yaml(yamlfile, nthread=nthread)

    # initialize the interfaces
    mf6api = Mf6API(wd, dll)
    phreeqcrm = PhreeqcBMI(yamlfile)
    mf6rtm = Mf6RTM(wd, mf6api, phreeqcrm)
    return mf6rtm


def set_nthread_yaml(yamlfile: PathLike, nthread: int = 1) -> None:
    """Function to set the number of threads in the yaml file"""
    with open(yamlfile, "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "nthreads" in line:
            lines[i] = f"  nthreads: {nthread}\n"
    with open(yamlfile, "w") as f:
        f.writelines(lines)
    return


class Mf6RTM(object):
    def __init__(self, wd: PathLike, mf6api: Mf6API, phreeqcbmi: PhreeqcBMI) -> None:
        """
        Initialize the Mf6RTM instance with specified working directory, MF6API,
        and PhreeqcBMI instances.

        Parameters
        ----------
        wd : PathLike
            The working directory path for the model.
        mf6api : Mf6API
            An instance of the Mf6API class, representing the Modflow 6 API.
        phreeqcbmi : PhreeqcBMI
            An instance of the PhreeqcBMI class, representing the PHREEQC BMI.

        Attributes
        ----------
        mf6api : Mf6API
            The Modflow 6 API instance.
        phreeqcbmi : PhreeqcBMI
            The PHREEQC BMI instance.
        charge_offset : float
            Offset for charge, initialized to 0.0.
        wd : PathLike
            The working directory path.
        sout_fname : str
            Filename for the output, default is "sout.csv".
        reactive : bool
            Flag indicating if the model is reactive, default is True.
        epsaqu : float
            ??Epsaqueous value??, initialized to 0.0.
        fixed_components : Any
            Fixed components, default is None.
        get_selected_output_on : bool
            Flag indicating if selected output is on, default is True.
        component_model_dict : dict[str, str]
            Dictionary mapping PHREEQC aqueous chemical components to their
            corresponding Modflow 6 groundwater transport (gwt6) model names.
        nxyz : int
            Total number of cells in the grid.
        """

        assert isinstance(mf6api, Mf6API), "MF6API must be an instance of Mf6API"
        assert isinstance(
            phreeqcbmi, PhreeqcBMI
        ), "PhreeqcBMI must be an instance of PhreeqcBMI"
        self.mf6api = mf6api
        self.phreeqcbmi = phreeqcbmi
        self.charge_offset = 0.0
        self.wd = wd
        self.sout_fname = "sout.csv"
        self.reactive = True
        self.epsaqu = 0.0
        self.fixed_components = None
        self.get_selected_output_on = True

        # set component model dictionary
        self.component_model_dict = self._create_component_model_dict()

        # set discretization
        self.nxyz = total_cells_in_grid(self.mf6api)
        # set time conversion factor
        self.set_time_conversion()


    def get_saturation_from_mf6(self) -> dict[Any, np.ndarray]:
        """
        Get the saturation

        Parameters
        ----------
        mf6 (modflowapi): the modflow api object

        Returns
        -------
        array: the saturation
        """
        sat = {
            component: self.mf6api.get_value(
                self.mf6api.get_var_address(
                    "FMI/GWFSAT",
                    f"{self.component_model_dict[component]}"
                )
            )
            for component in self.phreeqcbmi.components
        }
        # select the first component to get the length of the array
        sat = sat[
            self.phreeqcbmi.components[0]
        ]  # saturation is the same for all components
        self.phreeqcbmi.sat_now = sat  # set phreeqcmbi saturation
        return sat

    def get_time_units_from_mf6(self) -> str:
        """Function to get the time units from mf6"""
        return self.mf6api.sim.tdis.time_units.get_data()

    def set_time_conversion(self) -> None:
        """Function to set the time conversion factor"""
        time_units = self.get_time_units_from_mf6()
        self.time_conversion = 1.0 / time_units_dict[time_units]
        self.phreeqcbmi.SetTimeConversion(self.time_conversion)

    def _create_component_model_dict(self) -> dict[str, str]:
        """
        Create a dictionary of PHREEQC aqueous chemical component names and
        their corresponding Modflow 6 Groundwater Transport (GWT) model names.

        Returns
        -------
        component_model_dict : dict[str, str]
            A dictionary where the keys are the component names and the values are
            the corresponding model names.
        """
        components = self.phreeqcbmi.get_value_ptr("Components")
        components = [str(component) for component in components]

        component_codes = []
        for component in components:
            component_code = component.lower()
            if component_code == 'charge':
                component_code = 'ch'
            component_codes.append(component_code)

        model_names = self.mf6api.sim.model_names
        gwt_model_names = []
        for model_name in model_names:
            if self.mf6api.sim.get_model(model_name).model_type == 'gwt6':
                gwt_model_names.append(model_name)

        # Confirm alignment
        assert len(components) == len(gwt_model_names)
        assert len(components) == len(component_codes)
        for component_code, gwt_model_name in zip(component_codes, gwt_model_names):
            assert component_code.lower() in gwt_model_name.lower()

        return dict(zip(components, gwt_model_names))


    # TODO: remove or have raise not implemented error
    def _set_fixed_components(self, fixed_components): ...

    # TODO: make reactive a property
    def _set_reactive(self, reactive: bool) -> None:
        """Set the model to run only transport or transport and reactions"""
        self.reactive = reactive

    def _prepare_to_solve(self) -> None:
        """Prepare the model to solve"""
        # check if sout fname exists
        if self._check_sout_exist():
            # if found remove it
            self._rm_sout_file()

        self.mf6api._prepare_mf6()
        self.phreeqcbmi._prepare_phreeqcrm_bmi()

        # get and write sout headers
        self._write_sout_headers()

    def _set_ctime(self) -> float:
        """Set the current time of the simulation from mf6api"""
        self.ctime = self.mf6api.get_current_time()
        self.phreeqcbmi._set_ctime(self.ctime)
        return self.ctime

    def _set_etime(self) -> float:
        """Set the end time of the simulation from mf6api"""
        self.etime = self.mf6api.get_end_time()
        return self.etime

    def _set_time_step(self) -> float:
        self.time_step = self.mf6api.get_time_step()
        return self.time_step

    def _finalize(self) -> None:
        """Finalize the APIs"""
        self._finalize_mf6api()
        self._finalize_phreeqcrm()

    def _finalize_mf6api(self) -> None:
        """Finalize the mf6api"""
        self.mf6api.finalize()

    def _finalize_phreeqcrm(self) -> None:
        """Finalize the phreeqcrm api"""
        self.phreeqcbmi.finalize()

    def _get_cdlbl_vect(self) -> np.ndarray[np.float64]:
        """Get the concentration array from phreeqc bmi reshape to (ncomps, nxyz)"""
        c_dbl_vect = self.phreeqcbmi.GetConcentrations()

        conc = [
            c_dbl_vect[i : i + self.nxyz] for i in range(0, len(c_dbl_vect), self.nxyz)
        ]  # reshape array
        return conc

    def _set_conc_at_current_kstep(self, c_dbl_vect: np.ndarray[np.float64]):
        """Saves the current concentration array to the object"""
        self.current_iteration_conc = np.reshape(
            c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz)
        )

    def _set_conc_at_previous_kstep(self, c_dbl_vect: np.ndarray[np.float64]):
        """Saves the current concentration array to the object"""
        self.previous_iteration_conc = np.reshape(
            c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz)
        )

    def _transfer_array_to_mf6(self) -> np.ndarray[np.float64]:
        """Transfer the concentration array to mf6"""
        c_dbl_vect = self._get_cdlbl_vect()

        # check if reactive cells were skipped due to small changes from transport and replace with previous conc
        if self._check_previous_conc_exists() and self._check_inactive_cells_exist(
            self.diffmask
        ):
            c_dbl_vect = self._replace_inactive_cells(c_dbl_vect, self.diffmask)
        else:
            pass

        conc_dict = {}
        for i, c in enumerate(self.phreeqcbmi.components):
            conc_dict[c] = c_dbl_vect[i]
            # Set concentrations in mf6
            gwt_model_name = self.component_model_dict[c]
            if gwt_model_name.lower() == "charge":
                self.mf6api.set_value(
                    f"{gwt_model_name.upper()}/X",
                    concentration_l_to_m3(conc_dict[c]) + self.charge_offset,
                )
            else:
                self.mf6api.set_value(
                    f"{gwt_model_name.upper()}/X",
                    concentration_l_to_m3(conc_dict[c]),
                )
        return c_dbl_vect

    def _check_previous_conc_exists(self) -> bool:
        """Function to replace inactive cells in the concentration array"""
        # check if self.previous_iteration_conc is a property
        return hasattr(self, "previous_iteration_conc")

    def _check_inactive_cells_exist(self, diffmask: np.ndarray[np.float64]) -> bool:
        """Function to check if inactive cells exist in the concentration array"""
        inact = utils.get_indices(0, diffmask)
        return len(inact) > 0

    def _replace_inactive_cells(
        self,
        c_dbl_vect: np.ndarray[np.float64],
        diffmask: np.ndarray[np.float64],
    ) -> np.ndarray[np.float64]:
        """Function to replace inactive cells in the concentration array"""
        c_dbl_vect = np.reshape(c_dbl_vect, (self.phreeqcbmi.ncomps, self.nxyz))
        # get inactive cells
        inactive_idx = [
            utils.get_indices(0, diffmask) for k in range(self.phreeqcbmi.ncomps)
        ]
        c_dbl_vect[:, inactive_idx] = self.previous_iteration_conc[:, inactive_idx]
        c_dbl_vect = c_dbl_vect.flatten()
        conc = [
            c_dbl_vect[i : i + self.nxyz] for i in range(0, len(c_dbl_vect), self.nxyz)
        ]
        return conc

    def _transfer_array_to_phreeqcrm(self) -> np.ndarray[np.float64]:
        """Transfer the concentration array to phreeqc bmi"""
        mf6_conc_array = []
        for c in self.phreeqcbmi.components:
            if c.lower() == "charge":
                mf6_conc_array.append(
                    concentration_m3_to_l(
                        self.mf6api.get_value(
                            self.mf6api.get_var_address(
                                "X",
                                f"{self.component_model_dict[c].upper()}",
                            )
                        )
                        - self.charge_offset
                    )
                )

            else:
                mf6_conc_array.append(
                    concentration_m3_to_l(
                        self.mf6api.get_value(
                            self.mf6api.get_var_address(
                                "X",
                                f"{self.component_model_dict[c].upper()}",
                            )
                        )
                    )
                )

        c_dbl_vect = np.reshape(mf6_conc_array, self.nxyz * self.phreeqcbmi.ncomps)
        self.phreeqcbmi.SetConcentrations(c_dbl_vect)

        # set the kper and kstp
        self.phreeqcbmi._get_kper_kstp_from_mf6api(
            self.mf6api
        )  # FIXME: calling this func here is not ideal

        return c_dbl_vect

    def _update_selected_output(self) -> None:
        """Update the selected output dataframe and save to attribute"""
        self._get_selected_output()
        updf = pd.concat(
            [
                self.phreeqcbmi.soutdf.astype(self._current_soutdf.dtypes),
                self._current_soutdf,
            ]
        )
        self._update_soutdf(updf)

    def __replace_inactive_cells_in_sout(self, sout, diffmask):
        """Function to replace inactive cells in the selected output dataframe"""
        # match headers in components closest string

        inactive_idx = utils.get_indices(0, diffmask)

        sout[:, inactive_idx] = self._sout_k[:, inactive_idx]
        return sout

    def _get_selected_output(self) -> None:
        """Get the selected output from phreeqc bmi and replace skipped reactive cells with previous conc"""
        # selected ouput
        self.phreeqcbmi.set_scalar("NthSelectedOutput", 0)
        sout = self.phreeqcbmi.GetSelectedOutput()
        sout = [sout[i : i + self.nxyz] for i in range(0, len(sout), self.nxyz)]
        sout = np.array(sout)
        if self._check_inactive_cells_exist(self.diffmask) and hasattr(self, "_sout_k"):

            sout = self.__replace_inactive_cells_in_sout(sout, self.diffmask)
        self._sout_k = sout  # save sout to a private attribute
        # add time to selected ouput
        sout[0] = np.ones_like(sout[0]) * (self.ctime + self.time_step)
        df = pd.DataFrame(columns=self.phreeqcbmi.soutdf.columns)
        for col, arr in zip(df.columns, sout):
            df[col] = arr
        self._current_soutdf = df

    def _update_soutdf(self, df: pd.DataFrame) -> None:
        """Update the selected output dataframe to phreeqcrm object"""
        self.phreeqcbmi.soutdf = df

    def _check_sout_exist(self) -> bool:
        """Check if selected output file exists"""
        return os.path.exists(os.path.join(self.wd, self.sout_fname))

    def _write_sout_headers(self) -> None:
        """Write selected output headers to a file"""
        with open(os.path.join(self.wd, self.sout_fname), "w") as f:
            f.write(",".join(self.phreeqcbmi.sout_headers))
            f.write("\n")

    def _rm_sout_file(self) -> None:
        """Remove the selected output file"""
        try:
            os.remove(os.path.join(self.wd, self.sout_fname))
        except:
            pass

    def _append_to_soutdf_file(self) -> None:
        """Append the current selected output to the selected output file"""
        assert not self._current_soutdf.empty, "current sout is empty"
        self._current_soutdf.to_csv(
            os.path.join(self.wd, self.sout_fname), mode="a", index=False, header=False
        )

    def _export_soutdf(self) -> None:
        """Export the selected output dataframe to a csv file"""
        self.phreeqcbmi.soutdf.to_csv(
            os.path.join(self.wd, self.sout_fname), index=False
        )

    def _solve(self) -> bool:
        """Alias for the solve method to provide backward compatibility"""
        return self.solve()

    def solve(self) -> bool:
        """Solve the model"""
        success = False  # initialize success flag
        sim_start = datetime.now()
        self._prepare_to_solve()

        # check sout was created
        assert self._check_sout_exist(), f"{self.sout_fname} not found"

        print("Starting transport solution at {0}".format(sim_start.strftime(DT_FMT)))
        ctime = self._set_ctime()
        etime = self._set_etime()
        while ctime < etime:
            temp_time = datetime.now()
            print(f"Starting solution at {temp_time.strftime(DT_FMT)}")
            # length of the current solve time
            dt = self._set_time_step()
            self.mf6api.prepare_time_step(dt)
            self.mf6api._solve_gwt()

            # get saturation
            self.get_saturation_from_mf6()

            if self.reactive:
                # reaction block
                c_dbl_vect = self._transfer_array_to_phreeqcrm()
                self._set_conc_at_current_kstep(c_dbl_vect)
                if ctime == 0.0:
                    self.diffmask = np.ones(self.nxyz)
                else:
                    diffmask = get_conc_change_mask(
                        self.current_iteration_conc,
                        self.previous_iteration_conc,
                        self.phreeqcbmi.ncomps,
                        self.nxyz,
                        treshold=self.epsaqu,
                    )
                    self.diffmask = diffmask
                # solve reactions
                self.phreeqcbmi._solve_phreeqcrm(dt, diffmask=self.diffmask)
                c_dbl_vect = self._transfer_array_to_mf6()
                if self.get_selected_output_on:
                    # get sout and update df
                    self._update_selected_output()
                    # append current sout rows to file
                    self._append_to_soutdf_file()
                self._set_conc_at_previous_kstep(c_dbl_vect)

            self.mf6api.finalize_time_step()
            ctime = self._set_ctime()  # update the current time tracking

        sim_end = datetime.now()
        td = (sim_end - sim_start).total_seconds() / 60.0

        self.mf6api._check_num_fails()

        print(
            "\nReactive transport solution finished at {0} --- it took: {1:10.5G} mins".format(
                sim_end.strftime(DT_FMT), td
            )
        )

        # Clean up and close api objs
        try:
            self._finalize()
            success = True
            print(mrbeaker())
            print(
                "\nMR BEAKER IMPORTANT MESSAGE: MODEL RUN FINISHED BUT CHECK THE RESULTS .. THEY ARE PROLY RUBBISH\n"
            )
        except:
            print("MR BEAKER IMPORTANT MESSAGE: SOMETHING WENT WRONG. BUMMER\n")
            pass
        return success


def get_less_than_zero_idx(arr):
    """Function to get the index of all occurrences of <0 in an array"""
    idx = np.where(arr < 0)
    return idx


def get_inactive_idx(arr: np.ndarray, val: float = 1e30):
    """Function to get the index of all occurrences of <0 in an array"""
    idx = list(np.where(arr >= val)[0])
    return idx


def get_conc_change_mask(
    ci: np.ndarray[np.float64],
    ck: np.ndarray[np.float64],
    ncomp: int,
    nxyz: int,
    treshold: float = 1e-10,
) -> np.ndarray[np.float64]:
    """Function to get the active-inactive cell mask for concentration change to inform phreeqc which cells to update"""
    # reshape arrays to 2D (nxyz, ncomp)
    ci = ci.reshape(nxyz, ncomp)
    ck = ck.reshape(nxyz, ncomp) + 1e-30

    # get the difference between the two arrays and divide by ci
    diff = np.abs((ci - ck.reshape(-1 * nxyz, ncomp)) / ci) < treshold
    diff = np.where(diff, 0, 1)
    diff = diff.sum(axis=1)

    # where values <0 put -1 else 1
    diff = np.where(diff == 0, 0, 1)
    return diff


def mrbeaker() -> str:
    """ASCII art of Mr. Beaker"""
    # get the path of this file
    whereismrbeaker = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "mrbeaker.png"
    )
    mr_beaker_image = Image.open(whereismrbeaker)

    # Resize the image to fit the terminal width
    terminal_width = 80  # Adjust this based on your terminal width
    aspect_ratio = mr_beaker_image.width / mr_beaker_image.height
    terminal_height = int(terminal_width / aspect_ratio * 0.5)
    mr_beaker_image = mr_beaker_image.resize((terminal_width, terminal_height))

    # Convert the image to grayscale
    mr_beaker_image = mr_beaker_image.convert("L")

    # Convert the grayscale image to ASCII art
    ascii_chars = "%,.?>#*+=-:."

    mrbeaker = ""
    for y in range(int(mr_beaker_image.height)):
        mrbeaker += "\n"
        for x in range(int(mr_beaker_image.width)):
            pixel_value = mr_beaker_image.getpixel((x, y))
            mrbeaker += ascii_chars[pixel_value // 64]
        mrbeaker += "\n"
    return mrbeaker


def flatten_list(xss):
    """Flatten a list of lists"""
    return [x for xs in xss for x in xs]


def concentration_l_to_m3(x):
    """Convert M/L to M/m3"""
    c = x * 1e3
    return c


def concentration_m3_to_l(x):
    """Convert M/L to M/m3"""
    c = x * 1e-3
    return c


def concentration_to_massrate(q, conc):
    """Calculate mass rate from rate (L3/T) and concentration (M/L3)"""
    mrate = q * conc  # M/T
    return mrate


def concentration_volbulk_to_volwater(conc_volbulk, porosity):
    """Calculate concentrations as volume of pore water from bulk volume and porosity"""
    conc_volwater = conc_volbulk * (1 / porosity)
    return conc_volwater


def run_cmd():
    # get the current directory
    cwd = os.getcwd()
    # run the solve function
    solve(cwd)

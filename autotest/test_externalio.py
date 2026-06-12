"""Unit tests for externalio.py.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from mf6rtm.io.externalio import SelectedOutput, Regenerator


# Shared fixtures

@pytest.fixture
def mock_mf6rtm():
    m = MagicMock()
    m.wd = tempfile.mkdtemp()
    m.phreeqcbmi.sout_headers = ["time_d", "cell", "pH", "pe"]
    m.phreeqcbmi.soutdf.columns = ["time_d", "cell", "pH", "pe"]
    yield m
    shutil.rmtree(m.wd, ignore_errors=True)


# configurable sout filename and format

class TestSelectedOutputInit:
    def test_default_params(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        assert so.sout_fname == "sout.csv"
        assert so.output_format == "csv"

    def test_custom_csv_fname(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm, sout_fname="results.csv")
        assert so.sout_fname == "results.csv"
        assert so.output_format == "csv"

    def test_explicit_output_format_csv(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.csv", output_format="csv")
        assert so.output_format == "csv"

    def test_h5_extension_auto_detected(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.h5")
        assert so.output_format == "hdf5"

    def test_hdf5_extension_auto_detected(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.hdf5")
        assert so.output_format == "hdf5"

    def test_explicit_hdf5_format(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.h5", output_format="hdf5")
        assert so.output_format == "hdf5"

    def test_hdf5_missing_tables_raises(self, mock_mf6rtm):
        with patch.dict("sys.modules", {"tables": None}):
            with pytest.raises(ImportError, match="pip install tables"):
                SelectedOutput(mock_mf6rtm, output_format="hdf5")

    def test_csv_extension_with_h5_name_no_conflict(self, mock_mf6rtm):
        """Explicit output_format='csv' overrides extension detection."""
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.csv", output_format="csv")
        assert so.output_format == "csv"


# spatial cell identifier columns

class TestAddSpatialColumns:
    def test_dis_adds_cell_layer_row_col(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 24, "pH": range(24), "pe": range(24)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(2, 3, 4)):
            result = so._add_spatial_columns(df)
        assert list(result.columns[1:5]) == ["cell", "layer", "row", "col"]
        assert result["cell"].iloc[0] == 1
        assert result["layer"].iloc[0] == 1
        assert result["row"].iloc[0] == 1
        assert result["col"].iloc[0] == 1
        assert result["cell"].iloc[-1] == 24
        assert result["layer"].iloc[-1] == 2

    def test_disv_adds_cell_layer_cell2d(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 30, "pH": range(30)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(3, 10)):
            result = so._add_spatial_columns(df)
        assert list(result.columns[1:4]) == ["cell", "layer", "cell2d"]
        assert "col" not in result.columns
        assert result["layer"].iloc[10] == 2   # 11th cell (0-indexed=10) → layer 2

    def test_cell_not_duplicated_if_phreeqc_has_it(self, mock_mf6rtm):
        """PHREEQC cell column is dropped and replaced by our spatial cell."""
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 6, "cell": [1, 2, 3, 4, 5, 6], "pH": range(6)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 2, 3)):
            result = so._add_spatial_columns(df)
        assert list(result.columns).count("cell") == 1
        assert result.columns[0] == "time_d"
        assert result.columns[1] == "cell"

    def test_dis_indices_are_one_based(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 6, "pH": range(6)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 2, 3)):
            result = so._add_spatial_columns(df)
        assert result["cell"].min() == 1
        assert result["layer"].min() == 1
        assert result["row"].min() == 1
        assert result["col"].min() == 1

    def test_time_column_remains_first(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        df = pd.DataFrame({"time_d": [1.0] * 6, "pH": range(6)})
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 2, 3)):
            result = so._add_spatial_columns(df)
        assert result.columns[0] == "time_d"

    def test_write_sout_headers_dis_includes_spatial(self, mock_mf6rtm):
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time_d", "pH", "pe"]
        so = SelectedOutput(mock_mf6rtm)
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(2, 3, 4)):
            so._write_sout_headers()
        header_line = open(os.path.join(mock_mf6rtm.wd, "sout.csv")).readline().strip()
        cols = header_line.split(",")
        assert cols[0] == "time_d"
        assert cols[1] == "cell"
        assert cols[2] == "layer"
        assert cols[3] == "row"
        assert cols[4] == "col"
        assert "pH" in cols
        assert "pe" in cols

    def test_write_sout_headers_disv_includes_spatial(self, mock_mf6rtm):
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time_d", "pH"]
        so = SelectedOutput(mock_mf6rtm)
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(3, 10)):
            so._write_sout_headers()
        header_line = open(os.path.join(mock_mf6rtm.wd, "sout.csv")).readline().strip()
        cols = header_line.split(",")
        assert cols[0] == "time_d"
        assert cols[1] == "cell"
        assert cols[2] == "layer"
        assert cols[3] == "cell2d"
        assert "col" not in cols

    def test_write_sout_headers_no_duplicate_cell_when_phreeqc_has_it(self, mock_mf6rtm):
        """PHREEQC 'cell' is excluded — our spatial cell replaces it, time_d stays first."""
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time_d", "cell", "pH"]
        so = SelectedOutput(mock_mf6rtm)
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(2, 3, 4)):
            so._write_sout_headers()
        header_line = open(os.path.join(mock_mf6rtm.wd, "sout.csv")).readline().strip()
        cols = header_line.split(",")
        assert cols.count("cell") == 1
        assert cols[0] == "time_d"
        assert cols[1] == "cell"
        assert cols[2] == "layer"
        assert cols[3] == "row"
        assert cols[4] == "col"


# HDF5 output and CSV append

class TestAppendToSoutFile:
    def test_csv_append(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        so._current_soutdf = pd.DataFrame({"time_d": [1.0], "pH": [7.0]})
        so._append_to_soutdf_file()
        path = os.path.join(mock_mf6rtm.wd, "sout.csv")
        assert os.path.exists(path)
        assert "7.0" in open(path).read()

    def test_hdf5_append(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="sout.h5")
        so._current_soutdf = pd.DataFrame({"time_d": [1.0], "pH": [7.0]})
        so._append_to_soutdf_file()
        df = pd.read_hdf(os.path.join(mock_mf6rtm.wd, "sout.h5"), key="sout")
        assert len(df) == 1
        assert "pH" in df.columns

    def test_hdf5_multiple_appends_accumulate(self, mock_mf6rtm):
        pytest.importorskip("tables")
        so = SelectedOutput(mock_mf6rtm, sout_fname="out.hdf5")
        for t in [1.0, 2.0, 3.0]:
            so._current_soutdf = pd.DataFrame({"time_d": [t], "pH": [7.0]})
            so._append_to_soutdf_file()
        df = pd.read_hdf(os.path.join(mock_mf6rtm.wd, "out.hdf5"), key="sout")
        assert len(df) == 3

    def test_rm_sout_file(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        path = os.path.join(mock_mf6rtm.wd, "sout.csv")
        open(path, "w").close()
        so._rm_sout_file()
        assert not os.path.exists(path)

    def test_rm_sout_file_missing_no_error(self, mock_mf6rtm):
        so = SelectedOutput(mock_mf6rtm)
        so._rm_sout_file()  # must not raise

    def test_write_sout_headers_skipped_for_hdf5(self, mock_mf6rtm):
        pytest.importorskip("tables")
        mock_mf6rtm.phreeqcbmi.sout_headers = ["time_d", "pH"]
        so = SelectedOutput(mock_mf6rtm, sout_fname="sout.h5")
        with patch("mf6rtm.io.externalio.grid_dimensions", return_value=(1, 2, 3)):
            so._write_sout_headers()
        assert not os.path.exists(os.path.join(mock_mf6rtm.wd, "sout.h5"))


# Regenerator: PHREEQC keyword-block parsing / classification / reassembly

# A synthetic phinp exercising every category, including the tricky cases:
#   - SOLUTION_SPECIES / KNOBS with no trailing END (definitions)
#   - SOLUTION vs SOLUTION_SPECIES, EXCHANGE vs EXCHANGE_SPECIES (exact-token match)
#   - SURFACE preserved verbatim; EQUILIBRIUM_PHASES/KINETICS/EXCHANGE dropped
#   - SELECTED_OUTPUT/USER_PUNCH with column-0 data lines (-reset, numbered PUNCH)
_SYNTHETIC_PHINP = """\
SOLUTION_MASTER_SPECIES
    Foo Foo 0.0 Foo 1.0
SOLUTION_SPECIES
    H2O + 0.01e- = H2O-0.01
    log_k -9.0
KNOBS
    iterations 500
SOLUTION 1
    temp 25.0
    pH 7.0
END
SOLUTION 2
    temp 10.0
END
EQUILIBRIUM_PHASES 1
    Calcite 0.0 1.0
END
EXCHANGE 1
    X 0.1
END
KINETICS 1
    Pyrite
        -m0 1.0
END
SURFACE 1
    Hfo_w Goethite equilibrium_phase 0.2 600.0
    -equilibrate 1
END
PRINT
    -reset false
END
SELECTED_OUTPUT
-reset         false
-file          out.dat
USER_PUNCH
    -headings time_d cell pH
1   PUNCH sim_time/86400
2   PUNCH cell_no
END
"""


def _make_regenerator(wd, phinp_text, config=None):
    """Build a Regenerator that bypasses the heavy __init__ (no live model/dll)."""
    reg = Regenerator.__new__(Regenerator)
    reg.wd = wd
    reg.phinp = "phinp.dat"
    reg.config = config if config is not None else {"reactive": {}, "output": {}}
    with open(os.path.join(wd, "phinp.dat"), "w") as f:
        f.write(phinp_text)
    return reg


@pytest.fixture
def tmp_wd():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestRegeneratorBlockParsing:
    def test_is_block_start_distinguishes_exact_keywords(self):
        f = Regenerator._is_block_start
        assert f("SOLUTION 1\n") == "SOLUTION"
        assert f("SOLUTION_SPECIES\n") == "SOLUTION_SPECIES"
        assert f("EXCHANGE 1\n") == "EXCHANGE"
        assert f("EXCHANGE_SPECIES\n") == "EXCHANGE_SPECIES"
        assert f("SURFACE 1\n") == "SURFACE"

    def test_is_block_start_rejects_data_end_blank(self):
        f = Regenerator._is_block_start
        assert f("    pH 7.0\n") is None        # indented data
        assert f("END\n") is None               # END attaches to current block
        assert f("\n") is None                  # blank
        assert f("-reset false\n") is None      # leading dash
        assert f("1   PUNCH x\n") is None        # leading digit

    def test_classify_block(self):
        c = Regenerator._classify_block
        assert c("EQUILIBRIUM_PHASES") == "regenerated"
        assert c("KINETICS") == "regenerated"
        assert c("EXCHANGE") == "regenerated"
        assert c("SOLUTION") == "solution"
        assert c("SURFACE") == "preserved"
        assert c("GAS_PHASE") == "preserved"
        assert c("SELECTED_OUTPUT") == "output"
        assert c("USER_PUNCH") == "output"
        assert c("KNOBS") == "definition"
        assert c("SOLUTION_SPECIES") == "definition"
        assert c("SOLUTION_MASTER_SPECIES") == "definition"

    def test_split_into_blocks_carries_trailing_end(self, tmp_wd):
        reg = _make_regenerator(tmp_wd, _SYNTHETIC_PHINP)
        blocks = reg._split_into_blocks(_SYNTHETIC_PHINP.splitlines(keepends=True))
        kws = [kw for kw, _ in blocks]
        assert kws == [
            "SOLUTION_MASTER_SPECIES", "SOLUTION_SPECIES", "KNOBS",
            "SOLUTION", "SOLUTION",
            "EQUILIBRIUM_PHASES", "EXCHANGE", "KINETICS",
            "SURFACE", "PRINT", "SELECTED_OUTPUT", "USER_PUNCH",
        ]
        # SOLUTION 1 block keeps its own END
        sol1 = next(lines for kw, lines in blocks if kw == "SOLUTION")
        assert any(l.strip() == "END" for l in sol1)


class TestRegeneratorReassembly:
    def test_definitions_before_solution_and_output_once(self, tmp_wd):
        reg = _make_regenerator(tmp_wd, _SYNTHETIC_PHINP)
        script = reg.generate_new_script()
        starts = [
            l.split()[0] for l in script.splitlines()
            if Regenerator._is_block_start(l) is not None
        ]
        # definitions emitted at the top, before the first SOLUTION
        assert starts.index("SOLUTION_SPECIES") < starts.index("SOLUTION")
        assert starts.index("KNOBS") < starts.index("SOLUTION")
        assert starts.index("SOLUTION_MASTER_SPECIES") < starts.index("SOLUTION")
        # output blocks appear exactly once (no duplication)
        assert starts.count("SELECTED_OUTPUT") == 1
        assert starts.count("USER_PUNCH") == 1

    def test_regenerated_blocks_dropped_preserved_kept(self, tmp_wd):
        # config has no *_phases keys -> generators are skipped, so any
        # EQUILIBRIUM_PHASES/KINETICS/EXCHANGE in the output would be leftovers.
        reg = _make_regenerator(tmp_wd, _SYNTHETIC_PHINP)
        script = reg.generate_new_script()
        starts = [
            l.split()[0] for l in script.splitlines()
            if Regenerator._is_block_start(l) is not None
        ]
        assert "EQUILIBRIUM_PHASES" not in starts
        assert "KINETICS" not in starts
        assert "EXCHANGE" not in starts
        # SURFACE preserved verbatim, in the reaction section
        assert "SURFACE" in starts
        assert "-equilibrate 1" in script
        assert "Hfo_w" in script
        last_solution = max(i for i, s in enumerate(starts) if s == "SOLUTION")
        assert starts.index("SURFACE") > last_solution
        assert starts.index("SURFACE") < starts.index("SELECTED_OUTPUT")

    def test_definition_group_terminated_by_end(self, tmp_wd):
        reg = _make_regenerator(tmp_wd, _SYNTHETIC_PHINP)
        script = reg.generate_new_script()
        lines = [l for l in script.splitlines()]
        # the defensive END must sit between the last definition and the first SOLUTION
        knobs_i = next(i for i, l in enumerate(lines) if l.strip() == "KNOBS")
        sol_i = next(i for i, l in enumerate(lines) if l.strip().startswith("SOLUTION ")
                     and "SPECIES" not in l and "MASTER" not in l)
        assert any(lines[i].strip() == "END" for i in range(knobs_i, sol_i))

    def test_no_merge_when_last_block_lacks_trailing_newline(self, tmp_wd):
        """A definition block at EOF with no trailing newline must not merge with the
        defensive END (regression for '-diagonal_scale trueEND')."""
        phinp = (
            "SOLUTION 1\n    pH 7.0\nEND\n"
            "SELECTED_OUTPUT\n-file out.dat\n"
            "KNOBS\n    iterations 500\n    -diagonal_scale true"  # no trailing newline
        )
        reg = _make_regenerator(tmp_wd, phinp)
        script = reg.generate_new_script()
        assert "trueEND" not in script
        lines = [l.strip() for l in script.splitlines()]
        # KNOBS moved to the top as a definition, terminated by its own END line
        assert lines[0] == "KNOBS"
        assert "-diagonal_scale true" in lines
        diag_i = lines.index("-diagonal_scale true")
        assert lines[diag_i + 1] == "END"

"""Unit tests for externalio.py.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from mf6rtm.io.externalio import SelectedOutput


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

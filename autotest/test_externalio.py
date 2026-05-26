"""Unit tests for externalio.py.
"""
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

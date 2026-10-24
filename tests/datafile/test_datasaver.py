"""Regression tests for datafile path helpers.

The dict save/load layer (save_data / load_data / save_local_data /
load_local_data) is gone (see ADR-0063 and lib/zcu_tools/datafile/README.md); persistence is native labber_io. These
tests cover reserve_labber_filepath and format_ext.
"""

from __future__ import annotations

import zcu_tools.datafile as datafile
from zcu_tools.datafile import (
    format_ext,
    reserve_labber_filepath,
)

# ---------------------------------------------------------------------------
# extension helpers
# ---------------------------------------------------------------------------


def test_format_ext_normalizes_to_hdf5():
    assert format_ext("data") == "data.hdf5"
    assert format_ext("data.h5") == "data.hdf5"
    assert format_ext("data.hdf5") == "data.hdf5"


def test_format_ext_only_rewrites_suffix():
    assert format_ext("run.h5_backup/data.h5") == "run.h5_backup/data.hdf5"
    assert format_ext("run.h5_backup/data") == "run.h5_backup/data.hdf5"


# ---------------------------------------------------------------------------
# reserve_labber_filepath: unique-name suffixing
# ---------------------------------------------------------------------------


def test_reserve_labber_filepath_appends_suffix_and_ext(tmp_path):
    """First call yields the _1 suffix with .hdf5 extension."""
    out = reserve_labber_filepath(str(tmp_path / "run"))
    assert out == str(tmp_path / "run_1.hdf5")


def test_reserve_labber_filepath_increments_past_existing(tmp_path):
    """An existing _1 file forces the next call to bump to _2."""
    (tmp_path / "run_1.hdf5").write_bytes(b"")
    out = reserve_labber_filepath(str(tmp_path / "run"))
    assert out == str(tmp_path / "run_2.hdf5")


def test_reserve_labber_filepath_preserves_isolated_numeric_suffix(tmp_path):
    out = reserve_labber_filepath(str(tmp_path / "data_2024.hdf5"))

    assert out == str(tmp_path / "data_2024.hdf5")


def test_reserve_labber_filepath_does_not_increment_year_suffix(tmp_path):
    (tmp_path / "data_2024.hdf5").write_bytes(b"")

    out = reserve_labber_filepath(str(tmp_path / "data_2024.hdf5"))

    assert out == str(tmp_path / "data_2024_1.hdf5")


def test_legacy_reservation_alias_is_not_exported():
    legacy_name = "safe" + "_labber_filepath"
    assert not hasattr(datafile, legacy_name)

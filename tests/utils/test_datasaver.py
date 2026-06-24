"""Regression tests for datasaver path helpers.

The dict save/load layer (save_data / load_data / save_local_data /
load_local_data) is gone (ADR-0027); persistence is native labber_io. These
tests cover the surviving path helpers: safe_labber_filepath / format_ext /
remove_ext.
"""

from __future__ import annotations

from zcu_tools.utils.datasaver import (
    format_ext,
    remove_ext,
    safe_labber_filepath,
)

# ---------------------------------------------------------------------------
# extension helpers
# ---------------------------------------------------------------------------


def test_format_ext_normalizes_to_hdf5():
    assert format_ext("data") == "data.hdf5"
    assert format_ext("data.h5") == "data.hdf5"
    assert format_ext("data.hdf5") == "data.hdf5"


def test_remove_ext_strips_known_extensions():
    assert remove_ext("data.h5") == "data"
    assert remove_ext("data.hdf5") == "data"
    assert remove_ext("data") == "data"


# ---------------------------------------------------------------------------
# safe_labber_filepath: unique-name suffixing
# ---------------------------------------------------------------------------


def test_safe_labber_filepath_appends_suffix_and_ext(tmp_path):
    """First call yields the _1 suffix with .hdf5 extension."""
    out = safe_labber_filepath(str(tmp_path / "run"))
    assert out == str(tmp_path / "run_1.hdf5")


def test_safe_labber_filepath_increments_past_existing(tmp_path):
    """An existing _1 file forces the next call to bump to _2."""
    (tmp_path / "run_1.hdf5").write_bytes(b"")
    out = safe_labber_filepath(str(tmp_path / "run"))
    assert out == str(tmp_path / "run_2.hdf5")

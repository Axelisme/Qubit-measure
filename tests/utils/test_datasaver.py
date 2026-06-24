"""Regression tests for datasaver path/transport helpers + load_comment.

The dict save/load layer (save_data / load_data / save_local_data /
load_local_data) is gone (ADR-0027); persistence is native labber_io. These
tests cover the surviving helpers: load_comment (a thin read over labber_io,
constructed here via save_labber_data) and safe_labber_filepath / format_ext /
remove_ext.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.utils.datasaver import (
    format_ext,
    load_comment,
    remove_ext,
    safe_labber_filepath,
)
from zcu_tools.utils.labber_io import save_labber_data

# ---------------------------------------------------------------------------
# load_comment round-trip (file built via the native labber_io saver)
# ---------------------------------------------------------------------------


def _write_labber_file(path: str, comment: str) -> None:
    freq = np.linspace(4e9, 5e9, 11)
    z = np.ones(11, dtype=complex)
    save_labber_data(
        format_ext(path),
        z=("S21", "arb", z),
        axes=[("Frequency", "Hz", freq)],
        comment=comment,
    )


def test_load_comment(tmp_path):
    """load_comment reads back the comment written into a labber file."""
    path = str(tmp_path / "comment_test")
    _write_labber_file(path, "hello world")

    assert load_comment(path) == "hello world"


def test_load_comment_missing_file(tmp_path):
    """load_comment warns and returns None (does not raise) on a missing file.

    The UserWarning is the documented contract under failure, so assert it
    explicitly rather than letting it leak into the suite's warning summary.
    """
    with pytest.warns(UserWarning, match="Failed to load comment"):
        result = load_comment(str(tmp_path / "nonexistent"))
    assert result is None


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

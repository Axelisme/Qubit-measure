"""Tests for the friendly load/restore/export error messages.

_friendly_io_message maps the low-level exceptions these IO actions raise (h5py /
OSError / a bare AssertionError from a wrong file format) to a human message with
a fix hint, keeping the raw error on a "Details:" line. Pure function — no Qt.
"""

from __future__ import annotations

from zcu_tools.fluxdep_gui.ui.error_messages import (
    friendly_fit_message,
)
from zcu_tools.fluxdep_gui.ui.error_messages import (
    friendly_io_message as _friendly_io_message,
)


def test_missing_file():
    exc = FileNotFoundError(
        "[Errno 2] Unable to synchronously open file (unable to open file: "
        "name = '/x/y.hdf5')"
    )
    msg = _friendly_io_message("Load", "/x/y.hdf5", exc)
    assert "File not found" in msg
    assert "y.hdf5" in msg
    assert "Details:" in msg  # raw error preserved


def test_not_an_hdf5(tmp_path):
    f = tmp_path / "bogus.hdf5"
    f.write_text("not hdf5")
    exc = OSError("Unable to synchronously open file (file signature not found)")
    msg = _friendly_io_message("Load", str(f), exc)
    assert "not a valid HDF5 file" in msg


def test_restore_raw_file_hint_says_use_add(tmp_path):
    f = tmp_path / "raw.hdf5"
    f.write_text("x")
    # load_spectrums raises a bare AssertionError on a non-processed file
    msg = _friendly_io_message("Restore", str(f), AssertionError())
    assert "not a processed spectrums.hdf5" in msg
    assert "use Add instead of Restore" in msg
    # a bare AssertionError still leaves a Details line with the type name
    assert "AssertionError" in msg


def test_export_unwritable():
    exc = FileNotFoundError(
        "[Errno 2] Unable to synchronously create file (unable to open file: "
        "name = '/proc/x.hdf5')"
    )
    msg = _friendly_io_message("Export", "/proc/x.hdf5", exc)
    assert "Could not write" in msg
    assert "not writable" in msg


def test_export_empty_collection():
    msg = _friendly_io_message(
        "Export", "/tmp/x.hdf5", ValueError("no spectra to export")
    )
    assert "Nothing to export" in msg


def test_no_frequency_axis():
    msg = _friendly_io_message(
        "Load", "/tmp/x.hdf5", ValueError("'/tmp/x.hdf5' has no frequency axis")
    )
    assert "not a 2D spectrum" in msg


def test_unknown_error_falls_back_gracefully():
    msg = _friendly_io_message("Load", "/tmp/weird.hdf5", RuntimeError("boom"))
    assert "Load of 'weird.hdf5' failed" in msg
    assert "boom" in msg  # raw error still shown


# --- fit (search / export params) messages ---------------------------------


def test_fit_no_database():
    msg = friendly_fit_message("Search", "no database path set (call set_params first)")
    assert "Select a database" in msg


def test_fit_mirror_needs_sample_f():
    msg = friendly_fit_message("Search", "sample_f is required for mirror transitions")
    assert "sample_f" in msg
    assert "mirror" in msg


def test_fit_no_candidate():
    msg = friendly_fit_message(
        "Search",
        "No valid candidate found in database (all parameter bounds infeasible).",
    )
    assert "No match found" in msg
    assert "Widen" in msg


def test_fit_export_no_result():
    msg = friendly_fit_message(
        "Export", ValueError("no fit result to export (run search first)")
    )
    assert "run a search first" in msg


def test_fit_export_no_aligned():
    msg = friendly_fit_message(
        "Export", ValueError("no aligned spectrum (align one before exporting)")
    )
    assert "Align at least one spectrum" in msg


def test_fit_accepts_both_string_and_exception():
    # search crosses a worker signal as a string; export catches the exception
    from_str = friendly_fit_message("Search", "no selected points to fit")
    from_exc = friendly_fit_message("Search", ValueError("no selected points to fit"))
    assert "select points" in from_str.lower()
    assert from_str.split("\n")[0] == from_exc.split("\n")[0]

"""Tests for the dispersive LoadOnetoneDialog (file + transpose + preview)."""

from __future__ import annotations

from zcu_tools.gui.app.dispersive.ui.load_dialog import (
    LoadOnetoneDialog,
    LoadOnetoneRequest,
)


def test_dialog_starts_with_no_file_and_disabled_ok(qapp):
    dialog = LoadOnetoneDialog(start_dir="")
    assert dialog.result_request() is None  # no file chosen
    assert dialog._ok_button is not None
    assert dialog._ok_button.isEnabled() is False


def test_result_request_after_file_chosen(qapp, onetone_hdf5):
    path, *_ = onetone_hdf5
    dialog = LoadOnetoneDialog(start_dir="")
    # simulate a browse: set the file + read preview the way _on_browse does
    dialog._filepath = path
    dialog._raw = dialog._read_best_effort(path)
    req = dialog.result_request()
    assert isinstance(req, LoadOnetoneRequest)
    assert req.filepath == path
    assert req.transpose_axes is False


def test_transpose_toggle_flips_request(qapp, onetone_hdf5):
    path, *_ = onetone_hdf5
    dialog = LoadOnetoneDialog(start_dir="")
    dialog._filepath = path
    dialog._raw = dialog._read_best_effort(path)
    dialog._on_transpose_toggled(True)
    req = dialog.result_request()
    assert req is not None and req.transpose_axes is True


def test_preview_read_best_effort_handles_bad_file(qapp, tmp_path):
    bad = str(tmp_path / "not_an_hdf5.h5")
    with open(bad, "w") as f:
        f.write("garbage")
    dialog = LoadOnetoneDialog(start_dir="")
    # best-effort: a bad file yields None, the dialog does not raise
    assert dialog._read_best_effort(bad) is None
    dialog._raw = None
    dialog._render_preview()  # blank fallback, no raise


def test_preview_renders_for_real_file(qapp, onetone_hdf5):
    path, *_ = onetone_hdf5
    dialog = LoadOnetoneDialog(start_dir="")
    dialog._raw = dialog._read_best_effort(path)
    assert dialog._raw is not None
    dialog._render_preview()  # imshow path, no raise
    # toggling transpose re-renders without raising
    dialog._on_transpose_toggled(True)

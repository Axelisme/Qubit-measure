"""Tests for the dispersive file-selection and transpose request contract."""

from __future__ import annotations

from qtpy.QtWidgets import QDialogButtonBox, QFileDialog, QPushButton
from zcu_tools.gui.app.dispersive.ui.load_dialog import (
    LoadOnetoneDialog,
    LoadOnetoneRequest,
)


def _button(dialog: LoadOnetoneDialog, text: str) -> QPushButton:
    return next(
        button for button in dialog.findChildren(QPushButton) if button.text() == text
    )


def test_dialog_starts_with_no_file_and_disabled_ok(qapp):
    dialog = LoadOnetoneDialog(start_dir="")
    assert dialog.result_request() is None
    box = dialog.findChildren(QDialogButtonBox)[0]
    button = box.button(QDialogButtonBox.StandardButton.Ok)
    assert button is not None and not button.isEnabled()


def test_result_request_after_file_chosen(qapp, onetone_hdf5, monkeypatch):
    path, *_ = onetone_hdf5
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (path, "")
    )
    dialog = LoadOnetoneDialog(start_dir="")
    _button(dialog, "Browse…").click()
    assert dialog.result_request() == LoadOnetoneRequest(path, False)


def test_transpose_toggle_flips_request(qapp, onetone_hdf5, monkeypatch):
    path, *_ = onetone_hdf5
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (path, "")
    )
    dialog = LoadOnetoneDialog(start_dir="")
    _button(dialog, "Browse…").click()
    _button(dialog, "Transpose axes").click()
    assert dialog.result_request() == LoadOnetoneRequest(path, True)


def test_bad_preview_does_not_prevent_file_selection(qapp, tmp_path, monkeypatch):
    bad = str(tmp_path / "not_an_hdf5.h5")
    with open(bad, "w") as f:
        f.write("garbage")
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: (bad, "")
    )
    dialog = LoadOnetoneDialog(start_dir="")
    _button(dialog, "Browse…").click()
    assert dialog.result_request() == LoadOnetoneRequest(bad, False)

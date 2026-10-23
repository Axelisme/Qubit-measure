"""Headless tests for LoadSpectrumDialog (preview + transpose + inherit)."""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy.QtWidgets import QComboBox, QDialogButtonBox, QFileDialog, QPushButton
from zcu_tools.gui.app.fluxdep.ui.load_dialog import LoadSpectrumDialog

from tests.gui._user_input import ComponentDriver


@pytest.fixture
def dialog(qapp):
    d = LoadSpectrumDialog({"existing.hdf5": (0.5, 1.5)})
    yield d
    d.deleteLater()


def test_preview_axis_labels_follow_transpose_and_restore(
    dialog, spectrum_hdf5, qapp, tmp_path: Path
):
    dialog.show()
    qapp.processEvents()
    driver = ComponentDriver(dialog)
    driver.choose_file("Browse…", Path(spectrum_hdf5[0]))
    driver.screenshot(tmp_path / "original.png")
    original = driver.plots().axes[0]
    assert (original.xlabel, original.ylabel) == ("Device value (flux)", "Frequency")

    driver.click("Transpose axes")
    driver.screenshot(tmp_path / "transposed.png")
    transposed = driver.plots().axes[0]
    assert (transposed.xlabel, transposed.ylabel) == (
        "Frequency",
        "Device value (flux)",
    )

    driver.click("Transpose axes")
    driver.screenshot(tmp_path / "restored.png")
    assert driver.plots().axes[0] == original


def test_builds_with_no_file(dialog):
    # no file chosen → OK disabled, no request
    assert dialog.result_request() is None
    boxes = dialog.findChildren(QDialogButtonBox)
    ok = next(
        b.button(QDialogButtonBox.StandardButton.Ok)
        for b in boxes
        if b.button(QDialogButtonBox.StandardButton.Ok) is not None
    )
    assert ok is not None and not ok.isEnabled()


def test_inherit_dropdown_lists_loaded_names(dialog):
    inherit = next(
        c for c in dialog.findChildren(QComboBox) if c.findText("existing.hdf5") >= 0
    )
    items = [inherit.itemText(i) for i in range(inherit.count())]
    assert items == ["(none)", "existing.hdf5"]
    # default selection is (none) → None
    assert inherit.currentData() is None


def test_result_request_reflects_choices(dialog, spectrum_hdf5, monkeypatch):
    filepath = spectrum_hdf5[0]
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: (filepath, ""))
    buttons = dialog.findChildren(QPushButton)
    next(b for b in buttons if b.text() == "Browse…").click()
    for combo in dialog.findChildren(QComboBox):
        combo.setCurrentText(
            "TwoTone" if combo.findText("TwoTone") >= 0 else "existing.hdf5"
        )
    next(b for b in buttons if b.isCheckable()).click()

    req = dialog.result_request()
    assert req is not None
    assert req.filepath == filepath
    assert req.spec_type == "TwoTone"
    assert req.transpose_axes is True
    assert req.inherit_from == "existing.hdf5"

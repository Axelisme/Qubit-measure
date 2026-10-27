"""Tests for the dispersive file-selection and transpose request contract."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialogButtonBox, QFileDialog, QPushButton
from zcu_tools.gui.app.analysis_project.native_dialog import NativeDeviceChoices
from zcu_tools.gui.app.dispersive.ui.load_dialog import (
    LoadOnetoneDialog,
    LoadOnetoneRequest,
)

from tests.gui._user_input import ComponentDriver
from tests.gui.app.analysis_project._native import native_spectrum_file


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


def test_pasted_path_loads_on_focus_loss_and_clearing_removes_selection(
    qapp, onetone_hdf5, tmp_path
):
    path, *_ = onetone_hdf5
    qapp.clipboard().setText(path)
    dialog = LoadOnetoneDialog()
    try:
        dialog.resize(1100, 700)
        dialog.show()
        qapp.processEvents()
        driver = ComponentDriver(dialog)
        driver.enter_text("", value="", commit=False)
        driver.press_key(Qt.Key.Key_V, modifiers=Qt.KeyboardModifier.ControlModifier)
        driver.screenshot(tmp_path / "pasted-path.png")
        driver.press_key(Qt.Key.Key_Tab)
        driver.screenshot(tmp_path / "loaded-path.png")
        assert driver.plots().axes[0].ylabel == "Frequency"
        assert dialog.result_request() == LoadOnetoneRequest(path, False)
        driver.press_key(Qt.Key.Key_Backtab)
        driver.enter_text("", value="", commit=False)
        driver.press_key(Qt.Key.Key_Tab)
        driver.screenshot(tmp_path / "cleared-path.png")
        assert dialog.result_request() is None
        assert not driver.observe("OK").enabled
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


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
    dialog.show()
    qapp.processEvents()
    driver = ComponentDriver(dialog)
    driver.choose_option("X=Flux, Y=Frequency", value="X=Frequency, Y=Flux")
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


def test_native_choices_come_from_predictor_and_are_editable(qapp, tmp_path):
    path = native_spectrum_file(tmp_path / "native.h5", {})
    dialog = LoadOnetoneDialog(
        start_dir=str(tmp_path),
        choices=NativeDeviceChoices({"bias": "A", "coil": "V"}, default_device="bias"),
    )
    dialog.show()
    qapp.processEvents()
    driver = ComponentDriver(dialog)
    try:
        driver.screenshot(tmp_path / "before.png")
        driver.choose_file("Browse…", path, evidence=tmp_path / "chooser.png")
        driver.screenshot(tmp_path / "preselected.png")
        assert driver.observe(object_name="native_variable").text == "signal"
        assert driver.observe(object_name="native_device_axis").text == "bias"
        assert driver.observe(object_name="native_device").text == "bias"
        assert driver.observe("OK").enabled
        driver.select_combo("coil", object_name="native_device")
        driver.screenshot(tmp_path / "selected-coil.png")
        assert driver.observe(object_name="native_device").text == "coil"
        assert driver.observe("V").visible
    finally:
        driver.click("Cancel")
        dialog.deleteLater()
        qapp.processEvents()

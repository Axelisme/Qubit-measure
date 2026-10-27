"""Headless tests for LoadSpectrumDialog (preview + transpose + inherit)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QDialogButtonBox, QFileDialog, QPushButton
from zcu_tools.gui.app.fluxdep.ui.load_dialog import LoadSpectrumDialog
from zcu_tools.gui.app.fluxdep.ui.main_window import MainWindow

from tests.gui._user_input import ComponentDriver


@pytest.fixture
def dialog(qapp):
    d = LoadSpectrumDialog({"existing.hdf5": (0.5, 1.5)})
    yield d
    d.deleteLater()


@pytest.fixture
def preview_dialog(qapp, spectrum_hdf5, monkeypatch):
    """Open Browse at the visible fixture directory used to resolve driver input."""
    with (
        TemporaryDirectory(prefix="spectrum-chooser-", dir=Path.cwd()) as directory,
        monkeypatch.context() as context,
    ):
        context.chdir(directory)
        filename = Path("spectrum with spaces.hdf5")
        copyfile(spectrum_hdf5[0], filename)
        d = LoadSpectrumDialog({"existing.hdf5": (0.5, 1.5)}, start_dir=directory)
        try:
            yield d, filename
        finally:
            d.deleteLater()
            qapp.processEvents()


def test_preview_axis_labels_follow_transpose_and_restore(
    preview_dialog, qapp, tmp_path: Path
):
    dialog, filename = preview_dialog
    dialog.show()
    qapp.processEvents()
    driver = ComponentDriver(dialog)
    driver.choose_file("Browse…", filename, evidence=tmp_path / "chooser.png")
    driver.screenshot(tmp_path / "original.png")
    original = driver.plots().axes[0]
    assert (original.xlabel, original.ylabel) == ("Device value (flux)", "Frequency")

    driver.choose_option("X=Flux, Y=Frequency", value="X=Frequency, Y=Flux")
    driver.screenshot(tmp_path / "transposed.png")
    transposed = driver.plots().axes[0]
    assert (transposed.xlabel, transposed.ylabel) == (
        "Frequency",
        "Device value (flux)",
    )

    driver.choose_option("X=Frequency, Y=Flux", value="X=Flux, Y=Frequency")
    driver.screenshot(tmp_path / "restored.png")
    assert driver.plots().axes[0] == original


@pytest.fixture
def inheritance_window(qapp, cross_controller):
    # Include aligned-only, seeded-only and fresh identity-default sources.
    for name, aligned, seeded in (
        ("a", True, False),
        ("b", False, True),
        ("empty", False, False),
    ):
        cross_controller.state.put_spectrum(
            replace(
                cross_controller.state.get_spectrum(name),
                aligned=aligned,
                alignment_seeded=seeded,
            )
        )
    window = MainWindow(cross_controller)
    window.show()
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_add_spectrum_offers_only_aligned_or_seeded_inheritance(
    inheritance_window, tmp_path
):
    driver = ComponentDriver(inheritance_window)
    driver.click("Add spectrum…")
    try:
        driver.screenshot(tmp_path / "inheritance-none.png")
        driver.choose_option("(none)", "a")
        assert driver.observe("a").text == "a"
        driver.screenshot(tmp_path / "inheritance-aligned.png")
        driver.choose_option("a", "b")
        assert driver.observe("b").text == "b"
        driver.screenshot(tmp_path / "inheritance-seeded.png")
        driver.click("b")
        driver.screenshot(tmp_path / "inheritance-options.png")
        driver.press_key(Qt.Key.Key_Escape)
        with pytest.raises(LookupError):
            driver.choose_option("b", "empty")
        assert driver.observe("b").text == "b"
    finally:
        driver.click("Cancel")


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


def test_result_request_reflects_choices(dialog, qapp, spectrum_hdf5, monkeypatch):
    filepath = spectrum_hdf5[0]
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **kw: (filepath, ""))
    buttons = dialog.findChildren(QPushButton)
    next(b for b in buttons if b.text() == "Browse…").click()
    for combo in dialog.findChildren(QComboBox):
        combo.setCurrentText(
            "TwoTone" if combo.findText("TwoTone") >= 0 else "existing.hdf5"
        )
    dialog.show()
    qapp.processEvents()
    ComponentDriver(dialog).choose_option(
        "X=Flux, Y=Frequency", value="X=Frequency, Y=Flux"
    )

    req = dialog.result_request()
    assert req is not None
    assert req.filepath == filepath
    assert req.spec_type == "TwoTone"
    assert req.transpose_axes is True
    assert req.inherit_from == "existing.hdf5"

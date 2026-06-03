"""Headless tests for ExportSpectrumsDialog (chip/qubit → dynamic default path)."""

from __future__ import annotations

import os

import pytest
from zcu_tools.fluxdep_gui.ui.export_dialog import ExportSpectrumsDialog


@pytest.fixture
def dialog(qapp):
    d = ExportSpectrumsDialog()
    yield d
    d.deleteLater()


def test_default_names_and_path(dialog):
    assert dialog._chip_edit.text() == "unknown_chip"
    assert dialog._qub_edit.text() == "unknown_qubit"
    assert dialog.export_path() == os.path.join(
        "result", "unknown_chip", "unknown_qubit", "data", "fluxdep", "spectrums.hdf5"
    )


def test_changing_names_updates_path(dialog):
    dialog._chip_edit.setText("Q5_2D")
    dialog._qub_edit.setText("Q1")
    assert dialog.export_path() == os.path.join(
        "result", "Q5_2D", "Q1", "data", "fluxdep", "spectrums.hdf5"
    )


def test_initial_names_from_project(qapp):
    d = ExportSpectrumsDialog(chip_name="Chip", qub_name="Qub")
    assert d.export_path() == os.path.join(
        "result", "Chip", "Qub", "data", "fluxdep", "spectrums.hdf5"
    )
    d.deleteLater()


def test_manual_path_override_detaches_from_names(dialog):
    dialog._path_edit.setText("/tmp/custom.hdf5")
    dialog._on_path_edited("/tmp/custom.hdf5")  # simulate textEdited
    dialog._chip_edit.setText("Q5_2D")  # should NOT overwrite the manual path
    assert dialog.export_path() == "/tmp/custom.hdf5"

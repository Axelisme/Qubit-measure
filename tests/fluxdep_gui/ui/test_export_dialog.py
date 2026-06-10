"""Headless tests for ExportSpectrumsDialog.

The dialog no longer edits chip/qubit (those move to the Project dialog) — it
just shows the default path derived from the project's chip/qubit and lets the
user override it.
"""

from __future__ import annotations

import os

import pytest
from zcu_tools.gui.app.fluxdep.ui.export_dialog import ExportSpectrumsDialog


@pytest.fixture
def dialog(qapp):
    d = ExportSpectrumsDialog(os.path.join("result", "unknown_chip", "unknown_qubit"))
    yield d
    d.deleteLater()


def test_default_path_under_result_dir(dialog):
    assert dialog.export_path() == os.path.join(
        "result", "unknown_chip", "unknown_qubit", "data", "fluxdep", "spectrums.hdf5"
    )


def test_default_path_from_project_result_dir(qapp):
    d = ExportSpectrumsDialog(os.path.join("result", "Q5_2D", "Q1"))
    assert d.export_path() == os.path.join(
        "result", "Q5_2D", "Q1", "data", "fluxdep", "spectrums.hdf5"
    )
    d.deleteLater()


def test_path_can_be_overridden(dialog):
    dialog._path_edit.setText("/tmp/custom.hdf5")
    assert dialog.export_path() == "/tmp/custom.hdf5"


def test_no_chip_qub_edit_fields(dialog):
    # the chip/qubit editing moved to the Project dialog
    assert not hasattr(dialog, "_chip_edit")
    assert not hasattr(dialog, "_qub_edit")

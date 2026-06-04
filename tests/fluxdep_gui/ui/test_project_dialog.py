"""Headless tests for ProjectDialog (chip/qubit + optional roots)."""

from __future__ import annotations

import pytest
from zcu_tools.fluxdep_gui.state import ProjectInfo
from zcu_tools.fluxdep_gui.ui.project_dialog import ProjectDialog


@pytest.fixture
def dialog(qapp):
    d = ProjectDialog(ProjectInfo(chip_name="Q5_2D", qub_name="Q1"))
    yield d
    d.deleteLater()


def test_prefills_from_project(dialog):
    assert dialog._chip_edit.text() == "Q5_2D"
    assert dialog._qub_edit.text() == "Q1"


def test_result_project_returns_edited_values(dialog):
    dialog._chip_edit.setText("Chip2")
    dialog._qub_edit.setText("Q3")
    dialog._result_edit.setText("/r")
    dialog._database_edit.setText("/db")
    project = dialog.result_project()
    assert project.chip_name == "Chip2"
    assert project.qub_name == "Q3"
    assert project.result_dir == "/r"
    assert project.database_path == "/db"


def test_names_are_trimmed(dialog):
    dialog._chip_edit.setText("  spacey  ")
    assert dialog.result_project().chip_name == "spacey"

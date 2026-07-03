"""Headless tests for ProjectDialog (chip/qubit + optional roots)."""

from __future__ import annotations

import pytest
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.gui.widgets.project_dialog import ProjectDialog


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


def test_result_dir_auto_derives_from_names(qapp):
    import os

    d = ProjectDialog(ProjectInfo())
    d._chip_edit.setText("Q5_2D")
    d._qub_edit.setText("Q1")
    assert d._result_edit.text() == os.path.join("result", "Q5_2D", "Q1")
    d.deleteLater()


def test_auto_derivation_anchors_at_project_root_dir(qapp):
    import os

    # A project carrying a root_dir (injected repo root) makes the dialog's
    # auto-derivation anchor there, not at the cwd-relative default.
    root = os.path.join(os.sep, "repo")
    d = ProjectDialog(ProjectInfo(root_dir=root))
    d._chip_edit.setText("Q5_2D")
    d._qub_edit.setText("Q1")
    assert d._result_edit.text() == os.path.join(root, "result", "Q5_2D", "Q1")
    # result_project carries the root_dir forward so re-derivation stays anchored.
    assert d.result_project().root_dir == root
    d.deleteLater()


def test_manual_result_dir_stops_auto_derivation(qapp):
    d = ProjectDialog(ProjectInfo())
    d._result_edit.setText("/custom/out")
    d._on_result_edited("/custom/out")  # simulate textEdited
    d._chip_edit.setText("Q5_2D")  # must NOT overwrite the manual dir
    assert d._result_edit.text() == "/custom/out"
    d.deleteLater()


def test_database_path_auto_derives_from_names(qapp):
    import os

    d = ProjectDialog(ProjectInfo())
    d._chip_edit.setText("Q5_2D")
    d._qub_edit.setText("Q1")
    # the raw-spectrum root tracks chip/qubit, but under Database/ (not result/)
    assert d._database_edit.text() == os.path.join("Database", "Q5_2D", "Q1")
    d.deleteLater()


def test_manual_database_path_stops_auto_derivation(qapp):
    d = ProjectDialog(ProjectInfo())
    d._database_edit.setText("/custom/raw")
    d._on_database_edited("/custom/raw")  # simulate textEdited
    d._chip_edit.setText("Q5_2D")  # must NOT overwrite the manual db path
    assert d._database_edit.text() == "/custom/raw"
    d.deleteLater()


def test_browse_buttons_exist(dialog):
    from qtpy.QtWidgets import QPushButton

    labels = [b.text() for b in dialog.findChildren(QPushButton)]
    # a Browse… for result dir and one for database path
    assert labels.count("Browse…") == 2


def test_result_scope_dropdown_lists_discovered_params(qapp, tmp_path):
    from zcu_tools.meta_tool import ParamsProject, QubitParams

    params_path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    QubitParams(params_path).ensure_project(ParamsProject("ChipA", "Q1"))

    d = ProjectDialog(ProjectInfo(root_dir=str(tmp_path)), project_root=str(tmp_path))
    try:
        assert d._scope_combo.findData(str(params_path.parent.resolve())) >= 0
    finally:
        d.deleteLater()


def test_selecting_result_scope_updates_names_and_paths(qapp, tmp_path):
    import os

    from zcu_tools.meta_tool import ParamsProject, QubitParams

    params_path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    QubitParams(params_path).ensure_project(ParamsProject("ChipA", "Q1"))

    d = ProjectDialog(
        ProjectInfo(chip_name="Other", qub_name="Q2", root_dir=str(tmp_path)),
        project_root=str(tmp_path),
    )
    try:
        idx = d._scope_combo.findData(str(params_path.parent.resolve()))
        assert idx >= 0

        d._scope_combo.setCurrentIndex(idx)

        assert d._chip_edit.text() == "ChipA"
        assert d._qub_edit.text() == "Q1"
        assert d._result_edit.text() == str(params_path.parent.resolve())
        assert d._database_edit.text() == os.path.join(
            str(tmp_path), "Database", "ChipA", "Q1"
        )
    finally:
        d.deleteLater()

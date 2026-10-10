"""Headless tests for ProjectDialog (chip/qubit + optional roots)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit, QPushButton
from zcu_tools.gui.project import ProjectInfo
from zcu_tools.gui.result_scope import (
    ResultScope,
    ResultScopeManager,
    write_params_identity,
)
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
    from zcu_tools.resources.qubit_params import ParamsProject, QubitParams

    params_path = tmp_path / "result" / "ChipA" / "Q1" / "params.json"
    QubitParams(params_path).ensure_project(ParamsProject("ChipA", "Q1"))

    d = ProjectDialog(ProjectInfo(root_dir=str(tmp_path)), project_root=str(tmp_path))
    try:
        assert d._scope_combo.findData(str(params_path.parent.resolve())) >= 0
    finally:
        d.deleteLater()


def test_selecting_result_scope_updates_names_and_paths(qapp, tmp_path):
    import os

    from zcu_tools.resources.qubit_params import ParamsProject, QubitParams

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


@pytest.fixture
def scope_dialog(
    qapp: QApplication,
) -> Iterator[Callable[[ProjectInfo], ProjectDialog]]:
    dialogs: list[ProjectDialog] = []

    def create(project: ProjectInfo) -> ProjectDialog:
        widget = ProjectDialog(project)
        dialogs.append(widget)
        return widget

    yield create
    for widget in dialogs:
        widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def scope_catalog(tmp_path: Path) -> tuple[ResultScope, ...]:
    for chip, qub in (("Previous", "QP"), ("Result", "QR"), ("Names", "QN")):
        write_params_identity(
            tmp_path / "result" / chip / qub / "params.json",
            chip_name=chip,
            qub_name=qub,
        )
    return ResultScopeManager(tmp_path).list_scopes()


def _scope_picker(widget: ProjectDialog) -> QComboBox:
    picker = widget.findChild(QComboBox)
    assert picker is not None
    return picker


def _edit_with_text(widget: ProjectDialog, text: str) -> QLineEdit:
    return next(edit for edit in widget.findChildren(QLineEdit) if edit.text() == text)


def test_scope_dropdown_preserves_previous_selection_over_result_and_names(
    scope_dialog: Callable[[ProjectInfo], ProjectDialog],
    scope_catalog: tuple[ResultScope, ...],
    tmp_path: Path,
):
    previous = next(scope for scope in scope_catalog if scope.chip_name == "Previous")
    result = next(scope for scope in scope_catalog if scope.chip_name == "Result")
    names = next(scope for scope in scope_catalog if scope.chip_name == "Names")
    project = ProjectInfo(
        chip_name="Initial",
        qub_name="Q0",
        result_dir=previous.result_dir,
        database_path=str(tmp_path / "custom-database"),
        root_dir=str(tmp_path),
    )
    widget = scope_dialog(project)
    picker = _scope_picker(widget)
    assert picker.currentData() == previous.scope_id

    _edit_with_text(widget, previous.result_dir).setText(result.result_dir)
    _edit_with_text(widget, "Initial").setText(names.chip_name)
    _edit_with_text(widget, "Q0").setText(names.qub_name)

    assert picker.currentData() == previous.scope_id
    assert widget.result_project() == ProjectInfo(
        chip_name=names.chip_name,
        qub_name=names.qub_name,
        result_dir=result.result_dir,
        database_path=project.database_path,
        root_dir=project.root_dir,
    )


def test_scope_dropdown_matches_result_before_names(
    scope_dialog: Callable[[ProjectInfo], ProjectDialog],
    scope_catalog: tuple[ResultScope, ...],
    tmp_path: Path,
):
    result = next(scope for scope in scope_catalog if scope.chip_name == "Result")
    names = next(scope for scope in scope_catalog if scope.chip_name == "Names")
    project = ProjectInfo(
        chip_name=names.chip_name,
        qub_name=names.qub_name,
        result_dir=result.result_dir,
        root_dir=str(tmp_path),
    )
    widget = scope_dialog(project)

    assert _scope_picker(widget).currentData() == result.scope_id
    assert widget.result_project() == project


def test_scope_dropdown_matches_names_after_unmatched_result(
    scope_dialog: Callable[[ProjectInfo], ProjectDialog],
    scope_catalog: tuple[ResultScope, ...],
    tmp_path: Path,
):
    names = next(scope for scope in scope_catalog if scope.chip_name == "Names")
    project = ProjectInfo(
        chip_name=names.chip_name,
        qub_name=names.qub_name,
        result_dir=str(tmp_path / "unmatched-result"),
        root_dir=str(tmp_path),
    )
    widget = scope_dialog(project)

    assert _scope_picker(widget).currentData() == names.scope_id
    assert widget.result_project() == project


@pytest.mark.parametrize("discovery", ["discovered", "empty"])
def test_scope_dropdown_fallback_keeps_unnamed_project(
    scope_dialog: Callable[[ProjectInfo], ProjectDialog],
    tmp_path: Path,
    discovery: str,
):
    params_path = tmp_path / "result" / "Discovered" / "Q1" / "params.json"
    if discovery == "discovered":
        write_params_identity(params_path, chip_name="Discovered", qub_name="Q1")
    project = ProjectInfo(
        chip_name="",
        qub_name="",
        result_dir=str(tmp_path / "custom-result"),
        database_path=str(tmp_path / "custom-database"),
        root_dir=str(tmp_path),
    )
    widget = scope_dialog(project)
    picker = _scope_picker(widget)

    assert picker.count() == 1
    if discovery == "discovered":
        assert picker.currentData() == str(params_path.parent.resolve())
    else:
        assert picker.currentData() is None
        assert picker.currentText() == "(no result scopes found)"
    assert widget.result_project() == project


@pytest.mark.parametrize("checked", [False, True], ids=["unchecked", "checked"])
def test_scope_refresh_accepts_clicked_check_state(
    scope_dialog: Callable[[ProjectInfo], ProjectDialog],
    tmp_path: Path,
    checked: bool,
):
    existing_path = tmp_path / "result" / "Existing" / "Q1" / "params.json"
    write_params_identity(existing_path, chip_name="Existing", qub_name="Q1")
    project = ProjectInfo(chip_name="Initial", qub_name="Q0", root_dir=str(tmp_path))
    widget = scope_dialog(project)
    picker = _scope_picker(widget)
    new_path = tmp_path / "result" / "New" / "Q2" / "params.json"
    new_scope_id = str(new_path.parent.resolve())
    assert picker.findData(str(existing_path.parent.resolve())) >= 0
    assert picker.findData(new_scope_id) < 0
    write_params_identity(new_path, chip_name="New", qub_name="Q2")

    refresh = next(
        button
        for button in widget.findChildren(QPushButton)
        if button.toolTip() == "Refresh result scopes"
    )
    refresh.clicked.emit(checked)

    assert widget.result_project() == project
    new_index = picker.findData(new_scope_id)
    assert new_index >= 0
    picker.setCurrentIndex(new_index)
    assert widget.result_project() == ProjectInfo(
        chip_name="New",
        qub_name="Q2",
        result_dir=new_scope_id,
        root_dir=str(tmp_path),
    )

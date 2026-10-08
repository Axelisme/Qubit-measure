"""Artifact-level commands and per-output evidence at the Qt save-center seam."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton
from zcu_tools.experiment.run_saving import ArtifactRef, OutputResult, SaveReport
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactDeclaration,
    ArtifactTracker,
)
from zcu_tools.gui.app.measure.services.ports import TabSnapshot
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.app.measure.ui.artifact_save_center import ArtifactSaveCenter
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.resources.entry import ArtifactKey, Output, SaveLayout

_CAPABILITIES = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
_REFS = (
    ArtifactRef("run", "measurement_data"),
    ArtifactRef("analysis", "results"),
    ArtifactRef("post_analysis", "results"),
)


def snapshot(tmp_path: Path) -> tuple[ArtifactTracker, TabSnapshot]:
    """Create declarations using the production planner, without payload or I/O."""
    tracker = ArtifactTracker()
    layout = SaveLayout(
        result_path=tmp_path / "Result",
        database_path=tmp_path / "Database",
        run_id="20261008T000000Z-a1b2c3",
        point="p",
        saved_at=datetime(2026, 10, 8, tzinfo=timezone.utc),
    )
    for ref in _REFS:
        keys = (
            (ArtifactKey(ref.section, ref.name, "data"),)
            if ref.section == "run"
            else (
                (ArtifactKey(ref.section, ref.name, "figure", "fit"),)
                if ref.section == "analysis"
                else ()
            )
            + (ArtifactKey(ref.section, ref.name, "analysis", ref.section),)
        )
        tracker.replace_section(
            ref.section, (ArtifactDeclaration(ref, keys),), layout=layout
        )
    return tracker, TabSnapshot(
        adapter_name="saving",
        cfg_schema=CfgSchema(CfgSectionSpec(), CfgSectionValue()),
        tab_id="tab",
        capabilities=_CAPABILITIES,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_active_context=True,
            has_context=True,
            has_soc=False,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        ),
        artifacts=tracker.snapshot(),
    )


def test_all_output_paths_are_read_only_and_refresh_retains_selection(
    qapp: QApplication, tmp_path: Path
) -> None:
    _, current = snapshot(tmp_path)
    center = ArtifactSaveCenter(_CAPABILITIES)
    try:
        center.update_from_snapshot(current)
        center.show()
        qapp.processEvents()
        paths = [
            str(output.path)
            for section in current.artifacts
            for artifact in section.artifacts
            for member in artifact.members
            for output in member.outputs
        ]
        editors = center.findChildren(QLineEdit)
        assert len(editors) == len(paths) == 6
        assert [editor.text() for editor in editors] == paths
        for editor, path in zip(editors, paths, strict=True):
            editor.setFocus()
            editor.selectAll()
            QApplication.sendEvent(
                editor,
                QKeyEvent(
                    QEvent.Type.KeyPress,
                    Qt.Key.Key_X,
                    Qt.KeyboardModifier.NoModifier,
                    "x",
                ),
            )
            assert editor.text() == path
            editor.selectAll()
            center.update_from_snapshot(current)
            assert editor.selectedText() == path
        assert center.findChildren(QLineEdit) == editors
    finally:
        center.deleteLater()
        qapp.processEvents()


def test_outputs_follow_actual_report_paths_and_errors(
    qapp: QApplication, tmp_path: Path
) -> None:
    tracker, current = snapshot(tmp_path)
    center = ArtifactSaveCenter(_CAPABILITIES)
    try:
        center.update_from_snapshot(current)
        editors = center.findChildren(QLineEdit)
        results = tuple(
            OutputResult(
                member.key,
                Output(
                    output.format,
                    tmp_path
                    / "actual"
                    / f"{index}-{member.key.member}-{output.path.name}",
                ),
                "failed" if index == 1 else "saved",
                "copy denied" if index == 1 else None,
            )
            for section in current.artifacts
            for artifact in section.artifacts
            for member in artifact.members
            for index, output in enumerate(member.outputs)
        )
        tracker.apply_report(SaveReport("20261008T000000Z-a1b2c3", results))
        center.update_from_snapshot(replace(current, artifacts=tracker.snapshot()))
        assert center.findChildren(QLineEdit) == editors
        assert [editor.text() for editor in editors] == [
            str(result.output.path) for result in results
        ]
        assert any(
            "copy denied" in label.text() for label in center.findChildren(QLabel)
        )
        assert center.has_unsaved_data()
    finally:
        center.deleteLater()
        qapp.processEvents()


def test_whole_artifact_commands_include_json_only_and_saved_items(
    qapp: QApplication, tmp_path: Path
) -> None:
    tracker, current = snapshot(tmp_path)
    tracker.apply_report(
        SaveReport(
            "20261008T000000Z-a1b2c3",
            tuple(
                OutputResult(
                    member.key, Output(output.format, output.path), "saved", None
                )
                for section in current.artifacts
                for artifact in section.artifacts
                for member in artifact.members
                for output in member.outputs
            ),
        )
    )
    current = replace(current, artifacts=tracker.snapshot())
    selected: list[ArtifactRef] = []
    all_requests: list[bool] = []
    center = ArtifactSaveCenter(_CAPABILITIES)
    center.bind_artifact_save(selected.append)
    center.bind_save_all(lambda: all_requests.append(True))
    try:
        center.update_interaction(current)
        assert not center.has_unsaved_data()
        assert center.save_all_button.isEnabled()
        buttons = [
            button
            for button in center.findChildren(QPushButton)
            if button.text() == "Save"
        ]
        assert len(buttons) == 3
        for button in buttons:
            button.click()
        center.save_all_button.click()
        assert selected == list(_REFS)
        assert all_requests == [True]
        assert current.interaction is not None
        center.update_interaction(
            replace(
                current, interaction=replace(current.interaction, is_saving_data=True)
            )
        )
        assert not center.save_all_button.isEnabled()
        assert all(not button.isEnabled() for button in buttons)
        center.save_all_button.click()
        assert all_requests == [True]
        center.update_interaction(replace(current, artifacts=()))
        qapp.processEvents()
        assert not center.save_all_button.isEnabled()
        assert not center.has_unsaved_data()
    finally:
        center.deleteLater()
        qapp.processEvents()

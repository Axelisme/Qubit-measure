"""Artifact-level commands and per-output evidence at the Qt save-center seam."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import QEvent, Qt, QUrl
from qtpy.QtGui import QDesktopServices, QKeyEvent
from qtpy.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton
from zcu_tools.experiment.run_saving import ArtifactRef, OutputResult, SaveReport
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactDeclaration,
    ArtifactTracker,
)
from zcu_tools.gui.app.measure.services.ports import (
    AnalysisPaneSnapshot,
    PostAnalysisPaneSnapshot,
    RunPaneSnapshot,
    SavePaneSnapshot,
    TabSnapshot,
)
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.app.measure.ui.artifact_save_center import ArtifactSaveCenter
from zcu_tools.resources.entry import ArtifactKey, Output, SaveLayout

from tests.gui.app.measure.ui._artifact_snapshots import ready_figures, with_artifacts

_CAPABILITIES = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)
_REFS = (
    ArtifactRef("run", "measurement_data"),
    ArtifactRef("analysis", "results"),
    ArtifactRef("post_analysis", "results"),
)


@pytest.mark.parametrize("has_figures, output_count", [(False, 4), (True, 8)])
def test_pane_specimens_render_data_json_and_named_figure_outputs(
    qapp: QApplication, tmp_path: Path, has_figures: bool, output_count: int
) -> None:
    """Project view fixtures through the shipped Qt seam, including JSON-only panes."""
    _, current = snapshot(tmp_path)
    current = with_artifacts(
        replace(
            current,
            run=RunPaneSnapshot(result=object(), source_path=None),
            analysis=AnalysisPaneSnapshot(
                params=None,
                result=object(),
                figures=ready_figures(Figure() if has_figures else None),
                writeback_items=(),
            ),
            post_analysis=PostAnalysisPaneSnapshot(
                params=None,
                result=object(),
                figures=ready_figures(Figure() if has_figures else None),
                writeback_items=(),
            ),
        )
    )
    selected: list[ArtifactRef] = []
    center = ArtifactSaveCenter(_CAPABILITIES)
    center.bind_artifact_save(selected.append)
    try:
        center.update_interaction(current)
        paths = [edit for edit in center.findChildren(QLineEdit) if edit.isReadOnly()]
        assert len(paths) == output_count
        assert all(Path(edit.text()).is_absolute() for edit in paths)
        labels = [label.text() for label in center.findChildren(QLabel)]
        assert all(Path(edit.toolTip()).name in labels for edit in paths)
        if has_figures:
            assert sum(label.endswith(".png") for label in labels) == 4
        assert sum("NOT SAVED" in label for label in labels) == 3
        assert center.has_unsaved_data()
        assert center.save_all_button.isEnabled()
        buttons = [
            button
            for button in center.findChildren(QPushButton)
            if button.text() == "Save"
        ]
        assert len(buttons) == 3
        for button in buttons:
            button.click()
        assert selected == list(_REFS)
    finally:
        center.deleteLater()
        qapp.processEvents()


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
        cfg_input={},
        tab_id="tab",
        capabilities=_CAPABILITIES,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            has_active_context=True,
            has_context=True,
            has_soc=False,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        ),
        save=SavePaneSnapshot(),
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
        assert [editor.toolTip() for editor in editors] == paths
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
            assert editor.toolTip() == path
            editor.selectAll()
            center.update_from_snapshot(current)
            # Selection covers the shown (possibly elided) text; the tooltip keeps the path.
            assert editor.selectedText() == editor.text()
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
        assert [editor.toolTip() for editor in editors] == [
            str(result.output.path) for result in results
        ]
        assert any(
            "copy denied" in label.text() for label in center.findChildren(QLabel)
        )
        assert center.has_unsaved_data()
    finally:
        center.deleteLater()
        qapp.processEvents()


def test_folder_actions_follow_latest_outputs_without_creating_directories(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker, current = snapshot(tmp_path / "預留 空間")
    opened: list[QUrl] = []

    def open_url(url: QUrl) -> bool:
        opened.append(QUrl(url))
        return True

    monkeypatch.setattr(QDesktopServices, "openUrl", open_url)
    center = ArtifactSaveCenter(_CAPABILITIES)
    try:
        center.update_from_snapshot(replace(current, artifacts=()))
        assert not any(
            button.text() == "Open folder"
            for button in center.findChildren(QPushButton)
        )
        assert opened == []
        center.update_from_snapshot(current)
        paths = tuple(
            output.path
            for section in current.artifacts
            for artifact in section.artifacts
            for member in artifact.members
            for output in member.outputs
        )
        folders = [
            button
            for button in center.findChildren(QPushButton)
            if button.text() == "Open folder"
        ]
        assert len(folders) == len(paths)
        for button in folders:
            button.click()
        assert all(url.isLocalFile() for url in opened)
        assert [url.toLocalFile() for url in opened] == [
            str(path.parent) for path in paths
        ]
        assert all(not path.parent.exists() for path in paths)
        outputs = tuple(
            OutputResult(
                member.key,
                Output(
                    output.format,
                    tmp_path / "實際 存檔" / str(index) / output.path.name,
                ),
                "saved",
                None,
            )
            for section in current.artifacts
            for artifact in section.artifacts
            for member in artifact.members
            for index, output in enumerate(member.outputs)
        )
        tracker.apply_report(SaveReport("20261008T000000Z-a1b2c3", outputs))
        center.update_from_snapshot(replace(current, artifacts=tracker.snapshot()))
        opened.clear()
        for button in folders:
            button.click()
        assert all(url.isLocalFile() for url in opened)
        assert [url.toLocalFile() for url in opened] == [
            str(item.output.path.parent) for item in outputs
        ]
        assert all(not item.output.path.parent.exists() for item in outputs)
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
        center.update_interaction(replace(current, artifacts=()))
        qapp.processEvents()
        assert not center.save_all_button.isEnabled()
        assert not center.has_unsaved_data()
    finally:
        center.deleteLater()
        qapp.processEvents()

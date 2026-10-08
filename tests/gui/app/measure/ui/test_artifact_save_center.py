"""Path display and folder actions at the production save-center seam."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from qtpy.QtCore import QEvent, Qt, QUrl
from qtpy.QtGui import QDesktopServices, QKeyEvent
from qtpy.QtWidgets import QApplication, QLineEdit, QPushButton
from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    ArtifactObservation,
    ArtifactTracker,
)
from zcu_tools.gui.app.measure.services.ports import TabSnapshot
from zcu_tools.gui.app.measure.ui.artifact_save_center import ArtifactSaveCenter
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue

_DATA = ArtifactKey(ArtifactKind.DATA)
_FIT = ArtifactKey(ArtifactKind.ANALYSIS, "fit")
_POST = ArtifactKey(ArtifactKind.POST_ANALYSIS, "fit")
_CAPABILITIES = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=True)


def _snapshot(paths: tuple[Path | None, ...]) -> TabSnapshot:
    tracker = ArtifactTracker()
    artifacts = tracker.project(
        tuple(
            ArtifactObservation(
                key=key,
                result=object(),
                figure=object() if key.kind is not ArtifactKind.DATA else None,
                path=str(path) if path is not None else None,
            )
            for key, path in zip((_DATA, _FIT, _POST), paths, strict=True)
        )
    )
    return TabSnapshot(
        adapter_name="fake",
        cfg_schema=CfgSchema(CfgSectionSpec(), CfgSectionValue()),
        tab_id="tab-1",
        capabilities=_CAPABILITIES,
        artifacts=artifacts,
    )


def test_paths_reject_keyboard_edits_and_follow_published_snapshots(
    qapp: QApplication, tmp_path: Path
) -> None:
    paths = tuple(tmp_path / name for name in ("data.h5", "fit.png", "post.png"))
    center = ArtifactSaveCenter("tab-1", _CAPABILITIES)
    try:
        snapshot = _snapshot(paths)
        center.update_from_snapshot(snapshot)
        center.show()
        qapp.processEvents()
        for path in paths:
            editor = next(
                edit
                for edit in center.findChildren(QLineEdit)
                if edit.text() == str(path)
            )
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
            assert editor.text() == str(path)
            editor.selectAll()
            center.update_from_snapshot(snapshot)
            assert editor.selectedText() == str(path)
        updated = tuple(tmp_path / "new" / path.name for path in paths)
        center.update_from_snapshot(_snapshot(updated))
        assert tuple(center.path_for(key) for key in (_DATA, _FIT, _POST)) == tuple(
            str(path) for path in updated
        )
        center.set_comment_text("still editable")
        assert center.get_comment() == "still editable"
    finally:
        center.deleteLater()
        qapp.processEvents()


def test_folder_actions_use_latest_published_paths_without_creating_directories(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[QUrl] = []

    def open_url(url: QUrl) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(QDesktopServices, "openUrl", open_url)
    paths = tuple(
        tmp_path / "條目 with spaces" / name
        for name in ("data.h5", "fit.png", "post.png")
    )
    center = ArtifactSaveCenter("tab-1", _CAPABILITIES)
    try:
        center.update_from_snapshot(_snapshot(paths))
        buttons = [
            button
            for button in center.findChildren(QPushButton)
            if button.text() == "Open folder"
        ]
        assert len(buttons) == len(paths)
        for button in buttons:
            button.click()
        assert [url.toLocalFile() for url in opened] == [
            str(path.parent) for path in paths
        ]
        assert all(not path.parent.exists() for path in paths)
        updated = tuple(tmp_path / "new" / path.name for path in paths)
        center.update_from_snapshot(_snapshot(updated))
        opened.clear()
        for button in buttons:
            button.click()
        assert [url.toLocalFile() for url in opened] == [
            str(path.parent) for path in updated
        ]
        snapshot = _snapshot(updated)
        empty = replace(
            snapshot,
            artifacts=tuple(
                replace(artifact, default_path=None) for artifact in snapshot.artifacts
            ),
        )
        center.update_from_snapshot(empty)
        opened.clear()
        for button in buttons:
            button.click()
        assert opened == []
    finally:
        center.deleteLater()
        qapp.processEvents()

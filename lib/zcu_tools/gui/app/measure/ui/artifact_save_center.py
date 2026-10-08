"""Data save center for current named artifacts and their published paths."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QUrl  # type: ignore[attr-defined]
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.adapter import AnalysisMode
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactKey,
    ArtifactKind,
    ArtifactSnapshot,
    SaveStatus,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities
    from zcu_tools.gui.app.measure.services import TabSnapshot

_DATA = ArtifactKey(ArtifactKind.DATA)
_STATUS_TEXT = {
    SaveStatus.NO_RESULT: "— NO RESULT",
    SaveStatus.NOT_SAVED: "○ NOT SAVED",
    SaveStatus.UNSAVED_CHANGES: "● UNSAVED CHANGES",
    SaveStatus.SAVED: "✓ SAVED",
}
_STATUS_COLOR = {
    SaveStatus.NO_RESULT: "#7b2cbf",
    SaveStatus.NOT_SAVED: "#3f3f3f",
    SaveStatus.UNSAVED_CHANGES: "#c45100",
    SaveStatus.SAVED: "#0067c0",
}


class ArtifactSaveCenter(QWidget):
    """Render DATA and only the named images currently published by State.

    Image rows are keyed by (stage, name), not by stage. Published paths are
    read-only and can be selected or copied. Updating save status retains the
    widgets; a pane replacement retires only its old rows.
    """

    def __init__(
        self,
        tab_id: str,
        capabilities: AdapterCapabilities,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tab_id = tab_id
        self._has_analysis = capabilities.analysis is not AnalysisMode.NONE
        self._has_post = bool(capabilities.post_analysis)
        self._has_load = bool(capabilities.load_data)
        self._snapshots: dict[ArtifactKey, ArtifactSnapshot] = {}
        self._rows: dict[ArtifactKey, QWidget] = {}
        self._status_labels: dict[ArtifactKey, QLabel] = {}
        self._path_edits: dict[ArtifactKey, QLineEdit] = {}
        self._save_btns: dict[ArtifactKey, QPushButton] = {}
        self._image_save_handler: Callable[[ArtifactKey], None] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        outer.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]
        heading = QLabel("Save results")
        font = heading.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        heading.setFont(font)
        outer.addWidget(heading)
        detail = QLabel("Only current outputs can be saved here.")
        detail.setWordWrap(True)
        detail.setStyleSheet("color: #666;")
        outer.addWidget(detail)

        outer.addWidget(self._build_actions())

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(8)
        outer.addLayout(self._rows_layout)
        self._rows[_DATA] = self._build_row(
            _DATA, "Measurement data", with_comment=True
        )
        self._rows_layout.addWidget(self._rows[_DATA])
        outer.addStretch()

    def _build_actions(self) -> QWidget:
        actions = QWidget()
        actions.setObjectName("dataActions")
        layout = QHBoxLayout(actions)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.load_button = QPushButton("Load Data")
        self.load_button.setFixedHeight(36)
        self.load_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.save_all_button = QPushButton("Save All")
        self.save_all_button.setFixedHeight(36)
        self.save_all_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.save_all_button.setDefault(True)
        if self._has_load:
            layout.addWidget(self.load_button, stretch=1)
            layout.addWidget(self.save_all_button, stretch=1)
        else:
            layout.addWidget(self.save_all_button, stretch=1)
            self.load_button.hide()
        return actions

    def _build_row(
        self, key: ArtifactKey, title: str, *, with_comment: bool = False
    ) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        header = QHBoxLayout()
        title_label = QLabel(title)
        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)
        header.addWidget(title_label)
        header.addStretch()
        status = QLabel()
        sf = status.font()
        sf.setBold(True)
        status.setFont(sf)
        status.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        header.addWidget(status)
        layout.addLayout(header)
        self._status_labels[key] = status

        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText(
            "/tmp/data.hdf5" if key.kind is ArtifactKind.DATA else "/tmp/image.png"
        )
        path_row.addWidget(edit, stretch=1)
        self._path_edits[key] = edit
        open_folder = QPushButton("Open folder")
        open_folder.setToolTip("Open the published output's parent folder")
        open_folder.clicked.connect(lambda _checked=False, k=key: self._open_folder(k))
        path_row.addWidget(open_folder)
        save = QPushButton("Save")
        save.setFixedWidth(72)
        if key.kind is not ArtifactKind.DATA:
            save.clicked.connect(lambda _checked=False, k=key: self._image_save(k))
        path_row.addWidget(save)
        self._save_btns[key] = save
        layout.addLayout(path_row)

        if with_comment:
            self._comment_edit = QTextEdit()
            self._comment_edit.setPlaceholderText("Optional comment…")
            self._comment_edit.setFixedHeight(60)
            self._comment_edit.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            layout.addWidget(self._comment_edit)
        return container

    def _image_save(self, key: ArtifactKey) -> None:
        if self._image_save_handler is not None:
            self._image_save_handler(key)

    def _open_folder(self, key: ArtifactKey) -> None:
        path = self._path_edits[key].text()
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def get_data_path(self) -> str:
        return self._path_edits[_DATA].text()

    def path_for(self, key: ArtifactKey) -> str:
        return self._path_edits[key].text()

    def get_comment(self) -> str:
        return self._comment_edit.toPlainText()

    def set_data_path(self, path: str) -> None:
        self._path_edits[_DATA].setText(path)

    def set_image_path(self, key: ArtifactKey, path: str) -> None:
        if key.kind is ArtifactKind.DATA:
            raise ValueError("Use set_data_path for DATA")
        self._path_edits[key].setText(path)

    def set_comment_text(self, text: str) -> None:
        if self._comment_edit.toPlainText() != text:
            blocked = self._comment_edit.blockSignals(True)
            try:
                self._comment_edit.setPlainText(text)
            finally:
                self._comment_edit.blockSignals(blocked)

    def bind_comment_changed(self, handler: Callable[[str], None]) -> None:
        self._comment_edit.textChanged.connect(lambda: handler(self.get_comment()))

    def bind_image_save(self, handler: Callable[[ArtifactKey], None]) -> None:
        self._image_save_handler = handler

    def bind_save_data(self, handler: Callable[[], None]) -> None:
        self._save_btns[_DATA].clicked.connect(lambda _checked=False: handler())

    def bind_save_all(self, handler: Callable[[], None]) -> None:
        self.save_all_button.clicked.connect(lambda _checked=False: handler())

    def bind_load(self, handler: Callable[[], None]) -> None:
        if self._has_load:
            self.load_button.clicked.connect(lambda _checked=False: handler())

    @property
    def artifact_keys(self) -> list[ArtifactKey]:
        return list(self._rows)

    def has_artifact(self, key: ArtifactKey) -> bool:
        return key in self._rows

    def is_save_enabled(self, key: ArtifactKey) -> bool:
        button = self._save_btns.get(key)
        return bool(button is not None and button.isEnabled())

    def is_save_all_enabled(self) -> bool:
        return self.save_all_button.isEnabled()

    def is_load_enabled(self) -> bool:
        return bool(self._has_load and self.load_button.isEnabled())

    def is_load_visible(self) -> bool:
        return self._has_load and not self.load_button.isHidden()

    def is_path_enabled(self, key: ArtifactKey) -> bool:
        edit = self._path_edits.get(key)
        return bool(edit is not None and edit.isEnabled())

    def has_unsaved_data(self) -> bool:
        data = self._snapshots.get(_DATA)
        return data is not None and data.status in (
            SaveStatus.NOT_SAVED,
            SaveStatus.UNSAVED_CHANGES,
        )

    def update_from_snapshot(self, snapshot: TabSnapshot) -> None:
        artifacts = {artifact.key: artifact for artifact in snapshot.artifacts}
        if _DATA not in artifacts:
            raise RuntimeError("State must project the DATA artifact")
        images = [key for key in artifacts if key.kind is not ArtifactKind.DATA]
        for key in images:
            if (key.kind is ArtifactKind.ANALYSIS and not self._has_analysis) or (
                key.kind is ArtifactKind.POST_ANALYSIS and not self._has_post
            ):
                raise RuntimeError(
                    f"Unexpected artifact {key!r} for tab {self._tab_id!r}"
                )
        for key in tuple(self._rows):
            if key is _DATA or key in artifacts:
                continue
            widget = self._rows.pop(key)
            self._rows_layout.removeWidget(widget)
            widget.deleteLater()
            self._status_labels.pop(key)
            self._path_edits.pop(key)
            self._save_btns.pop(key)
        for index, key in enumerate(images, start=1):
            if key not in self._rows:
                stage = (
                    "Analysis" if key.kind is ArtifactKind.ANALYSIS else "Post-analysis"
                )
                self._rows[key] = self._build_row(key, f"{stage}: {key.figure_name}")
            self._rows_layout.insertWidget(index, self._rows[key])
        self._snapshots = artifacts
        for key, artifact in artifacts.items():
            label = self._status_labels[key]
            label.setText(_STATUS_TEXT[artifact.status])
            label.setStyleSheet(f"color: {_STATUS_COLOR[artifact.status]};")
            # No figure gets a fake artifact row; every image path is per-name.
            self._path_edits[key].setText(artifact.default_path or "")

    def update_interaction(self, snapshot: TabSnapshot) -> None:
        if snapshot.interaction is None or snapshot.capabilities is None:
            raise RuntimeError("Save center needs a live tab snapshot")
        self.update_from_snapshot(snapshot)
        state = snapshot.interaction
        idle = not (state.is_running or state.is_analyzing or state.is_saving_data)
        for key, artifact in self._snapshots.items():
            self._save_btns[key].setEnabled(
                idle and state.has_active_context and artifact.is_saveable
            )
        if self._has_load:
            self.load_button.setEnabled(idle and state.has_context)
        self.save_all_button.setEnabled(
            idle
            and state.has_active_context
            and any(artifact.needs_save for artifact in self._snapshots.values())
        )

    def status_text(self, key: ArtifactKey) -> str:
        return self._status_labels[key].text()

    def status_color(self, key: ArtifactKey) -> str:
        style = self._status_labels[key].styleSheet()
        return style.split("color:", 1)[-1].strip().strip(";").strip()

    def save_button(self, key: ArtifactKey) -> QPushButton:
        return self._save_btns[key]

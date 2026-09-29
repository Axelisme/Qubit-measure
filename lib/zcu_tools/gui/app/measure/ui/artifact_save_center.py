"""Data save center for current named artifacts and their path drafts."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QEvent, Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFileDialog,
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


class _FocusPreservingSaveAllButton(QPushButton):
    """Keep the DATA editor's focus and selection across a Save All click."""

    def __init__(
        self,
        capture_editor_state: Callable[[], None],
        restore_editor_state: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Save All", parent)
        self._capture_editor_state = capture_editor_state
        self._restore_editor_state = restore_editor_state

    def mousePressEvent(self, e: Any) -> None:
        self._capture_editor_state()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: Any) -> None:
        try:
            super().mouseReleaseEvent(e)
        finally:
            self._restore_editor_state()


class ArtifactSaveCenter(QWidget):
    """Render DATA and only the named images currently published by State.

    Image rows are keyed by (stage, name), not by stage. Updating save status
    retains the editor widgets; a pane replacement retires only its old rows.
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
        self._image_path_handler: Callable[[ArtifactKey, str], None] | None = None
        self._image_save_handler: Callable[[ArtifactKey], None] | None = None
        self._image_path_slots: dict[ArtifactKey, Callable[[str], None]] = {}
        self._local_path_edits: set[ArtifactKey] = set()
        self._saved_data_editor_state: (
            tuple[QLineEdit, str, int, int, int, bool] | None
        ) = None

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
        self._path_edits[_DATA].installEventFilter(self)

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
        self.save_all_button = _FocusPreservingSaveAllButton(
            self._capture_data_editor_state, self._restore_data_editor_state
        )
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
        edit.setPlaceholderText(
            "/tmp/data.hdf5" if key.kind is ArtifactKind.DATA else "/tmp/image.png"
        )
        path_row.addWidget(edit, stretch=1)
        self._path_edits[key] = edit
        if key.kind is not ArtifactKind.DATA:

            def slot(text: str, *, bound_key: ArtifactKey = key) -> None:
                self._image_path_changed(bound_key, text)

            edit.textChanged.connect(slot)
            self._image_path_slots[key] = slot
        browse = QPushButton("Browse…")
        browse.setFixedWidth(80)
        browse.setToolTip("Choose a destination")
        browse.clicked.connect(lambda _checked=False, k=key: self._on_browse(k))
        path_row.addWidget(browse)
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

    def _image_path_changed(self, key: ArtifactKey, text: str) -> None:
        if self._image_path_handler is not None:
            self._image_path_handler(key, text)

    def _image_save(self, key: ArtifactKey) -> None:
        if self._image_save_handler is not None:
            self._image_save_handler(key)

    def _on_browse(self, key: ArtifactKey) -> None:
        data = key.kind is ArtifactKind.DATA
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save data file" if data else "Save image file",
            "",
            "HDF5 files (*.hdf5);;All files (*)"
            if data
            else "PNG files (*.png);;All files (*)",
        )
        if path:
            self._path_edits[key].setText(path)

    def _remember_data_editor_state(self, edit: QLineEdit) -> None:
        start = edit.selectionStart()
        self._saved_data_editor_state = (
            edit,
            edit.text(),
            edit.cursorPosition(),
            start,
            edit.selectionLength(),
            start >= 0 and edit.cursorPosition() == start,
        )

    @staticmethod
    def _restore_cursor_and_selection(
        edit: QLineEdit,
        cursor: int,
        selection_start: int,
        selection_length: int,
        *,
        selection_reversed: bool,
    ) -> None:
        text_length = len(edit.text())
        if selection_start < 0 or selection_length <= 0:
            edit.setCursorPosition(min(cursor, text_length))
            return
        start = min(selection_start, text_length)
        length = min(selection_length, text_length - start)
        if length <= 0:
            edit.setCursorPosition(start)
            return
        if selection_reversed:
            edit.setCursorPosition(start + length)
            edit.cursorBackward(True, length)
        else:
            edit.setCursorPosition(start)
            edit.cursorForward(True, length)

    def _capture_data_editor_state(self) -> None:
        edit = self._path_edits[_DATA]
        if edit.hasFocus():
            self._remember_data_editor_state(edit)

    def eventFilter(self, a0: Any, a1: Any) -> bool:
        if (
            a0 is self._path_edits.get(_DATA)
            and a1.type() == QEvent.Type.FocusOut
            and self.save_all_button.hasFocus()
        ):
            self._remember_data_editor_state(a0)
        return super().eventFilter(a0, a1)

    def _restore_data_editor_state(self) -> None:
        state = self._saved_data_editor_state
        self._saved_data_editor_state = None
        if state is None:
            return
        edit, text, cursor, start, length, reversed_selection = state
        if edit.text() != text:
            return
        self._restore_cursor_and_selection(
            edit, cursor, start, length, selection_reversed=reversed_selection
        )
        edit.setFocus()

    def get_data_path(self) -> str:
        return self._path_edits[_DATA].text()

    def path_for(self, key: ArtifactKey) -> str:
        return self._path_edits[key].text()

    def get_comment(self) -> str:
        return self._comment_edit.toPlainText()

    @contextmanager
    def retain_local_path_edit(self, key: ArtifactKey) -> Generator[None]:
        """Project save status without overwriting an in-flight editor change.

        A cleared override resolves to the default path in State, but the blank
        editor must remain blank until the next independent refresh.
        """
        self._local_path_edits.add(key)
        try:
            yield
        finally:
            self._local_path_edits.discard(key)

    def _set_path_preserving_editor_state(self, key: ArtifactKey, path: str) -> None:
        if key in self._local_path_edits:
            return
        edit = self._path_edits[key]
        if edit.text() == path:
            return
        had_focus = edit.hasFocus()
        cursor = edit.cursorPosition()
        start = edit.selectionStart()
        length = edit.selectionLength()
        reversed_selection = start >= 0 and cursor == start
        edit.blockSignals(True)
        try:
            edit.setText(path)
        finally:
            edit.blockSignals(False)
        self._restore_cursor_and_selection(
            edit, cursor, start, length, selection_reversed=reversed_selection
        )
        if had_focus:
            edit.setFocus()

    def set_data_path(self, path: str) -> None:
        self._set_path_preserving_editor_state(_DATA, path)

    def set_image_path(self, key: ArtifactKey, path: str) -> None:
        if key.kind is ArtifactKind.DATA:
            raise ValueError("Use set_data_path for DATA")
        self._set_path_preserving_editor_state(key, path)

    def set_comment_text(self, text: str) -> None:
        if self._comment_edit.toPlainText() != text:
            blocked = self._comment_edit.blockSignals(True)
            try:
                self._comment_edit.setPlainText(text)
            finally:
                self._comment_edit.blockSignals(blocked)

    def bind_data_path_changed(self, handler: Callable[[str], None]) -> None:
        self._path_edits[_DATA].textChanged.connect(handler)

    def bind_comment_changed(self, handler: Callable[[str], None]) -> None:
        self._comment_edit.textChanged.connect(lambda: handler(self.get_comment()))

    def bind_image_path_changed(
        self, handler: Callable[[ArtifactKey, str], None]
    ) -> None:
        self._image_path_handler = handler

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
            self._image_path_slots.pop(key)
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
            self._set_path_preserving_editor_state(key, artifact.default_path or "")

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

    def unbind_data_path_changed(self, handler: Callable[[str], None]) -> None:
        with suppress(TypeError, RuntimeError):
            self._path_edits[_DATA].textChanged.disconnect(handler)

    def unbind_image_path_changed(self) -> None:
        self._image_path_handler = None

    def save_button(self, key: ArtifactKey) -> QPushButton:
        return self._save_btns[key]

"""Data save center for declared artifacts and every member output."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt, QUrl
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

from zcu_tools.experiment.run_saving import ArtifactRef
from zcu_tools.gui.app.measure.artifact_tracker import ArtifactSnapshot, SaveStatus
from zcu_tools.resources.entry import ArtifactKey, OutputFormat

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities
    from zcu_tools.gui.app.measure.services import TabSnapshot

_STATUS_TEXT = {
    SaveStatus.NOT_SAVED: "○ NOT SAVED",
    SaveStatus.PARTIALLY_SAVED: "◐ PARTIALLY SAVED",
    SaveStatus.UNSAVED_CHANGES: "● UNSAVED CHANGES",
    SaveStatus.SAVED: "✓ SAVED",
}
_STATUS_COLOR = {
    SaveStatus.NOT_SAVED: "#3f3f3f",
    SaveStatus.PARTIALLY_SAVED: "#c45100",
    SaveStatus.UNSAVED_CHANGES: "#c45100",
    SaveStatus.SAVED: "#0067c0",
}
_SECTION_TITLE = {
    "run": "Measurement data",
    "analysis": "Analysis results",
    "post_analysis": "Post-analysis results",
}


class ArtifactSaveCenter(QWidget):
    """Render whole-artifact Save actions and read-only per-output evidence.

    Empty sections produce no placeholder artifact. Status and path refreshes
    retain existing widgets and text selections. Save All includes Saved items;
    close warnings concern only incompletely saved measurement data.
    """

    def __init__(
        self,
        capabilities: AdapterCapabilities,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._has_load = capabilities.load_data
        self._snapshots: dict[ArtifactRef, ArtifactSnapshot] = {}
        self._rows: dict[ArtifactRef, QWidget] = {}
        self._status_labels: dict[ArtifactRef, QLabel] = {}
        self._save_btns: dict[ArtifactRef, QPushButton] = {}
        self._output_layouts: dict[ArtifactRef, QVBoxLayout] = {}
        self._output_rows: dict[tuple[ArtifactKey, int], QWidget] = {}
        self._path_edits: dict[tuple[ArtifactKey, int], QLineEdit] = {}
        self._output_status: dict[tuple[ArtifactKey, int], QLabel] = {}
        self._save_handler: Callable[[ArtifactRef], None] | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        outer.setAlignment(Qt.AlignmentFlag.AlignTop)
        heading = QLabel("Save results")
        font = heading.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        heading.setFont(font)
        outer.addWidget(heading)
        actions = QWidget()
        action_layout = QHBoxLayout(actions)
        action_layout.setContentsMargins(0, 0, 0, 0)
        self.load_button = QPushButton("Load Data")
        self.save_all_button = QPushButton("Save All")
        self.save_all_button.setDefault(True)
        for button in (self.load_button, self.save_all_button):
            button.setFixedHeight(36)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if self._has_load:
            action_layout.addWidget(self.load_button, stretch=1)
        else:
            self.load_button.hide()
        action_layout.addWidget(self.save_all_button, stretch=1)
        outer.addWidget(actions)
        self._comment_edit = QTextEdit()
        self._comment_edit.setPlaceholderText("Optional comment…")
        self._comment_edit.setFixedHeight(60)
        outer.addWidget(self._comment_edit)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(8)
        outer.addLayout(self._rows_layout)
        outer.addStretch()

    def _build_row(self, ref: ArtifactRef) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        header = QHBoxLayout()
        title = QLabel(_SECTION_TITLE.get(ref.section, ref.section) + ": " + ref.name)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch()
        status = QLabel()
        header.addWidget(status)
        self._status_labels[ref] = status
        save = QPushButton("Save")
        save.setFixedWidth(72)
        save.clicked.connect(lambda _checked=False, r=ref: self._save_artifact(r))
        header.addWidget(save)
        self._save_btns[ref] = save
        layout.addLayout(header)
        outputs = QVBoxLayout()
        layout.addLayout(outputs)
        self._output_layouts[ref] = outputs
        return container

    def _build_output(
        self, identity: tuple[ArtifactKey, int], format_name: OutputFormat
    ) -> QWidget:
        key, _output_index = identity
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"{key.member_name or key.member} · {format_name}")
        layout.addWidget(label)
        path_row = QHBoxLayout()
        edit = QLineEdit()
        edit.setReadOnly(True)
        path_row.addWidget(edit, stretch=1)
        self._path_edits[identity] = edit
        open_folder = QPushButton("Open folder")
        open_folder.clicked.connect(
            lambda _checked=False, e=edit: self._open_folder(e.text())
        )
        path_row.addWidget(open_folder)
        layout.addLayout(path_row)
        status = QLabel()
        status.setTextFormat(Qt.TextFormat.PlainText)
        status.setWordWrap(True)
        layout.addWidget(status)
        self._output_status[identity] = status
        return container

    def _save_artifact(self, ref: ArtifactRef) -> None:
        if self._save_handler is not None:
            self._save_handler(ref)

    def _open_folder(self, path: str) -> None:
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))

    def get_comment(self) -> str:
        """Return the displayed annotation without changing State."""
        return self._comment_edit.toPlainText()

    def set_comment_text(self, text: str) -> None:
        """Project State's annotation without generating an editing command."""
        if self._comment_edit.toPlainText() != text:
            blocked = self._comment_edit.blockSignals(True)
            try:
                self._comment_edit.setPlainText(text)
            finally:
                self._comment_edit.blockSignals(blocked)

    def bind_comment_changed(self, handler: Callable[[str], None]) -> None:
        """Deliver user annotation edits to the supplied State command."""
        self._comment_edit.textChanged.connect(lambda: handler(self.get_comment()))

    def bind_artifact_save(self, handler: Callable[[ArtifactRef], None]) -> None:
        """Bind each row's Save to its whole artifact ref, not an output path."""
        self._save_handler = handler

    def bind_save_all(self, handler: Callable[[], None]) -> None:
        """Bind the single Save All intention; the widget never sequences saves."""
        self.save_all_button.clicked.connect(lambda _checked=False: handler())

    def bind_load(self, handler: Callable[[], None]) -> None:
        """Bind Load when the adapter declared that capability."""
        if self._has_load:
            self.load_button.clicked.connect(lambda _checked=False: handler())

    def has_unsaved_data(self) -> bool:
        """Warn only for a declared run artifact that is not completely Saved."""
        data = self._snapshots.get(ArtifactRef("run", "measurement_data"))
        return data is not None and data.status is not SaveStatus.SAVED

    def update_from_snapshot(self, snapshot: TabSnapshot) -> None:
        """Project declared sections and every planned/attempted member output.

        Status includes the latest report's per-output error. Replaced sections
        retire omitted rows; unchanged artifact/output identities retain widgets.
        """
        artifacts = {
            artifact.ref: artifact
            for section in snapshot.artifacts
            for artifact in section.artifacts
        }
        for ref in tuple(self._rows):
            if ref not in artifacts:
                widget = self._rows.pop(ref)
                self._rows_layout.removeWidget(widget)
                widget.deleteLater()
                self._status_labels.pop(ref)
                self._save_btns.pop(ref)
                self._output_layouts.pop(ref)
        wanted_outputs = {
            (member.key, index)
            for artifact in artifacts.values()
            for member in artifact.members
            for index, _output in enumerate(member.outputs)
        }
        for identity in tuple(self._output_rows):
            if identity not in wanted_outputs:
                widget = self._output_rows.pop(identity)
                widget.deleteLater()
                self._path_edits.pop(identity)
                self._output_status.pop(identity)
        for index, (ref, artifact) in enumerate(artifacts.items()):
            if ref not in self._rows:
                self._rows[ref] = self._build_row(ref)
            self._rows_layout.insertWidget(index, self._rows[ref])
            self._status_labels[ref].setText(_STATUS_TEXT[artifact.status])
            self._status_labels[ref].setStyleSheet(
                f"color: {_STATUS_COLOR[artifact.status]};"
            )
            output_index = 0
            for member in artifact.members:
                for member_output_index, output in enumerate(member.outputs):
                    identity = (member.key, member_output_index)
                    if identity not in self._output_rows:
                        self._output_rows[identity] = self._build_output(
                            identity, output.format
                        )
                    self._output_layouts[ref].insertWidget(
                        output_index, self._output_rows[identity]
                    )
                    output_index += 1
                    edit = self._path_edits[identity]
                    path = str(output.path)
                    if edit.text() != path:
                        edit.setText(path)
                    self._output_status[identity].setText(
                        f"{output.status}: {output.error}"
                        if output.error is not None
                        else output.status
                    )
        self._snapshots = artifacts

    def update_interaction(self, snapshot: TabSnapshot) -> None:
        """Refresh evidence and enable commands only for idle, ready tabs.

        A complete live snapshot is required. Saved artifacts remain saveable.
        The annotation is disabled while an operation owns the captured content.
        """
        if snapshot.interaction is None or snapshot.capabilities is None:
            raise RuntimeError("Save center needs a live tab snapshot")
        self.update_from_snapshot(snapshot)
        state = snapshot.interaction
        idle = not (state.is_running or state.is_analyzing or state.is_saving_data)
        for button in self._save_btns.values():
            button.setEnabled(idle and state.has_active_context)
        self._comment_edit.setEnabled(idle)
        if self._has_load:
            self.load_button.setEnabled(idle and state.has_context)
        self.save_all_button.setEnabled(
            idle and state.has_active_context and bool(self._snapshots)
        )

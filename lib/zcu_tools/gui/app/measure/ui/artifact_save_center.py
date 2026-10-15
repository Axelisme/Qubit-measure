"""Data save center for declared artifacts and every member output."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, Qt, QUrl
from qtpy.QtGui import QColor, QDesktopServices, QPalette, QResizeEvent
from qtpy.QtWidgets import (
    QApplication,
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
from zcu_tools.gui.app.measure.artifact_tracker import (
    ArtifactSnapshot,
    OutputSnapshot,
    SaveStatus,
)
from zcu_tools.gui.widgets import error_colour
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.resources.entry import ArtifactKey

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
_NO_PERSISTENCE_HINT = "This experiment does not save to the result entry yet."
# QLineEdit reserves a text/cursor inset even without a frame. contentsRect
# and textMargins do not report it; without this slack the elided text clips.
_LINE_EDIT_TEXT_INSET = 8


def _display_path(path: Path, bases: tuple[Path, ...]) -> str:
    """path relative to the first base that contains it, else the full path."""
    for base in bases:
        if path.is_relative_to(base):
            return str(path.relative_to(base))
    return str(path)


def _output_status_text(output: OutputSnapshot) -> str:
    """Return the row-end mark: a check when saved, empty otherwise."""
    return "✓" if output.status == "saved" else ""


def _output_error_text(output: OutputSnapshot) -> str:
    """Return the full-width failure line, or empty when the output did not fail."""
    if output.status != "failed":
        return ""
    return f"✗ {output.error}" if output.error else "✗"


def _hint_label(text: str) -> QLabel:
    """A hidden, wrapping gray note; callers decide when it shows."""
    label = QLabel(text)
    label.setStyleSheet("color: gray;")
    label.setWordWrap(True)
    label.hide()
    return label


class _ElidedPathEdit(QLineEdit):
    """Read-only path field showing its display text elided in the middle.

    set_path(display, full) shows display at the current width and keeps the
    absolute full path as tooltip and full_path() for Open folder.
    """

    def __init__(self) -> None:
        super().__init__()
        self._display = ""
        self._full = ""
        self.setReadOnly(True)
        self.setFrame(False)
        font = self.font()
        if font.pointSizeF() > 0:
            font.setPointSizeF(font.pointSizeF() * 0.9)
        else:
            font.setPixelSize(max(1, round(font.pixelSize() * 0.9)))
        self.setFont(font)
        self._update_colour()

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if a0 is not None:
            if a0.type() == QEvent.Type.PaletteChange:
                self._update_colour()
            elif a0.type() == QEvent.Type.FontChange:
                self._update_text()

    def _update_colour(self) -> None:
        shown = self.palette()
        # Read the inherited colours, not our already-muted Text colour.
        parent = self.parentWidget()
        source = parent.palette() if parent is not None else QApplication.palette()
        text = source.color(QPalette.ColorRole.Text)
        window = source.color(QPalette.ColorRole.Window)
        secondary = QColor(
            round(text.red() + (window.red() - text.red()) * 0.42),
            round(text.green() + (window.green() - text.green()) * 0.42),
            round(text.blue() + (window.blue() - text.blue()) * 0.42),
        )
        if (
            shown.color(QPalette.ColorRole.Text) != secondary
            or shown.color(QPalette.ColorRole.Base).alpha() != 0
        ):
            palette = QPalette()
            palette.setColor(QPalette.ColorRole.Text, secondary)
            palette.setColor(QPalette.ColorRole.Base, QColor(0, 0, 0, 0))
            self.setPalette(palette)

    def full_path(self) -> str:
        return self._full

    def set_path(self, display: str, full: str) -> None:
        if (display, full) == (self._display, self._full):
            return  # Unchanged evidence keeps the user's selection.
        self._display, self._full = display, full
        self.setToolTip(full)
        self._update_text()

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._update_text()

    def _update_text(self) -> None:
        margins = self.textMargins()
        width = (
            self.contentsRect().width()
            - margins.left()
            - margins.right()
            - _LINE_EDIT_TEXT_INSET
        )
        text = self.fontMetrics().elidedText(
            self._display, Qt.TextElideMode.ElideMiddle, max(width, 0)
        )
        if self.text() != text:
            self.setText(text)


@dataclass(frozen=True)
class _ArtifactRow:
    """Widgets of one artifact: header title, status, Save, and its outputs."""

    container: QWidget
    title: QLabel
    status: QLabel
    save: QPushButton
    outputs: QVBoxLayout


@dataclass(frozen=True)
class _OutputRow:
    """Widgets of one member output: file name, path, saved mark and failure."""

    container: QWidget
    name: QLabel
    path: _ElidedPathEdit
    status: QLabel
    error: QLabel


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
        self._rows: dict[ArtifactRef, _ArtifactRow] = {}
        self._output_rows: dict[tuple[ArtifactKey, int], _OutputRow] = {}
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
        self._no_persistence_hint = _hint_label(_NO_PERSISTENCE_HINT)
        outer.addWidget(self._no_persistence_hint)
        self._rows_layout = QVBoxLayout()
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(8)
        outer.addLayout(self._rows_layout)
        outer.addStretch()

    def _build_row(self, ref: ArtifactRef) -> _ArtifactRow:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        header = QHBoxLayout()
        title = QLabel()
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch()
        status = QLabel()
        header.addWidget(status)
        save = QPushButton("Save")
        save.setFixedWidth(72)
        save.clicked.connect(lambda _checked=False, r=ref: self._save_artifact(r))
        header.addWidget(save)
        layout.addLayout(header)
        outputs = QVBoxLayout()
        layout.addLayout(outputs)
        return _ArtifactRow(container, title, status, save, outputs)

    def _build_output(self) -> _OutputRow:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        row = QHBoxLayout()
        row.setSpacing(6)
        name = QLabel()
        make_label_selectable(name)
        row.addWidget(name)
        status = QLabel()
        row.addWidget(status)
        row.addStretch()
        edit = _ElidedPathEdit()
        open_folder = QPushButton("Open folder")
        open_folder.clicked.connect(
            lambda _checked=False, e=edit: self._open_folder(e.full_path())
        )
        row.addWidget(open_folder)
        layout.addLayout(row)
        layout.addWidget(edit)
        # A failure message needs the full width to wrap readably.
        error = QLabel()
        make_label_selectable(error)
        error.setTextFormat(Qt.TextFormat.PlainText)
        error.setWordWrap(True)
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.WindowText, error_colour(self))
        error.setPalette(palette)
        error.hide()
        layout.addWidget(error)
        return _OutputRow(container, name, edit, status, error)

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if a0 is not None and a0.type() == QEvent.Type.PaletteChange:
            palette = QPalette()
            palette.setColor(QPalette.ColorRole.WindowText, error_colour(self))
            for row in self._output_rows.values():
                row.error.setPalette(palette)

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
        A snapshot without its save pane raises RuntimeError.
        """
        if snapshot.save is None:
            raise RuntimeError("Save center needs the tab's save pane snapshot")
        artifacts = {
            artifact.ref: artifact
            for section in snapshot.artifacts
            for artifact in section.artifacts
        }
        for ref in tuple(self._rows):
            if ref not in artifacts:
                row = self._rows.pop(ref)
                self._rows_layout.removeWidget(row.container)
                row.container.deleteLater()
        wanted_outputs = {
            (member.key, index)
            for artifact in artifacts.values()
            for member in artifact.members
            for index, _output in enumerate(member.outputs)
        }
        for identity in tuple(self._output_rows):
            if identity not in wanted_outputs:
                self._output_rows.pop(identity).container.deleteLater()
        bases = snapshot.save.display_bases
        self._no_persistence_hint.setVisible(not snapshot.save.saves_to_entry)
        per_section = {
            section.section: len(section.artifacts) for section in snapshot.artifacts
        }
        for index, (ref, artifact) in enumerate(artifacts.items()):
            # One artifact per section needs no internal artifact name.
            section_title = _SECTION_TITLE.get(ref.section, ref.section)
            title = (
                section_title
                if per_section[ref.section] == 1
                else f"{section_title}: {ref.name}"
            )
            self._render_artifact(index, artifact, title, bases)
        self._snapshots = artifacts

    def _render_artifact(
        self,
        index: int,
        artifact: ArtifactSnapshot,
        title: str,
        bases: tuple[Path, ...],
    ) -> None:
        """Place one artifact row at index and refresh its title, status and outputs."""
        ref = artifact.ref
        if ref not in self._rows:
            self._rows[ref] = self._build_row(ref)
        row = self._rows[ref]
        self._rows_layout.insertWidget(index, row.container)
        row.title.setText(title)
        row.status.setText(_STATUS_TEXT[artifact.status])
        row.status.setStyleSheet(f"color: {_STATUS_COLOR[artifact.status]};")
        output_index = 0
        for member in artifact.members:
            for member_output_index, output in enumerate(member.outputs):
                identity = (member.key, member_output_index)
                if identity not in self._output_rows:
                    self._output_rows[identity] = self._build_output()
                output_row = self._output_rows[identity]
                row.outputs.insertWidget(output_index, output_row.container)
                output_index += 1
                output_row.name.setText(output.path.name)
                output_row.path.set_path(
                    _display_path(output.path, bases), str(output.path)
                )
                output_row.status.setText(_output_status_text(output))
                error_text = _output_error_text(output)
                output_row.error.setText(error_text)
                output_row.error.setVisible(bool(error_text))

    def update_interaction(
        self, snapshot: TabSnapshot, *, writable: bool = True
    ) -> None:
        """Refresh evidence and enable commands only for idle, ready tabs.

        A complete live snapshot is required. Saved artifacts remain saveable.
        The annotation is disabled while an operation owns the captured content.
        """
        if snapshot.interaction is None or snapshot.capabilities is None:
            raise RuntimeError("Save center needs a live tab snapshot")
        self.update_from_snapshot(snapshot)
        state = snapshot.interaction
        idle = writable and not (state.is_running or state.is_analyzing)
        for row in self._rows.values():
            row.save.setEnabled(idle and state.has_active_context)
        self._comment_edit.setEnabled(idle)
        if self._has_load:
            self.load_button.setEnabled(idle and state.has_context)
        self.save_all_button.setEnabled(
            idle and state.has_active_context and bool(self._snapshots)
        )

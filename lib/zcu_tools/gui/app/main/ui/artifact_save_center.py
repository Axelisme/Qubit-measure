"""Data save center — capability-driven artifact rows with terminal-outcome status.

Owns the Data subtab's compact save rows, high-contrast status rendering and
tab-local status lifecycle. Save All updates this center in place and preserves
the data-path editor's focus, cursor and selection. It does not call concrete
save services; it only renders interaction derived from :class:`TabSnapshot` and
the narrow status owner.

Status derivation (S3):
- NO RESULT — capability present but no result yet.
- NOT SAVED — result present, never saved for the current signature.
- UNSAVED CHANGES — result present, saved baseline exists but current signature
  differs (path/comment/result revision changed).
- SAVED — result present and current signature equals the last successfully
  saved signature.

For the async measurement data save, the pending signature is captured at
``notify_save_started`` and only promoted to ``saved`` on true terminal success.
If the current signature drifts during the async window, the terminal success
afterwards shows UNSAVED CHANGES, not an erroneous SAVED.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

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

from zcu_tools.gui.app.main.adapter import AnalysisMode

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import AdapterCapabilities
    from zcu_tools.gui.app.main.services import TabSnapshot

# ---------------------------------------------------------------------------
# Closed artifact kind
# ---------------------------------------------------------------------------


class ArtifactKind(enum.Enum):
    DATA = enum.auto()
    ANALYSIS = enum.auto()
    POST_ANALYSIS = enum.auto()


# ---------------------------------------------------------------------------
# Status model (tab-local, not persisted across process)
# ---------------------------------------------------------------------------


class SaveStatus(enum.Enum):
    NO_RESULT = enum.auto()
    NOT_SAVED = enum.auto()
    UNSAVED_CHANGES = enum.auto()
    SAVED = enum.auto()


_STATUS_TEXT: dict[SaveStatus, str] = {
    SaveStatus.NO_RESULT: "— NO RESULT",
    SaveStatus.NOT_SAVED: "○ NOT SAVED",
    SaveStatus.UNSAVED_CHANGES: "● UNSAVED CHANGES",
    SaveStatus.SAVED: "✓ SAVED",
}

_STATUS_COLOR: dict[SaveStatus, str] = {
    SaveStatus.NO_RESULT: "#7b2cbf",
    SaveStatus.NOT_SAVED: "#3f3f3f",
    SaveStatus.UNSAVED_CHANGES: "#c45100",
    SaveStatus.SAVED: "#0067c0",
}


# ---------------------------------------------------------------------------
# Internal status tracker — Qt-free, owned by the save center
# ---------------------------------------------------------------------------


@dataclass
class _ArtifactRecord:
    has_result: bool = False
    has_figure: bool = False
    result_obj: object | None = None
    result_rev: int = 0
    current_sig: tuple[object, ...] | None = None
    saved_sig: tuple[object, ...] | None = None
    pending_sig: tuple[object, ...] | None = None


class _StatusTracker:
    """Tab-local per-artifact status lifecycle (S3).

    Signature per artifact:
    - data: (result_rev, data_path, comment)
    - analysis: (result_rev, analysis_image_path)
    - post_analysis: (result_rev, post_image_path)

    ``result_rev`` is a monotonic token owned by the tracker; it increments
    when ``result_obj`` identity changes via ``is not``. Retaining the object
    avoids aliasing after Python ``id`` reuse.
    """

    def __init__(self, artifacts: list[ArtifactKind]) -> None:
        self._records: dict[ArtifactKind, _ArtifactRecord] = {
            k: _ArtifactRecord() for k in artifacts
        }

    def update_result(
        self,
        kind: ArtifactKind,
        has_result: bool,
        result_obj: object | None,
        has_figure: bool,
    ) -> None:
        rec = self._records[kind]
        if not has_result:
            rec.has_result = False
            rec.has_figure = False
            rec.result_obj = None
            return
        # has_result True — detect replacement via identity
        if rec.result_obj is not result_obj:
            rec.result_rev += 1
            rec.result_obj = result_obj
        rec.has_result = True
        # figure gating only for image artifacts
        if kind in (ArtifactKind.ANALYSIS, ArtifactKind.POST_ANALYSIS):
            rec.has_figure = bool(has_figure)
        else:
            rec.has_figure = True

    def set_current_sig(self, kind: ArtifactKind, sig: tuple[object, ...]) -> None:
        self._records[kind].current_sig = sig

    def notify_started(self, kind: ArtifactKind) -> None:
        rec = self._records[kind]
        if rec.current_sig is not None:
            rec.pending_sig = rec.current_sig

    def notify_succeeded(self, kind: ArtifactKind) -> None:
        rec = self._records[kind]
        if rec.pending_sig is not None:
            rec.saved_sig = rec.pending_sig
            rec.pending_sig = None
        elif rec.current_sig is not None:
            rec.saved_sig = rec.current_sig

    def notify_failed(self, kind: ArtifactKind) -> None:
        rec = self._records[kind]
        rec.pending_sig = None

    def handle_data_finished(self, error: str | None) -> None:
        rec = self._records[ArtifactKind.DATA]
        if error is None and rec.pending_sig is not None:
            rec.saved_sig = rec.pending_sig
        rec.pending_sig = None

    def status(self, kind: ArtifactKind) -> SaveStatus:
        rec = self._records[kind]
        if not rec.has_result:
            return SaveStatus.NO_RESULT
        if rec.saved_sig is None:
            return SaveStatus.NOT_SAVED
        if rec.current_sig == rec.saved_sig:
            return SaveStatus.SAVED
        return SaveStatus.UNSAVED_CHANGES

    def is_saveable(self, kind: ArtifactKind) -> bool:
        rec = self._records[kind]
        if kind == ArtifactKind.DATA:
            return bool(rec.has_result)
        return bool(rec.has_result and rec.has_figure)

    def result_rev(self, kind: ArtifactKind) -> int:
        return self._records[kind].result_rev

    def __repr__(self) -> str:
        return f"_StatusTracker({self._records!r})"


class _FocusPreservingSaveAllButton(QPushButton):
    """Run Save All without consuming the data-path editor's interaction state."""

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


# ---------------------------------------------------------------------------
# Artifact save center widget
# ---------------------------------------------------------------------------


class ArtifactSaveCenter(QWidget):
    """Compact Data save center with capability-driven rows and status.

    Construction is capability-driven; rows for Analysis/Post appear only when
    the adapter declares them. The center does not call save services — row
    Save buttons are wired via the narrow binding interface.
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

        self._artifacts: list[ArtifactKind] = [ArtifactKind.DATA]
        if self._has_analysis:
            self._artifacts.append(ArtifactKind.ANALYSIS)
        if self._has_post:
            self._artifacts.append(ArtifactKind.POST_ANALYSIS)

        self._tracker = _StatusTracker(self._artifacts)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)
        outer.setAlignment(Qt.AlignTop)  # type: ignore[attr-defined]

        heading = QLabel("Save results")
        hf = heading.font()
        hf.setBold(True)
        hf.setPointSize(hf.pointSize() + 1)
        heading.setFont(hf)
        outer.addWidget(heading)
        detail = QLabel("Only outputs supported by this experiment appear here.")
        detail.setWordWrap(True)
        detail.setStyleSheet("color: #666;")
        outer.addWidget(detail)

        self._status_labels: dict[ArtifactKind, QLabel] = {}
        self._path_edits: dict[ArtifactKind, QLineEdit] = {}
        self._save_btns: dict[ArtifactKind, QPushButton] = {}
        self._saved_data_editor_state: (
            tuple[QLineEdit, str, int, int, int, bool] | None
        ) = None

        actions = QWidget()
        actions.setObjectName("dataActions")
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)

        self.load_button = QPushButton("Load Data")
        self.load_button.setFixedHeight(36)
        self.load_button.setSizePolicy(
            QSizePolicy.Expanding,  # type: ignore[attr-defined]
            QSizePolicy.Fixed,  # type: ignore[attr-defined]
        )
        self.save_all_button = _FocusPreservingSaveAllButton(
            self._capture_data_editor_state,
            self._restore_data_editor_state,
        )
        self.save_all_button.setFixedHeight(36)
        self.save_all_button.setSizePolicy(
            QSizePolicy.Expanding,  # type: ignore[attr-defined]
            QSizePolicy.Fixed,  # type: ignore[attr-defined]
        )
        # Save All restores the data-path editor after its ordered saves have
        # updated this center; normal mouse focus lets the editor observe which
        # action caused its FocusOut event.
        self.save_all_button.setDefault(True)

        if self._has_load:
            actions_layout.addWidget(self.load_button, stretch=1)
            actions_layout.addWidget(self.save_all_button, stretch=1)
        else:
            actions_layout.addWidget(self.save_all_button, stretch=1)
            self.load_button.hide()
        outer.addWidget(actions)

        data_row = self._build_row(
            kind=ArtifactKind.DATA,
            title="Measurement data",
            placeholder="/tmp/data.hdf5",
            browse_tooltip="Choose data destination",
            save_label="Save",
            with_comment=True,
        )
        outer.addWidget(data_row)

        if self._has_analysis:
            analysis_row = self._build_row(
                kind=ArtifactKind.ANALYSIS,
                title="Analysis image",
                placeholder="/tmp/image.png",
                browse_tooltip="Choose analysis image destination",
                save_label="Save",
                with_comment=False,
            )
            outer.addWidget(analysis_row)

        if self._has_post:
            post_row = self._build_row(
                kind=ArtifactKind.POST_ANALYSIS,
                title="Post-analysis image",
                placeholder="/tmp/post_image.png",
                browse_tooltip="Choose post-analysis image destination",
                save_label="Save",
                with_comment=False,
            )
            outer.addWidget(post_row)

        outer.addStretch()

        # Internal comment edit is created in _build_row for DATA
        self._comment_edit: QTextEdit  # assigned in row construction

        self._refresh_all_status_labels()
        self._wire_internal_status_updates()

    # -- row construction --------------------------------------------

    def _build_row(
        self,
        *,
        kind: ArtifactKind,
        title: str,
        placeholder: str,
        browse_tooltip: str,
        save_label: str,
        with_comment: bool,
    ) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        tf = title_label.font()
        tf.setBold(True)
        title_label.setFont(tf)
        header.addWidget(title_label)
        header.addStretch()
        status = QLabel()
        sf = status.font()
        sf.setBold(True)
        status.setFont(sf)
        status.setTextFormat(Qt.RichText)  # type: ignore[attr-defined]
        header.addWidget(status)
        layout.addLayout(header)
        self._status_labels[kind] = status

        path_row = QHBoxLayout()
        path_row.setContentsMargins(0, 0, 0, 0)
        path_row.setSpacing(6)
        path_edit = QLineEdit()
        path_edit.setPlaceholderText(placeholder)
        path_row.addWidget(path_edit, stretch=1)
        self._path_edits[kind] = path_edit

        browse = QPushButton("Browse…")
        browse.setFixedWidth(80)
        browse.setToolTip(browse_tooltip)
        browse.clicked.connect(lambda _checked=False, k=kind: self._on_browse(k))
        path_row.addWidget(browse)

        save = QPushButton(save_label)
        save.setFixedWidth(72)
        path_row.addWidget(save)
        self._save_btns[kind] = save
        layout.addLayout(path_row)

        if with_comment:
            comment = QTextEdit()
            comment.setPlaceholderText("Optional comment…")
            comment.setFixedHeight(60)
            comment.setSizePolicy(
                QSizePolicy.Expanding,  # type: ignore[attr-defined]
                QSizePolicy.Fixed,  # type: ignore[attr-defined]
            )
            layout.addWidget(comment)
            self._comment_edit = comment

        return container

    def _wire_internal_status_updates(self) -> None:
        for kind, edit in self._path_edits.items():
            edit.textChanged.connect(
                lambda _text, k=kind: self._on_path_or_comment_changed(k)
            )
        self._path_edits[ArtifactKind.DATA].installEventFilter(self)
        if hasattr(self, "_comment_edit"):
            self._comment_edit.textChanged.connect(
                lambda: self._on_path_or_comment_changed(ArtifactKind.DATA)
            )

    # -- browse handlers (view-only, no controller) ------------------

    def _on_browse(self, kind: ArtifactKind) -> None:
        if kind == ArtifactKind.DATA:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save data file", "", "HDF5 files (*.hdf5);;All files (*)"
            )
        elif kind == ArtifactKind.ANALYSIS:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save image file", "", "PNG files (*.png);;All files (*)"
            )
        elif kind == ArtifactKind.POST_ANALYSIS:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Save post-analysis image file",
                "",
                "PNG files (*.png);;All files (*)",
            )
        else:
            return
        if path:
            self._path_edits[kind].setText(path)

    # -- path/comment accessors --------------------------------------

    def _remember_data_editor_state(self, edit: QLineEdit) -> None:
        selection_start = edit.selectionStart()
        cursor = edit.cursorPosition()
        selection_length = edit.selectionLength()
        self._saved_data_editor_state = (
            edit,
            edit.text(),
            cursor,
            selection_start,
            selection_length,
            selection_start >= 0 and cursor == selection_start,
        )

    @staticmethod
    def _restore_cursor_and_selection(
        edit: QLineEdit,
        cursor: int,
        selection_start: int,
        selection_length: int,
        selection_reversed: bool,
    ) -> None:
        """Restore a line edit with its active cursor endpoint intact."""
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
        edit = self._path_edits[ArtifactKind.DATA]
        # A FocusOut event may have already been delivered before the button's
        # mouse-press handler runs. In that case eventFilter captured the state;
        # do not replace it with the already-cleared selection.
        if edit.hasFocus():
            self._remember_data_editor_state(edit)

    def eventFilter(self, a0: Any, a1: Any) -> bool:
        if (
            a0 is self._path_edits.get(ArtifactKind.DATA)
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
        (
            edit,
            text,
            cursor,
            selection_start,
            selection_length,
            selection_reversed,
        ) = state
        if edit.text() != text:
            return
        self._restore_cursor_and_selection(
            edit,
            cursor,
            selection_start,
            selection_length,
            selection_reversed,
        )
        edit.setFocus()

    def get_data_path(self) -> str:
        return self._path_edits[ArtifactKind.DATA].text()

    def get_analysis_path(self) -> str:
        if not self._has_analysis:
            raise RuntimeError(f"tab {self._tab_id!r} does not support analysis")
        return self._path_edits[ArtifactKind.ANALYSIS].text()

    def get_post_analysis_path(self) -> str:
        if not self._has_post:
            raise RuntimeError(f"tab {self._tab_id!r} does not support post-analysis")
        return self._path_edits[ArtifactKind.POST_ANALYSIS].text()

    def get_comment(self) -> str:
        if hasattr(self, "_comment_edit"):
            return self._comment_edit.toPlainText()
        return ""

    def _set_path_preserving_editor_state(self, kind: ArtifactKind, path: str) -> None:
        """Apply a state-driven path without disturbing the active editor.

        Save lifecycle reactions can refresh a pane while a path is being
        edited. Avoiding a redundant ``setText`` is important: Qt clears a
        line edit's selection even when the replacement text is identical.
        When the text does change, retain the best valid cursor/selection
        projection and restore focus only when this editor owned it.
        """
        edit = self._path_edits[kind]
        if edit.text() != path:
            had_focus = edit.hasFocus()
            cursor = edit.cursorPosition()
            selection_start = edit.selectionStart()
            selection_length = edit.selectionLength()
            selection_reversed = selection_start >= 0 and cursor == selection_start

            edit.blockSignals(True)
            try:
                edit.setText(path)
            finally:
                edit.blockSignals(False)

            self._restore_cursor_and_selection(
                edit,
                cursor,
                selection_start,
                selection_length,
                selection_reversed,
            )
            if had_focus:
                edit.setFocus()
        self._recompute_current_sig(kind)

    def set_data_path(self, path: str) -> None:
        self._set_path_preserving_editor_state(ArtifactKind.DATA, path)

    def set_analysis_path(self, path: str) -> None:
        if not self._has_analysis:
            raise RuntimeError(f"tab {self._tab_id!r} does not support analysis")
        self._set_path_preserving_editor_state(ArtifactKind.ANALYSIS, path)

    def set_post_analysis_path(self, path: str) -> None:
        if not self._has_post:
            raise RuntimeError(f"tab {self._tab_id!r} does not support post-analysis")
        self._set_path_preserving_editor_state(ArtifactKind.POST_ANALYSIS, path)

    def set_comment_text(self, text: str) -> None:
        if hasattr(self, "_comment_edit"):
            self._comment_edit.blockSignals(True)
            self._comment_edit.setPlainText(text)
            self._comment_edit.blockSignals(False)
            self._recompute_current_sig(ArtifactKind.DATA)

    # -- narrow binding interface for ExpTabWidget -------------------

    def bind_data_path_changed(self, handler: Callable[[str], None]) -> None:
        self._path_edits[ArtifactKind.DATA].textChanged.connect(handler)

    def bind_analysis_path_changed(self, handler: Callable[[str], None]) -> None:
        if ArtifactKind.ANALYSIS in self._path_edits:
            self._path_edits[ArtifactKind.ANALYSIS].textChanged.connect(handler)

    def bind_post_path_changed(self, handler: Callable[[str], None]) -> None:
        if ArtifactKind.POST_ANALYSIS in self._path_edits:
            self._path_edits[ArtifactKind.POST_ANALYSIS].textChanged.connect(handler)

    def bind_save(self, kind: ArtifactKind, handler: Callable[[], None]) -> None:
        btn = self._save_btns.get(kind)
        if btn is None:
            raise RuntimeError(
                f"artifact {kind!r} not present for tab {self._tab_id!r}"
            )
        btn.clicked.connect(lambda _checked=False: handler())

    def bind_save_all(self, handler: Callable[[], None]) -> None:
        self.save_all_button.clicked.connect(lambda _checked=False: handler())

    def bind_load(self, handler: Callable[[], None]) -> None:
        if self._has_load:
            self.load_button.clicked.connect(lambda _checked=False: handler())

    # -- observable query interface for tests ------------------------

    @property
    def artifact_kinds(self) -> list[ArtifactKind]:
        return list(self._artifacts)

    def has_artifact(self, kind: ArtifactKind) -> bool:
        return kind in self._artifacts

    def is_save_enabled(self, kind: ArtifactKind) -> bool:
        btn = self._save_btns.get(kind)
        return bool(btn is not None and btn.isEnabled())

    def is_save_all_enabled(self) -> bool:
        return self.save_all_button.isEnabled()

    def is_load_enabled(self) -> bool:
        return bool(self._has_load and self.load_button.isEnabled())

    def is_load_visible(self) -> bool:
        if not self._has_load:
            return False
        return not self.load_button.isHidden()

    def is_path_enabled(self, kind: ArtifactKind) -> bool:
        edit = self._path_edits.get(kind)
        return bool(edit is not None and edit.isEnabled())

    def ordered_saveable_kinds(self, snapshot: TabSnapshot) -> list[ArtifactKind]:
        """Ordered saveable artifacts for Save All (analysis→post→data)."""
        kinds: list[ArtifactKind] = []
        if self._has_analysis:
            if (
                snapshot.analysis is not None
                and snapshot.analysis.result is not None
                and snapshot.analysis.figure is not None
            ):
                kinds.append(ArtifactKind.ANALYSIS)
        if self._has_post:
            if (
                snapshot.post_analysis is not None
                and snapshot.post_analysis.result is not None
                and snapshot.post_analysis.figure is not None
            ):
                kinds.append(ArtifactKind.POST_ANALYSIS)
        if snapshot.run is not None and snapshot.run.result is not None:
            kinds.append(ArtifactKind.DATA)
        return kinds

    # -- tracker helpers ----------------------------------------------

    def _current_sig_for(self, kind: ArtifactKind) -> tuple[object, ...]:
        rev = self._tracker.result_rev(kind)
        if kind == ArtifactKind.DATA:
            path = self._path_edits[ArtifactKind.DATA].text()
            comment = self.get_comment()
            return (rev, path, comment)
        else:
            path = self._path_edits[kind].text()
            return (rev, path)

    def _recompute_current_sig(self, kind: ArtifactKind) -> None:
        sig = self._current_sig_for(kind)
        self._tracker.set_current_sig(kind, sig)
        self._refresh_status_label(kind)

    def _on_path_or_comment_changed(self, kind: ArtifactKind) -> None:
        self._recompute_current_sig(kind)

    def _refresh_status_label(self, kind: ArtifactKind) -> None:
        status = self._tracker.status(kind)
        label = self._status_labels[kind]
        label.setText(_STATUS_TEXT[status])
        label.setStyleSheet(f"color: {_STATUS_COLOR[status]};")

    def _refresh_all_status_labels(self) -> None:
        for kind in self._artifacts:
            self._refresh_status_label(kind)

    # -- snapshot-driven updates --------------------------------------

    def update_from_snapshot(self, snapshot: TabSnapshot) -> None:
        if snapshot.run is not None and snapshot.run.result is not None:
            has_data = True
            data_obj = snapshot.run.result
        else:
            has_data = False
            data_obj = None
        self._tracker.update_result(ArtifactKind.DATA, has_data, data_obj, False)

        if self._has_analysis:
            if snapshot.analysis is not None and snapshot.analysis.result is not None:
                has_ana = True
                ana_obj = snapshot.analysis.result
            else:
                has_ana = False
                ana_obj = None
            has_fig = bool(
                snapshot.analysis is not None and snapshot.analysis.figure is not None
            )
            self._tracker.update_result(
                ArtifactKind.ANALYSIS, has_ana, ana_obj, has_fig
            )

        if self._has_post:
            if (
                snapshot.post_analysis is not None
                and snapshot.post_analysis.result is not None
            ):
                has_post = True
                post_obj = snapshot.post_analysis.result
            else:
                has_post = False
                post_obj = None
            has_fig = bool(
                snapshot.post_analysis is not None
                and snapshot.post_analysis.figure is not None
            )
            self._tracker.update_result(
                ArtifactKind.POST_ANALYSIS, has_post, post_obj, has_fig
            )

        for kind in self._artifacts:
            self._recompute_current_sig(kind)

    def update_interaction(self, snapshot: TabSnapshot) -> None:
        assert snapshot.interaction is not None
        assert snapshot.capabilities is not None
        self.update_from_snapshot(snapshot)
        state = snapshot.interaction
        idle = not (state.is_running or state.is_analyzing or state.is_saving_data)
        has_active = state.has_active_context
        has_context = state.has_context

        for kind in self._artifacts:
            btn = self._save_btns[kind]
            if not idle or not has_active:
                btn.setEnabled(False)
            else:
                btn.setEnabled(self._tracker.is_saveable(kind))
        if self._has_load:
            self.load_button.setEnabled(idle and has_context)
        any_saveable = any(self._tracker.is_saveable(k) for k in self._artifacts)
        self.save_all_button.setEnabled(idle and has_active and any_saveable)

    # -- save outcome notifications -----------------------------------

    def notify_save_started(self, kind: ArtifactKind) -> None:
        self._recompute_current_sig(kind)
        self._tracker.notify_started(kind)
        self._refresh_status_label(kind)

    def notify_save_succeeded(self, kind: ArtifactKind) -> None:
        self._recompute_current_sig(kind)
        self._tracker.notify_succeeded(kind)
        self._refresh_status_label(kind)

    def notify_save_failed(self, kind: ArtifactKind) -> None:
        self._tracker.notify_failed(kind)
        self._refresh_status_label(kind)

    def handle_data_finished(self, error: str | None) -> None:
        self._recompute_current_sig(ArtifactKind.DATA)
        self._tracker.handle_data_finished(error)
        self._refresh_status_label(ArtifactKind.DATA)

    # -- helpers for tests --------------------------------------------

    def status_text(self, kind: ArtifactKind) -> str:
        return self._status_labels[kind].text()

    def status_color(self, kind: ArtifactKind) -> str:
        ss = self._status_labels[kind].styleSheet()
        if "color:" in ss:
            return ss.split("color:")[1].strip().strip(";").strip()
        return ""

    # -- unbind helpers for detach cleanup -----------------------------
    def unbind_data_path_changed(self, handler: Callable[[str], None]) -> None:
        try:
            self._path_edits[ArtifactKind.DATA].textChanged.disconnect(handler)
        except (TypeError, RuntimeError):
            pass

    def unbind_analysis_path_changed(self, handler: Callable[[str], None]) -> None:
        try:
            self._path_edits[ArtifactKind.ANALYSIS].textChanged.disconnect(handler)
        except (TypeError, RuntimeError, KeyError):
            pass

    def unbind_post_path_changed(self, handler: Callable[[str], None]) -> None:
        try:
            self._path_edits[ArtifactKind.POST_ANALYSIS].textChanged.disconnect(handler)
        except (TypeError, RuntimeError, KeyError):
            pass

    def save_button(self, kind: ArtifactKind) -> QPushButton:
        btn = self._save_btns.get(kind)
        if btn is None:
            raise RuntimeError(f"artifact {kind!r} not present")
        return btn

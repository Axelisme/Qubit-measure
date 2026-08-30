"""Data save center — capability-driven artifact rows with terminal-outcome status.

Owns the Data subtab's compact save rows, high-contrast status rendering and
tab-local status lifecycle. It does not call concrete save services; it only
renders interaction derived from :class:`TabSnapshot` and the narrow status
owner.

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
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt  # type: ignore[attr-defined]
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
    result_id: int | None = None
    current_sig: tuple[object, ...] | None = None
    saved_sig: tuple[object, ...] | None = None
    pending_sig: tuple[object, ...] | None = None


class _StatusTracker:
    """Tab-local per-artifact status lifecycle (S3).

    Signature per artifact:
    - data: (result_id, data_path, comment)
    - analysis: (result_id, analysis_image_path)
    - post_analysis: (result_id, post_image_path)
    """

    def __init__(self, artifacts: list[str]) -> None:
        self._records: dict[str, _ArtifactRecord] = {
            k: _ArtifactRecord() for k in artifacts
        }

    def update_result(self, kind: str, has_result: bool, result_id: int | None) -> None:
        rec = self._records[kind]
        rec.has_result = has_result
        rec.result_id = result_id
        # current_sig will be recomputed by the caller after this.

    def set_current_sig(self, kind: str, sig: tuple[object, ...]) -> None:
        self._records[kind].current_sig = sig

    def notify_started(self, kind: str) -> None:
        rec = self._records[kind]
        if rec.current_sig is not None:
            rec.pending_sig = rec.current_sig

    def notify_succeeded(self, kind: str) -> None:
        rec = self._records[kind]
        # For data async: promote pending; for sync images: pending may be
        # None but we still promote current.
        if rec.pending_sig is not None:
            rec.saved_sig = rec.pending_sig
            rec.pending_sig = None
        elif rec.current_sig is not None:
            rec.saved_sig = rec.current_sig

    def notify_failed(self, kind: str) -> None:
        rec = self._records[kind]
        rec.pending_sig = None

    def handle_data_finished(self, error: str | None) -> None:
        rec = self._records["data"]
        if error is None and rec.pending_sig is not None:
            rec.saved_sig = rec.pending_sig
        # No pending -> ignore status transition (remote/MCP or unmatched completion).
        # Failure: keep saved unchanged.
        rec.pending_sig = None

    def status(self, kind: str) -> SaveStatus:
        rec = self._records[kind]
        if not rec.has_result:
            return SaveStatus.NO_RESULT
        if rec.saved_sig is None:
            return SaveStatus.NOT_SAVED
        if rec.current_sig == rec.saved_sig:
            return SaveStatus.SAVED
        return SaveStatus.UNSAVED_CHANGES

    def __repr__(self) -> str:
        return f"_StatusTracker({self._records!r})"


# ---------------------------------------------------------------------------
# Artifact save center widget
# ---------------------------------------------------------------------------


class ArtifactSaveCenter(QWidget):
    """Compact Data save center with capability-driven rows and status.

    Construction is capability-driven; rows for Analysis/Post appear only when
    the adapter declares them. The center does not call save services — row
    Save buttons are wired by :class:`ExpTabWidget` through ``TabActions``.
    """

    def __init__(
        self,
        tab_id: str,
        capabilities: AdapterCapabilities,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tab_id = tab_id
        self._capabilities = capabilities
        self._has_analysis = capabilities.analysis is not AnalysisMode.NONE
        self._has_post = bool(capabilities.post_analysis)
        self._has_load = bool(capabilities.load_data)

        self._artifacts: list[str] = ["data"]
        if self._has_analysis:
            self._artifacts.append("analysis")
        if self._has_post:
            self._artifacts.append("post_analysis")

        self._tracker = _StatusTracker(self._artifacts)

        # -- layout ---------------------------------------------------
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

        # Artifact rows
        self._status_labels: dict[str, QLabel] = {}
        self._path_edits: dict[str, QLineEdit] = {}
        self._save_btns: dict[str, QPushButton] = {}
        self._row_widgets: dict[str, QWidget] = {}

        # Data row (always)
        data_row = self._build_row(
            kind="data",
            title="Measurement data",
            placeholder="/tmp/data.hdf5",
            browse_tooltip="Choose data destination",
            save_label="Save",
            with_comment=True,
        )
        outer.addWidget(data_row)

        if self._has_analysis:
            analysis_row = self._build_row(
                kind="analysis",
                title="Analysis image",
                placeholder="/tmp/image.png",
                browse_tooltip="Choose analysis image destination",
                save_label="Save",
                with_comment=False,
            )
            outer.addWidget(analysis_row)

        if self._has_post:
            post_row = self._build_row(
                kind="post_analysis",
                title="Post-analysis image",
                placeholder="/tmp/post_image.png",
                browse_tooltip="Choose post-analysis image destination",
                save_label="Save",
                with_comment=False,
            )
            outer.addWidget(post_row)

        # Bottom: Load Data + Save All
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        self.load_button = QPushButton("Load Data")
        self.load_button.setFixedHeight(36)
        self.load_button.setSizePolicy(
            QSizePolicy.Expanding,  # type: ignore[attr-defined]
            QSizePolicy.Fixed,  # type: ignore[attr-defined]
        )
        self.save_all_button = QPushButton("Save All")
        self.save_all_button.setFixedHeight(36)
        self.save_all_button.setSizePolicy(
            QSizePolicy.Expanding,  # type: ignore[attr-defined]
            QSizePolicy.Fixed,  # type: ignore[attr-defined]
        )
        self.save_all_button.setDefault(True)

        if self._has_load:
            bottom_layout.addWidget(self.load_button, stretch=1)
            bottom_layout.addWidget(self.save_all_button, stretch=1)
        else:
            bottom_layout.addWidget(self.save_all_button, stretch=1)
            self.load_button.hide()

        outer.addWidget(bottom)
        outer.addStretch()

        # Expose aliases for ExpTabWidget compatibility
        # These are the same objects owned by the center; keep public names
        # for minimal churn in MainWindow and tests.
        self._data_path_edit: QLineEdit = self._path_edits["data"]
        self._comment_edit: QTextEdit
        # _comment_edit set in _build_row for data; assign after.
        # Keep type checkers happy: it is created in data row.
        # For analysis/post, expose optional edits.
        # Use getattr where needed.

        # Keep browse handlers inside center.
        # Path edits' textChanged already updates tracker via connections below.
        # Comment edit handling also updates tracker.

        # Initial status render (no result).
        self._refresh_all_status_labels()

        # Wire internal textChanged to status refresh (no controller write yet;
        # ExpTabWidget will wire to controller separately).
        self._wire_internal_status_updates()

    # -- row construction --------------------------------------------

    def _build_row(
        self,
        *,
        kind: str,
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

        # Header: bold title left, bold status right
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

        # Path row: stretch path, fixed Browse/Save
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
            layout.addWidget(comment)
            self._comment_edit = comment  # type: ignore[attr-defined]
            # Store for later alias
            self.comment_edit = comment  # public alias

        self._row_widgets[kind] = container
        return container

    def _wire_internal_status_updates(self) -> None:
        # Path changes affect current_sig and status.
        for kind, edit in self._path_edits.items():
            edit.textChanged.connect(
                lambda _text, k=kind: self._on_path_or_comment_changed(k)
            )
        if hasattr(self, "_comment_edit"):
            self._comment_edit.textChanged.connect(
                lambda: self._on_path_or_comment_changed("data")
            )

    # -- browse handlers (view-only, no controller) ------------------

    def _on_browse(self, kind: str) -> None:
        if kind == "data":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save data file", "", "HDF5 files (*.hdf5);;All files (*)"
            )
        elif kind == "analysis":
            path, _ = QFileDialog.getSaveFileName(
                self, "Save image file", "", "PNG files (*.png);;All files (*)"
            )
        elif kind == "post_analysis":
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

    # -- public widget accessors (for ExpTabWidget/MainWindow) --------

    @property
    def data_path_edit(self) -> QLineEdit:
        return self._path_edits["data"]

    @property
    def analysis_path_edit(self) -> QLineEdit | None:
        return self._path_edits.get("analysis")

    @property
    def post_analysis_path_edit(self) -> QLineEdit | None:
        return self._path_edits.get("post_analysis")

    @property
    def _data_save_btn(self) -> QPushButton:
        return self._save_btns["data"]

    @property
    def _analysis_save_btn(self) -> QPushButton | None:
        return self._save_btns.get("analysis")

    @property
    def _post_save_btn(self) -> QPushButton | None:
        return self._save_btns.get("post_analysis")

    def get_data_path(self) -> str:
        return self._path_edits["data"].text()

    def get_analysis_path(self) -> str:
        # Mirror ExpTabWidget's guard.
        if not self._has_analysis:
            raise RuntimeError(f"tab {self._tab_id!r} does not support analysis")
        return self._path_edits["analysis"].text()

    def get_post_analysis_path(self) -> str:
        if not self._has_post:
            raise RuntimeError(f"tab {self._tab_id!r} does not support post-analysis")
        return self._path_edits["post_analysis"].text()

    def get_comment(self) -> str:
        if hasattr(self, "_comment_edit"):
            return self._comment_edit.toPlainText()
        return ""

    def set_data_path(self, path: str) -> None:
        edit = self._path_edits["data"]
        edit.blockSignals(True)
        edit.setText(path)
        edit.blockSignals(False)
        # Update tracker after programmatic set.
        self._recompute_current_sig("data")

    def set_analysis_path(self, path: str) -> None:
        if not self._has_analysis:
            raise RuntimeError(f"tab {self._tab_id!r} does not support analysis")
        edit = self._path_edits["analysis"]
        edit.blockSignals(True)
        edit.setText(path)
        edit.blockSignals(False)
        self._recompute_current_sig("analysis")

    def set_post_analysis_path(self, path: str) -> None:
        if not self._has_post:
            raise RuntimeError(f"tab {self._tab_id!r} does not support post-analysis")
        edit = self._path_edits["post_analysis"]
        edit.blockSignals(True)
        edit.setText(path)
        edit.blockSignals(False)
        self._recompute_current_sig("post_analysis")

    def set_comment_text(self, text: str) -> None:
        if hasattr(self, "_comment_edit"):
            self._comment_edit.blockSignals(True)
            self._comment_edit.setPlainText(text)
            self._comment_edit.blockSignals(False)
            self._recompute_current_sig("data")

    # -- tracker helpers ----------------------------------------------

    def _current_sig_for(self, kind: str) -> tuple[object, ...]:
        rec = self._tracker._records[kind]
        result_id = rec.result_id
        if kind == "data":
            path = self._path_edits["data"].text()
            comment = self.get_comment()
            return (result_id, path, comment)
        else:
            path = self._path_edits[kind].text()
            return (result_id, path)

    def _recompute_current_sig(self, kind: str) -> None:
        sig = self._current_sig_for(kind)
        self._tracker.set_current_sig(kind, sig)
        self._refresh_status_label(kind)

    def _on_path_or_comment_changed(self, kind: str) -> None:
        # For data, path and comment both affect data sig; but we also need
        # to recompute data sig when analysis path changes? Keep per-kind.
        # Data comment change also affects data only.
        self._recompute_current_sig(kind)
        if kind == "data":
            # comment change already covered, but path change for data also
            # only data; nothing else.
            pass

    def _refresh_status_label(self, kind: str) -> None:
        status = self._tracker.status(kind)
        label = self._status_labels[kind]
        label.setText(_STATUS_TEXT[status])
        label.setStyleSheet(f"color: {_STATUS_COLOR[status]};")

    def _refresh_all_status_labels(self) -> None:
        for kind in self._artifacts:
            self._refresh_status_label(kind)

    # -- snapshot-driven updates --------------------------------------

    def update_from_snapshot(self, snapshot: TabSnapshot) -> None:
        """Sync result presence + path texts from a single snapshot fetch."""
        # Result availability + ids.
        # Paths are taken from current widget texts already? But we sync path
        # texts from snapshot if they differ and widget text is empty? Actually
        # snapshot's effective path is the state projection; widget text should
        # mirror it when no user edit. To keep single-source, we update widget
        # texts to snapshot's path only when snapshot provides a non-empty path
        # and widget differs? Simpler: always sync widget text to snapshot's
        # path when snapshot differs? But that would overwrite user typing
        # before controller update.
        # Instead we treat widget text as source of truth for path after attach;
        # snapshot path sync happens via ExpTabWidget's set_* which already calls
        # recompute. So here we only update result presence, not path text.
        # However after a context switch, snapshot's path may change due to new
        # context's adapter default; we should update widget text then.
        # To avoid overwriting user in-flight edit, we could compare.
        # Simplify: update widget texts from snapshot when they differ?
        # The existing ExpTabWidget's refresh_tab_save_paths already calls
        # set_data_path etc which blockSignals and recompute. So this method's
        # path sync is not needed if caller already did set_*.
        # We'll just update result tracking here and recompute sigs.

        # For each artifact, update has_result and result_id.
        # Data
        has_data = False
        data_id: int | None = None
        if snapshot.run is not None and snapshot.run.result is not None:
            has_data = True
            data_id = id(snapshot.run.result)
        self._tracker.update_result("data", has_data, data_id)

        if self._has_analysis:
            has_ana = False
            ana_id: int | None = None
            if snapshot.analysis is not None and snapshot.analysis.result is not None:
                # Use result id; figure life follows result, so result alone is enough.
                has_ana = True
                ana_id = id(snapshot.analysis.result)
            self._tracker.update_result("analysis", has_ana, ana_id)

        if self._has_post:
            has_post = False
            post_id: int | None = None
            if (
                snapshot.post_analysis is not None
                and snapshot.post_analysis.result is not None
            ):
                has_post = True
                post_id = id(snapshot.post_analysis.result)
            self._tracker.update_result("post_analysis", has_post, post_id)

        # Recompute current sigs for all artifacts (result_id may have changed)
        for kind in self._artifacts:
            self._recompute_current_sig(kind)

    def refresh_paths_from_snapshot(self, snapshot: TabSnapshot) -> None:
        """Explicit path sync from snapshot (called when ExpTabWidget sets paths)."""
        # This is called after ExpTabWidget's set_* which already recomputes,
        # but we also ensure tracker recomputes if snapshot path differ.
        # We just ensure current sigs reflect latest widget texts, which are already.
        for kind in self._artifacts:
            self._recompute_current_sig(kind)

    def update_interaction(self, snapshot: TabSnapshot) -> None:
        """Update enablement (idle/context) and status after interaction change."""
        assert snapshot.interaction is not None
        assert snapshot.capabilities is not None
        # Keep result tracking in sync (also handles busy->idle result stability)
        self.update_from_snapshot(snapshot)
        # Enablement
        state = snapshot.interaction
        idle = not (state.is_running or state.is_analyzing or state.is_saving_data)
        has_active = state.has_active_context
        has_context = state.has_context

        # Per-artifact Save enablement
        for kind in self._artifacts:
            btn = self._save_btns[kind]
            if not idle or not has_active:
                btn.setEnabled(False)
            else:
                # No result -> disabled
                rec = self._tracker._records[kind]
                btn.setEnabled(bool(rec.has_result))
        # Load Data enablement
        if self._has_load:
            self.load_button.setEnabled(idle and has_context)
        # Save All enablement
        any_has = any(self._tracker._records[k].has_result for k in self._artifacts)
        self.save_all_button.setEnabled(idle and has_active and any_has)

    # -- save outcome notifications -----------------------------------

    def notify_save_started(self, kind: str) -> None:
        # Capture current sig as pending.
        # Ensure current sig is up to date.
        self._recompute_current_sig(kind)
        self._tracker.notify_started(kind)
        # Status does not become SAVED yet; keep current label (still NOT SAVED or UNSAVED)
        self._refresh_status_label(kind)

    def notify_save_succeeded(self, kind: str) -> None:
        self._recompute_current_sig(kind)
        self._tracker.notify_succeeded(kind)
        self._refresh_status_label(kind)

    def notify_save_failed(self, kind: str) -> None:
        self._tracker.notify_failed(kind)
        self._refresh_status_label(kind)

    def handle_data_finished(self, error: str | None) -> None:
        # Update current sig before handling (in case path/comment changed during async)
        self._recompute_current_sig("data")
        self._tracker.handle_data_finished(error)
        self._refresh_status_label("data")

    # -- helpers for tests --------------------------------------------

    def status_text(self, kind: str) -> str:
        """Return the current status label text (for tests)."""
        return self._status_labels[kind].text()

    def status_color(self, kind: str) -> str:
        # Extract color from stylesheet
        ss = self._status_labels[kind].styleSheet()
        # ss is like "color: #xxxxxx;"
        if "color:" in ss:
            return ss.split("color:")[1].strip().strip(";").strip()
        return ""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QApplication,
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.measure.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
    WritebackItem,
)
from zcu_tools.gui.app.measure.ui.cfg_binding import make_value_source_input_enhancer
from zcu_tools.gui.widgets.cfg import CfgFormWidget

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.controller import Controller
    from zcu_tools.gui.app.measure.services.writeback import WritebackValues
    from zcu_tools.gui.app.measure.services.writeback_control import WritebackPane

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presentation helpers for non-scalar MetaDict values (S2)
# ---------------------------------------------------------------------------


def _is_matrix_value(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) == 0:
        return False
    if not all(isinstance(row, (list, tuple)) for row in value):
        return False
    first_len = len(value[0])  # type: ignore[arg-type]
    if first_len == 0:
        return False
    if not all(len(row) == first_len for row in value):  # type: ignore[arg-type]
        return False
    for row in value:  # type: ignore[assignment]
        for cell in row:
            if isinstance(cell, (list, tuple, dict)):
                return False
    return True


def _should_show_table(value: Any) -> bool:
    if not _is_matrix_value(value):
        return False
    rows = len(value)  # type: ignore[arg-type]
    cols = len(value[0])  # type: ignore[arg-type]
    return rows <= 5 and cols <= 5


def _bounded_summary(value: Any) -> str:
    if _is_matrix_value(value):
        return f"{len(value)} \u00d7 {len(value[0])} matrix"  # type: ignore[arg-type]
    if isinstance(value, (list, tuple)):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"map[{len(value)}]"
    text = repr(value)
    if len(text) > 48:
        return text[:45] + "..."
    return text


def _make_matrix_table(matrix: Sequence[Sequence[Any]]) -> QTableWidget:
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    table = QTableWidget(rows, cols)
    table.setObjectName("writebackProposedMatrixTable")
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # type: ignore[attr-defined]
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
    vertical_header = table.verticalHeader()
    if vertical_header is not None:
        vertical_header.setVisible(False)
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)  # type: ignore[attr-defined]
        vertical_header.setDefaultSectionSize(22)
    horizontal_header = table.horizontalHeader()
    if horizontal_header is not None:
        horizontal_header.setVisible(False)
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # type: ignore[attr-defined]
    for row in range(rows):
        for column in range(cols):
            value = matrix[row][column]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # type: ignore[attr-defined]
            table.setItem(row, column, item)
    table.setFixedHeight(rows * 22 + 4)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # type: ignore[attr-defined]
    table.setMinimumWidth(120)
    return table


# ---------------------------------------------------------------------------
# Private row helper — owns responsive arrangement for one item
# ---------------------------------------------------------------------------


class _WritebackRow(QFrame):
    """Compact ledger row with responsive reflow (S1/S3).

    Owns checkbox, current → proposed labels, arrow, action button, and an
    optional proposed-value matrix table. :meth:`set_narrow` moves those widgets
    between a single-line wide layout and a two-line narrow layout without
    duplicating widgets.
    """

    def __init__(
        self,
        *,
        cb: QCheckBox,
        current: QLabel,
        proposed: QLabel,
        action: QWidget,
        proposed_table: QTableWidget | None,
        actual_path: QLabel,
    ) -> None:
        super().__init__()
        self.setObjectName("writebackRow")
        self.setFrameShape(QFrame.Shape.NoFrame)  # type: ignore[attr-defined]
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # type: ignore[attr-defined]
        self._cb = cb
        self._cur = current
        self._arrow = QLabel("\u2192")
        self._arrow.setObjectName("writebackArrow")
        self._arrow.setStyleSheet("color: #8a94a3; font-weight: 700;")
        self._arrow.setFixedWidth(18)
        self._prop = proposed
        self._btn = action
        self._proposed_table = proposed_table
        self._actual_path = actual_path
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(8, 4, 8, 4)
        self._outer.setSpacing(2)
        self._heading = QHBoxLayout()
        self._heading.setSpacing(7)
        self._heading.addStretch(1)
        self._change = QHBoxLayout()
        self._change.setSpacing(7)
        self._cur.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._prop.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._change.addWidget(self._cur, 2)
        self._change.addWidget(self._arrow)
        self._change.addWidget(self._prop, 2)
        self._outer.addLayout(self._heading)
        self._outer.addLayout(self._change)
        self._outer.addWidget(self._actual_path)
        if self._proposed_table is not None:
            self._outer.addWidget(self._proposed_table)
        self.set_narrow(narrow=False)

    def set_narrow(self, *, narrow: bool) -> None:
        # Move only target/action; value and path widgets keep their layouts.
        self._heading.removeWidget(self._cb)
        self._heading.removeWidget(self._btn)
        self._change.removeWidget(self._cb)
        self._change.removeWidget(self._btn)
        if narrow:
            self._heading.insertWidget(0, self._cb)
            self._heading.addWidget(self._btn)
        else:
            self._change.insertWidget(0, self._cb, 3)
            self._change.addWidget(self._btn)
        self.updateGeometry()
        self.update()


class WritebackWidget(QWidget):
    """Compact unified writeback ledger (S1–S3).

    The service resolves captured destinations; this view renders its facts.

    - Draft-owned applied state projected as bold ``target*`` vs normal ``target``.
    - Centered Current → Proposed columns on a shared-background,
      continuous-boundary panel.
    - Equal 56×26 Edit/Copy actions; non-scalar MetaDict values use bounded
      summaries, while small proposed matrices also show a read-only matrix view.
      Current values remain summary-only and Copy retains the proposed JSON.
    - Width breakpoint near 450 px: wide rows single-line, narrow rows
      reflow to target/action above Current → Proposed.
    - Bordered ledger hugs its rendered rows; Apply Selected sits directly
      above the panel. Long content grows naturally and delegates
      vertical scrolling to the existing Analysis-pane outer scroll.
    """

    apply_requested: Signal = Signal()  # apply the persistent draft as-is

    def __init__(
        self,
        ctrl: Controller,
        parent: QWidget | None = None,
        *,
        tab_id: str,
        pane: WritebackPane = "analysis",
    ) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._tab_id = tab_id
        self._pane: WritebackPane = pane
        self._items: list[WritebackItem] = []
        self._checks: dict[str, QCheckBox] = {}
        self._applied_states: dict[str, bool] = {}

        # 13 px ledger per spec — item rows show current → proposed.
        font = self.font()
        font.setPixelSize(13)
        self.setFont(font)

        # Ledger panel stylesheet — unified backgrounds/borders (S1).
        self.setStyleSheet(
            "QFrame#writebackPanel { background: white; border: 1px solid #d7dde7; border-radius: 7px; }"
            "QFrame#writebackPanel QLabel, QFrame#writebackPanel QCheckBox { background: transparent; }"
            "QFrame#writebackRow { background: white; border: none; border-bottom: 1px solid #e8ecf2; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._hint = QLabel(
            "Select the items to write back. Use Edit to adjust values first."
        )
        self._hint.setWordWrap(True)
        self._hint.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,  # type: ignore[attr-defined]
        )
        layout.addWidget(self._hint)

        # Apply sits above the ledger so it stays close to the analyze output.
        self._apply_btn = QPushButton("Apply Selected")
        self._apply_btn.setObjectName("writebackApply")
        self._apply_btn.setFixedHeight(30)
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self._apply_btn)

        self._applied_legend = QLabel("* = not applied")
        self._applied_legend.setObjectName("writebackAppliedLegend")
        self._applied_legend.setStyleSheet("color: #6b7688;")
        layout.addWidget(self._applied_legend)

        self._rows_container = QFrame()
        self._rows_container.setObjectName("writebackPanel")
        self._rows_container.setFrameShape(QFrame.Shape.NoFrame)  # type: ignore[attr-defined]
        self._rows_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,  # type: ignore[attr-defined]
        )
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # type: ignore[attr-defined]
        layout.addWidget(self._rows_container)
        layout.addStretch(1)

        self._row_widgets: dict[str, tuple[QLabel, QLabel, QLabel, QCheckBox]] = {}
        self._rows: list[_WritebackRow] = []

        self._refresh_apply_enabled()

    def populate(
        self,
        items: Sequence[WritebackItem],
        values: Mapping[str, WritebackValues],
    ) -> None:
        """Render persistent items with service-owned captured preview facts.

        values maps every item session_id to its current preview. Missing keys
        raise KeyError; the view never resolves a target or substitutes active
        context. Rows show the logical target, actual path, current/proposed
        values and optional stderr in working units, without a unit label.
        Empty items clear the view. Items remain the persistent draft objects
        so checkbox and Edit commands use the same stable identities.
        Local cfg editing may retain invalid input. Its row shows proposed as
        "Invalid configuration" and current/path as "Unavailable (invalid
        configuration)", while preserving selection and updating unapplied
        state. Repairing input restores owner preview facts. Valid preview
        target/source/I/O failures still propagate; invalid input never becomes
        a missing value or an unmarked stale preview.
        """
        for item in items:
            values[item.session_id]
        self._values = values
        # Clear old rows
        while self._rows_layout.count():
            child = self._rows_layout.takeAt(0)
            if child is not None:
                w = child.widget()
                if w is not None:
                    w.deleteLater()

        # These are the *persistent* State items (ADR-0008) — do NOT copy. UI
        # edits (checkbox, value, cfg via the editor model) land on the same
        # objects the agent and apply read.
        self._items = list(items)
        self._applied_states = self._get_applied_states() if self._items else {}
        self._applied_legend.setVisible(bool(items))
        self._checks.clear()
        self._row_widgets.clear()
        self._rows.clear()

        for item in self._items:
            row = self._make_row(item)
            self._rows_layout.addWidget(row)
            self._rows.append(row)

        self._refresh_apply_enabled()
        self._update_responsive()

    def _make_row(self, item: WritebackItem) -> _WritebackRow:
        fact = self._values[item.session_id]
        cb = QCheckBox()
        cb.setChecked(item.selected)
        cb.setToolTip(item.description)
        self._update_applied_presentation(item, cb)
        cb.stateChanged.connect(lambda _state, it=item: self._on_check_toggled(it))
        self._checks[item.session_id] = cb

        current_label = QLabel(self._display_current(item))
        current_label.setObjectName("writebackCurrent")
        current_label.setStyleSheet("color: #5e6b7d;")
        current_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        current_label.setWordWrap(False)
        proposed_label, proposed_table = self._make_proposed_view(item, fact)
        actual_path = QLabel(fact.actual_path)
        actual_path.setObjectName("writebackActualPath")
        actual_path.setStyleSheet("color: #6b7688;")
        actual_path.setWordWrap(True)
        self._row_widgets[item.session_id] = (
            current_label,
            proposed_label,
            actual_path,
            cb,
        )
        row = _WritebackRow(
            cb=cb,
            current=current_label,
            proposed=proposed_label,
            action=self._make_action(item, fact),
            proposed_table=proposed_table,
            actual_path=actual_path,
        )
        row.setToolTip(item.description)
        return row

    def _make_proposed_view(
        self, item: WritebackItem, fact: WritebackValues
    ) -> tuple[QLabel, QTableWidget | None]:
        proposed_label = QLabel(self._display_proposed(item))
        is_matrix = isinstance(item, MetaDictWriteback) and _is_matrix_value(
            fact.proposed
        )
        if is_matrix:
            proposed_label.setObjectName("writebackProposedChip")
            proposed_label.setStyleSheet(
                "background: #eaf2fd; color: #1f5fae; border-radius: 8px; padding: 3px 7px; font-weight: 700;"
            )
        else:
            proposed_label.setObjectName("writebackProposed")
            proposed_label.setStyleSheet("color: #1f5fae; font-weight: 700;")
        proposed_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        proposed_label.setWordWrap(False)
        table = None
        if (
            is_matrix
            and isinstance(fact.proposed, (list, tuple))
            and _should_show_table(fact.proposed)
        ):
            table = _make_matrix_table(fact.proposed)
        return proposed_label, table

    def _make_action(self, item: WritebackItem, fact: WritebackValues) -> QWidget:
        if self._is_editable(item):
            button = QPushButton("Edit")
            button.setFixedSize(56, 26)
            button.clicked.connect(lambda _=False, it=item: self._edit_item(it))
            return button
        if isinstance(item, MetaDictWriteback) and not _is_scalar_md_value(
            fact.proposed
        ):
            button = QPushButton("Copy")
            button.setFixedSize(56, 26)
            button.clicked.connect(
                lambda _=False, val=fact.proposed: self._copy_value(val)
            )
            return button
        placeholder = QLabel("")
        placeholder.setFixedSize(56, 26)
        placeholder.setStyleSheet("background: transparent;")
        return placeholder

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_responsive()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._update_responsive()

    def _update_responsive(self) -> None:
        is_narrow = self.width() < 450
        for row in self._rows:
            row.set_narrow(narrow=is_narrow)

    def _copy_value(self, value: Any) -> None:
        try:
            text = json.dumps(value)
            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
        except Exception:
            logger.exception("failed to copy writeback value")

    def _on_check_toggled(self, item: WritebackItem) -> None:
        selected = self._checks[item.session_id].isChecked()
        assert self._tab_id is not None
        self._ctrl.set_writeback_item_for_pane(
            self._tab_id,
            self._pane,
            item.session_id,
            selected=selected,
        )
        self._refresh_apply_enabled()

    def _refresh_apply_enabled(self, *_: int) -> None:
        self._apply_btn.setEnabled(any(cb.isChecked() for cb in self._checks.values()))

    def _on_apply_clicked(self) -> None:
        self.apply_requested.emit()

    def _is_editable(self, item: WritebackItem) -> bool:
        if isinstance(item, MetaDictWriteback):
            # Scalar md values are hand-editable; a non-scalar (list/matrix, e.g.
            # the singleshot confusion matrix) is a derived value applied verbatim
            # — shown read-only, no Edit dialog (the scalar coercion can't parse a
            # matrix and the user does not hand-tune it).
            return _is_scalar_md_value(item.proposed_value)
        if isinstance(item, ModuleWriteback):
            return item.edit_schema is not None
        if isinstance(item, WaveformWriteback):
            return item.edit_schema is not None
        return False

    def _get_applied_states(self) -> dict[str, bool]:
        getter = getattr(self._ctrl, "get_writeback_applied_for_pane", None)
        if getter is None or not callable(getter):
            return {}
        result = getter(self._tab_id, self._pane)
        if not isinstance(result, dict):
            return {}
        return {str(key): bool(value) for key, value in result.items()}

    def _update_applied_presentation(
        self, item: WritebackItem, checkbox: QCheckBox
    ) -> None:
        applied = self._applied_states.get(item.session_id, False)
        checkbox.setText(item.target_name if applied else f"{item.target_name}*")
        font = checkbox.font()
        font.setBold(not applied)
        checkbox.setFont(font)
        checkbox.setStyleSheet("background: transparent;")

    def _refresh_item_applied_presentation(
        self, item: WritebackItem, checkbox: QCheckBox
    ) -> None:
        self._applied_states = self._get_applied_states()
        self._update_applied_presentation(item, checkbox)

    def _display_current(self, item: WritebackItem) -> str:
        current = self._values[item.session_id].current
        return "\u2014" if current is None else _bounded_summary(current)

    def _display_proposed(self, item: WritebackItem) -> str:
        proposed = _bounded_summary(self._values[item.session_id].proposed)
        if item.stderr is not None:
            proposed += f" ± {item.stderr}"
        return proposed

    def _refresh_item_preview(self, item: WritebackItem) -> None:
        self._values = self._ctrl.writeback_control.get_writeback_values_for_pane(
            self._tab_id, self._pane
        )
        fact = self._values[item.session_id]
        current_label, proposed_label, actual_path, checkbox = self._row_widgets[
            item.session_id
        ]
        current_label.setText(self._display_current(item))
        proposed_label.setText(self._display_proposed(item))
        actual_path.setText(fact.actual_path)
        self._refresh_item_applied_presentation(item, checkbox)

    def _refresh_cfg_item_preview(
        self, item: ModuleWriteback | WaveformWriteback, *, valid: bool
    ) -> None:
        # Incomplete edits belong to the attached model, not finished lowering.
        if not valid:
            current_label, proposed_label, actual_path, checkbox = self._row_widgets[
                item.session_id
            ]
            self._refresh_item_applied_presentation(item, checkbox)
            proposed_label.setText("Invalid configuration")
            current_label.setText("Unavailable (invalid configuration)")
            actual_path.setText("Unavailable (invalid configuration)")
            return
        self._refresh_item_preview(item)

    def _edit_item(self, item: WritebackItem) -> None:
        if isinstance(item, MetaDictWriteback):
            self._edit_md_item(item)
        elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
            self._edit_cfg_item(item)

    def _edit_md_item(self, item: MetaDictWriteback) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Value: {item.target_name}")
        layout = QVBoxLayout(dialog)

        form = QFormLayout()
        # Current is read-only (S2); target and proposed are editable.
        current_label = QLabel(self._display_current(item))
        current_label.setObjectName("writebackCurrentReadonly")
        current_label.setStyleSheet("color: #6b7688;")
        form.addRow("Current:", current_label)
        # target_name is the apply destination, decoupled from the stable
        # session_id (ADR-0008) — editable here so the user can retarget.
        name_edit = QLineEdit(item.target_name)
        form.addRow("Apply as:", name_edit)
        value_edit = QLineEdit(str(item.proposed_value))
        form.addRow("Proposed:", value_edit)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        cancel_btn.clicked.connect(dialog.reject)

        def save() -> None:
            try:
                new_name = _require_target_name(name_edit.text())
                new_value = _coerce_scalar_input(
                    value_edit.text(),
                    item.proposed_value,
                )
                assert self._tab_id is not None
                self._ctrl.set_writeback_item_for_pane(
                    self._tab_id,
                    self._pane,
                    item.session_id,
                    target_name=new_name,
                    proposed_value=new_value,
                )
                item.target_name = new_name
                item.proposed_value = new_value
                self._refresh_item_preview(item)
                dialog.accept()
            except Exception as exc:
                logger.exception("failed to edit writeback value %s", item.session_id)
                QMessageBox.critical(dialog, "Validation Error", str(exc))

        save_btn.clicked.connect(save)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # type: ignore[attr-defined]
        dialog.open()

    def _edit_cfg_item(
        self,
        item: ModuleWriteback | WaveformWriteback,
    ) -> None:
        assert self._tab_id is not None
        try:
            draft = self._ctrl.get_writeback_item_draft_for_pane(
                self._tab_id, self._pane, item.session_id
            )
        except Exception as exc:
            logger.exception(
                "failed to resolve writeback draft item %s", item.session_id
            )
            QMessageBox.critical(self, "Unable to edit writeback", str(exc))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Edit Config: {item.target_name}")
        dialog.setMinimumSize(560, 500)

        layout = QVBoxLayout(dialog)

        hint = QLabel("Edit the configuration below. Edits apply immediately.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # target_name is the apply destination, decoupled from the stable
        # session_id (ADR-0008) — editable so the user can retarget. Like the cfg
        # edits below, a valid change applies immediately; a blank field is left
        # on the previous name (revert on focus-out).
        name_row = QFormLayout()
        name_edit = QLineEdit(item.target_name)
        name_row.addRow("Apply as:", name_edit)
        layout.addLayout(name_row)

        def _commit_name() -> None:
            text = name_edit.text().strip()
            if not text:
                name_edit.setText(item.target_name)  # revert, no blank target
                return
            if text == item.target_name:
                return
            assert self._tab_id is not None
            self._ctrl.set_writeback_item_for_pane(
                self._tab_id,
                self._pane,
                item.session_id,
                target_name=text,
            )
            item.target_name = text
            self._refresh_cfg_item_preview(item, valid=draft.is_valid())

        name_edit.editingFinished.connect(_commit_name)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = CfgFormWidget(
            text_input_enhancer=make_value_source_input_enhancer(self._ctrl)
        )
        form_widget.attach(draft)

        def _on_draft_changed(*_: object) -> None:
            self._refresh_cfg_item_preview(item, valid=draft.is_valid())

        draft.on_change.connect(_on_draft_changed)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll, stretch=1)

        self._add_cfg_close_action(dialog, layout)

        def _on_finished(*_: object) -> None:
            try:
                _commit_name()
            finally:
                draft.on_change.disconnect(_on_draft_changed)
                form_widget.detach()
            self._refresh_cfg_item_preview(item, valid=draft.is_valid())

        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # type: ignore[attr-defined]
        dialog.finished.connect(_on_finished)
        dialog.open()

    def _add_cfg_close_action(self, dialog: QDialog, layout: QVBoxLayout) -> None:
        button_row = QHBoxLayout()
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)


def _is_scalar_md_value(value: Any) -> bool:
    """A md proposed_value the scalar Edit dialog can round-trip.

    Scalars (bool/int/float/complex/str/None) are hand-editable via
    ``_coerce_scalar_input``; a non-scalar (list/matrix, e.g. the confusion
    matrix) is a derived value applied verbatim, so the UI treats it read-only.
    """
    return value is None or isinstance(value, (bool, int, float, complex, str))


def _require_target_name(text: str) -> str:
    """Validate an apply-destination name (mirrors the tab.writeback_set guard)."""
    name = text.strip()
    if not name:
        raise RuntimeError("Apply-as name must not be empty")
    return name


def _coerce_scalar_input(text: str, original: Any) -> Any:
    if isinstance(original, bool):
        lowered = text.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise RuntimeError(f"Invalid bool value: {text}")
    if isinstance(original, int) and not isinstance(original, bool):
        return int(text)
    # complex before float: a complex md value (e.g. a single-shot IQ centre)
    # parses via Python's ``complex("1+2j")``. ``float`` would reject "1+2j", so
    # this branch must precede it. ``complex`` also accepts a bare real ("1.5"),
    # which keeps a real-only re-entry valid.
    if isinstance(original, complex):
        return complex(text.strip())
    if isinstance(original, float):
        return float(text)
    return text

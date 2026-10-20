from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import pyqtSignal
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QResizeEvent, QShowEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.widgets import (
    accent_colour,
    error_colour,
    secondary_colour,
    warning_colour,
)
from zcu_tools.gui.widgets.selectable_text import make_label_selectable

from .edit_dialogs import WritebackEditDialogs, is_scalar_value, module_changes
from .foregrounds import ThemedForegrounds
from .items import ModuleWriteback, ParameterWriteback, SampleWriteback, WritebackItem

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.controller import Controller

    from .control import WritebackPane
    from .service import WritebackValues

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Presentation helpers for non-scalar parameter context values (S2)
# ---------------------------------------------------------------------------


def _is_matrix_value(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return False
    if not all(isinstance(row, (list, tuple)) for row in value):
        return False
    first_len = len(value[0])
    if first_len == 0:
        return False
    if not all(len(row) == first_len for row in value):
        return False
    for row in value:
        for cell in row:
            if isinstance(cell, (list, tuple, dict)):
                return False
    return True


def _should_show_table(value: Any) -> bool:
    if not _is_matrix_value(value):
        return False
    rows = len(value)
    cols = len(value[0])
    return rows <= 5 and cols <= 5


def _bounded_summary(value: Any) -> str:
    if _is_matrix_value(value):
        return f"{len(value)} \u00d7 {len(value[0])} matrix"
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
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    vertical_header = table.verticalHeader()
    if vertical_header is not None:
        vertical_header.setVisible(False)
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        vertical_header.setDefaultSectionSize(22)
    horizontal_header = table.horizontalHeader()
    if horizontal_header is not None:
        horizontal_header.setVisible(False)
        horizontal_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    for row in range(rows):
        for column in range(cols):
            value = matrix[row][column]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, column, item)
    table.setFixedHeight(rows * 22 + 4)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    table.setMinimumWidth(120)
    return table


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, complex)) and not isinstance(value, bool)


def _compact_value(value: Any) -> str:
    """Six significant digits for numbers; bounded summaries for other values."""
    if _is_number(value):
        text = format(value, ".6g")
        # Keep the existing float spelling for small whole-valued previews.
        if isinstance(value, float) and text.lstrip("-").isdigit():
            text += ".0"
        return text
    return _bounded_summary(value)


def _full_value(value: Any) -> str:
    """Exact text for numbers (tooltips); bounded summaries for other values."""
    return repr(value) if _is_number(value) else _bounded_summary(value)


def _sample_summary(item: SampleWriteback, *, compact: bool) -> str:
    """List every sample field without truncating strings or nested JSON."""
    fields: list[str] = []
    for key, value in item.proposed_value.items():
        if isinstance(value, str):
            text = value
        elif _is_number(value):
            text = _compact_value(value) if compact else _full_value(value)
        else:
            text = json.dumps(value, ensure_ascii=False)
        unit = item.proposed_value.get("dev_unit")
        if key == "dev_value" and isinstance(unit, str) and unit:
            text += f" {unit}"
        fields.append(f"{key}={text}")
    return "\n".join(fields)


class _MiddleElidedLabel(QLabel):
    """Retain both ends of a preview as its available width changes.

    setText stores the full preview and shows it elided in the middle; the
    caller owns the tooltip.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.setText(text)

    def setText(self, a0: str | None) -> None:
        self._preview_text = a0 or ""
        self._update_text()

    def _update_text(self) -> None:
        super().setText(
            self.fontMetrics().elidedText(
                self._preview_text,
                Qt.TextElideMode.ElideMiddle,
                self.contentsRect().width(),
            )
        )

    def resizeEvent(self, a0: Any) -> None:
        super().resizeEvent(a0)
        self._update_text()


def _value_label(item: WritebackItem, text: str) -> QLabel:
    """Return a mouse-selectable label: middle-elided for a scalar parameter, else plain."""
    label: QLabel
    if isinstance(item, ParameterWriteback) and is_scalar_value(item.proposed_value):
        label = _MiddleElidedLabel(text)
    else:
        label = QLabel(text)
    make_label_selectable(label)
    return label


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
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._cb = cb
        self._cur = current
        self.arrow = QLabel("\u2192")
        self.arrow.setObjectName("writebackArrow")
        font = self.arrow.font()
        font.setBold(True)
        self.arrow.setFont(font)
        self.arrow.setFixedWidth(18)
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
        self.arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._change.addWidget(self._cur, 2)
        self._change.addWidget(self.arrow)
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
    - Equal 56×26 Edit/Copy actions; non-scalar parameter context values use bounded
      summaries, while small proposed matrices also show a read-only matrix view.
      Current values remain summary-only and Copy retains the proposed JSON.
    - Width breakpoint near 450 px: wide rows single-line, narrow rows
      reflow to target/action above Current → Proposed.
    - Bordered ledger hugs its rendered rows; Apply Selected sits directly
      above the panel. Long content grows naturally and delegates
      vertical scrolling to the existing Analysis-pane outer scroll.
    """

    apply_requested = pyqtSignal()  # apply the persistent draft as-is

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
        self._invalid_modules: set[str] = set()
        self._edit_dialogs = WritebackEditDialogs(self, ctrl, tab_id, pane)
        self._foregrounds = ThemedForegrounds(self)

        # 13 px ledger per spec — item rows show current → proposed.
        font = self.font()
        font.setPixelSize(13)
        self.setFont(font)

        # Ledger panel stylesheet — unified backgrounds/borders (S1).
        self.setStyleSheet(
            "QFrame#writebackPanel { background: palette(base); border: 1px solid palette(mid); border-radius: 7px; }"
            "QFrame#writebackPanel QLabel, QFrame#writebackPanel QCheckBox { background: transparent; }"
            "QFrame#writebackRow { background: palette(base); border: none; border-bottom: 1px solid palette(midlight); }"
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
            QSizePolicy.Policy.Maximum,
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
        self._foregrounds.set(self._applied_legend, secondary_colour)
        layout.addWidget(self._applied_legend)

        self._rows_container = QFrame()
        self._rows_container.setObjectName("writebackPanel")
        self._rows_container.setFrameShape(QFrame.Shape.NoFrame)
        self._rows_container.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Maximum,
        )
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._rows_container)
        layout.addStretch(1)

        self._row_widgets: dict[str, tuple[QLabel, QLabel, QLabel, QCheckBox]] = {}
        self._rows: list[_WritebackRow] = []

        self._refresh_apply_enabled()

    def changeEvent(self, a0: QEvent | None) -> None:
        super().changeEvent(a0)
        if a0 is not None and a0.type() == QEvent.Type.PaletteChange:
            self._foregrounds.refresh()

    def show_preview_unavailable(
        self, reason: str, *, unavailable_reasons: tuple[str, ...] = ()
    ) -> None:
        """Replace every row with one failure line naming why preview failed.

        reason is the owner's failure text, shown verbatim and wrapped, first.
        unavailable_reasons follow as populate shows them. The view keeps no
        items, so the selection hint and Apply are hidden until the next
        populate shows them again.
        """
        self.populate([], {}, unavailable_reasons=unavailable_reasons)
        self._hint.setVisible(False)
        self._apply_btn.setVisible(False)
        label = QLabel(f"Writeback preview unavailable: {reason}")
        make_label_selectable(label)
        label.setObjectName("writebackPreviewUnavailable")
        label.setTextFormat(Qt.TextFormat.PlainText)
        self._foregrounds.set(label, error_colour)
        label.setWordWrap(True)
        self._rows_layout.insertWidget(0, label)

    def populate(
        self,
        items: Sequence[WritebackItem],
        values: Mapping[str, WritebackValues],
        *,
        unavailable_reasons: tuple[str, ...] = (),
    ) -> None:
        """Render persistent items with service-owned captured preview facts.

        values maps every item session_id to its current preview. Missing keys
        raise KeyError; the view never resolves a target or substitutes active
        context. Rows show the logical target, actual path, current/proposed
        values and optional stderr in working units, without conversion. A
        service fact's display-only static unit follows a present current value
        and follows proposed stderr once. Missing current has no unit label;
        unit=None adds no whitespace. Empty items clear the view. Items remain
        the persistent draft objects so checkbox and Edit commands use the same
        stable identities. Sample rows identify the component with a sample suffix,
        show current as "entry sample", and list every proposed JSON field with
        device units after dev_value. Sample Copy preserves the complete JSON;
        samples have no Edit action.
        Local module editing may retain invalid State input. Its row shows proposed as
        "Invalid configuration" and current/path as "Unavailable (invalid
        configuration)", while preserving selection and updating unapplied
        state. Repairing input restores owner preview facts. Valid preview
        target/source/I/O failures still propagate; invalid input never becomes
        a missing value or an unmarked stale preview. Unparseable view text
        also marks the row invalid and locks Apply while its dialog is open.
        Closing drops that view text and restores the real State result, without
        writing the library. State model/expression errors survive closing.
        """
        for item in items:
            values[item.session_id]
        # Owner preview facts are valid. Only open views can still hold local
        # parse errors; closed State errors must be repaired before this read.
        self._invalid_modules.intersection_update(self._edit_dialogs.open_module_ids())
        self._render_rows(items, values, unavailable_reasons=unavailable_reasons)

    def refresh_module_validity(
        self,
        items: Sequence[WritebackItem],
        *,
        unavailable_reasons: tuple[str, ...] = (),
    ) -> None:
        """Reproject items when strict bulk preview rejects an invalid module.

        items are the persistent pane items, as in populate. Read each module's
        shared Form result, retaining open local errors. Invalid module rows keep
        selection and Edit but show Invalid configuration with unavailable current
        value/path. Read valid items through the service's single-item preview;
        parameter/sample facts are unchanged. Rebuild rows without closing dialogs
        or modifying drafts/library. unavailable_reasons follow the rows.
        Form access, target/source and I/O failures propagate; no preview values
        are fabricated and the bulk preview contract is not changed.
        """
        invalid: set[str] = set()
        values: dict[str, WritebackValues] = {}
        for item in items:
            if isinstance(item, ModuleWriteback):
                form = self._ctrl.get_writeback_item_form_for_pane(
                    self._tab_id, self._pane, item.session_id
                )
                local_invalid = (
                    item.session_id in self._edit_dialogs.open_module_ids()
                    and item.session_id in self._invalid_modules
                )
                if local_invalid or isinstance(form.result(), FormErrors):
                    invalid.add(item.session_id)
                    continue
            values.update(
                self._ctrl.writeback_control.get_writeback_values_for_pane(
                    self._tab_id, self._pane, session_id=item.session_id
                )
            )
        self._invalid_modules = invalid
        self._render_rows(items, values, unavailable_reasons=unavailable_reasons)

    def _render_rows(
        self,
        items: Sequence[WritebackItem],
        values: Mapping[str, WritebackValues],
        *,
        unavailable_reasons: tuple[str, ...],
    ) -> None:
        self._values = values
        self._hint.setVisible(True)
        self._apply_btn.setVisible(True)
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
            if (
                isinstance(item, ModuleWriteback)
                and item.session_id in self._invalid_modules
            ):
                self._refresh_cfg_item_preview(item, valid=False)

        for reason in unavailable_reasons:
            label = QLabel(f"⚠ {reason}")
            make_label_selectable(label)
            self._foregrounds.set(label, warning_colour)
            label.setObjectName("writebackUnavailableReason")
            label.setWordWrap(True)
            self._rows_layout.addWidget(label)
        self._foregrounds.refresh()
        self._refresh_apply_enabled()
        self._update_responsive()

    def _make_row(self, item: WritebackItem) -> _WritebackRow:
        cb = QCheckBox()
        cb.setChecked(item.selected)
        cb.setToolTip(item.description)
        self._update_applied_presentation(item, cb)
        cb.stateChanged.connect(lambda _state, it=item: self._on_check_toggled(it))
        self._checks[item.session_id] = cb

        current_label = _value_label(item, self._display_current(item, compact=True))
        current_label.setToolTip(self._display_current(item, compact=False))
        current_label.setObjectName("writebackCurrent")
        self._foregrounds.set(current_label, secondary_colour)
        current_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        current_label.setWordWrap(False)
        proposed_label, proposed_table = self._make_proposed_view(item)
        actual_path = QLabel(
            "Unavailable (invalid configuration)"
            if item.session_id in self._invalid_modules
            else self._values[item.session_id].actual_path
        )
        make_label_selectable(actual_path)
        actual_path.setObjectName("writebackActualPath")
        self._foregrounds.set(actual_path, secondary_colour)
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
            action=self._make_action(item),
            proposed_table=proposed_table,
            actual_path=actual_path,
        )
        self._foregrounds.set(row.arrow, secondary_colour)
        row.setToolTip(item.description)
        return row

    def _make_proposed_view(
        self, item: WritebackItem
    ) -> tuple[QLabel, QTableWidget | None]:
        proposed_label = _value_label(item, self._display_proposed(item, compact=True))
        proposed_label.setToolTip(self._display_proposed(item, compact=False))
        is_matrix = isinstance(item, ParameterWriteback) and _is_matrix_value(
            self._values[item.session_id].proposed
        )
        if is_matrix:
            proposed_label.setObjectName("writebackProposedChip")
            proposed_label.setStyleSheet(
                "background: #eaf2fd; color: #1f5fae; border-radius: 8px; padding: 3px 7px; font-weight: 700;"
            )
        else:
            proposed_label.setObjectName("writebackProposed")
            font = proposed_label.font()
            font.setBold(True)
            proposed_label.setFont(font)
            self._foregrounds.set(proposed_label, accent_colour)
        proposed_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        proposed_label.setWordWrap(isinstance(item, SampleWriteback))
        if isinstance(item, SampleWriteback):
            proposed_label.setTextFormat(Qt.TextFormat.PlainText)
        table = None
        if is_matrix:
            proposed = self._values[item.session_id].proposed
            if isinstance(proposed, (list, tuple)) and _should_show_table(proposed):
                table = _make_matrix_table(proposed)
        return proposed_label, table

    def _make_action(self, item: WritebackItem) -> QWidget:
        if self._is_editable(item):
            button = QPushButton("Edit")
            button.setFixedSize(56, 26)
            button.clicked.connect(lambda _=False, it=item: self._edit_item(it))
            return button
        if isinstance(item, SampleWriteback) or (
            isinstance(item, ParameterWriteback)
            and not is_scalar_value(self._values[item.session_id].proposed)
        ):
            button = QPushButton("Copy")
            button.setFixedSize(56, 26)
            button.clicked.connect(
                lambda _=False, val=self._values[item.session_id].proposed: (
                    self._copy_value(val)
                )
            )
            return button
        placeholder = QLabel("")
        placeholder.setFixedSize(56, 26)
        placeholder.setStyleSheet("background: transparent;")
        return placeholder

    def resizeEvent(self, a0: QResizeEvent | None) -> None:
        super().resizeEvent(a0)
        self._update_responsive()

    def showEvent(self, a0: QShowEvent | None) -> None:
        super().showEvent(a0)
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
        selected = {key for key, cb in self._checks.items() if cb.isChecked()}
        invalid = selected & self._invalid_modules
        self._apply_btn.setEnabled(bool(selected) and not bool(invalid))
        if invalid:
            targets = [
                item.target_name for item in self._items if item.session_id in invalid
            ]
            self._hint.setText(f"Fix or uncheck invalid module: {', '.join(targets)}")
            self._foregrounds.set(self._hint, warning_colour)
        else:
            self._hint.setText(
                "Select the items to write back. Use Edit to adjust values first."
            )
            self._foregrounds.set(
                self._hint, lambda owner: owner.palette().color(owner.foregroundRole())
            )

    def _on_apply_clicked(self) -> None:
        self.apply_requested.emit()

    def _is_editable(self, item: WritebackItem) -> bool:
        if isinstance(item, ParameterWriteback):
            # Scalar md values are hand-editable; a non-scalar (list/matrix, e.g.
            # the singleshot confusion matrix) is a derived value applied verbatim
            # — shown read-only, no Edit dialog (the scalar coercion can't parse a
            # matrix and the user does not hand-tune it).
            return is_scalar_value(item.proposed_value)
        return isinstance(item, ModuleWriteback)

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
        target = (
            f"{item.target_name} · sample"
            if isinstance(item, SampleWriteback)
            else item.target_name
        )
        checkbox.setText(target if applied else f"{target}*")
        font = checkbox.font()
        font.setBold(not applied)
        checkbox.setFont(font)
        checkbox.setStyleSheet("background: transparent;")

    def _refresh_item_applied_presentation(
        self, item: WritebackItem, checkbox: QCheckBox
    ) -> None:
        self._applied_states = self._get_applied_states()
        self._update_applied_presentation(item, checkbox)

    def _display_current(self, item: WritebackItem, *, compact: bool) -> str:
        """Current value text: compact for the label, full for its tooltip."""
        if item.session_id in self._invalid_modules:
            return "Unavailable (invalid configuration)"
        values = self._values[item.session_id]
        if isinstance(item, SampleWriteback):
            return "entry sample"
        if isinstance(item, ModuleWriteback):
            if not compact:
                return self._module_tooltip(item)
            return "new entry" if values.current is None else "library entry"
        if values.current is None:
            return "\u2014"
        current = (
            _compact_value(values.current) if compact else _full_value(values.current)
        )
        if values.unit is not None:
            current += f" {values.unit}"
        return current

    def _display_proposed(self, item: WritebackItem, *, compact: bool) -> str:
        """Proposed value (± stderr) text: compact for the label, full for its tooltip."""
        if item.session_id in self._invalid_modules:
            return "Invalid configuration"
        values = self._values[item.session_id]
        if isinstance(item, SampleWriteback):
            return _sample_summary(item, compact=compact)
        if isinstance(item, ModuleWriteback):
            if not compact:
                return self._module_tooltip(item)
            count = len(module_changes(values.current, values.proposed))
            if count == 1:
                return "1 change"
            return f"{count} changes" if count else "no changes"
        proposed = (
            _compact_value(values.proposed) if compact else _full_value(values.proposed)
        )
        if item.stderr is not None:
            stderr = (
                _compact_value(item.stderr) if compact else _full_value(item.stderr)
            )
            proposed += f" ± {stderr}"
        if values.unit is not None:
            proposed += f" {values.unit}"
        return proposed

    def _module_tooltip(self, item: ModuleWriteback) -> str:
        values = self._values[item.session_id]
        lines = [
            change.text() for change in module_changes(values.current, values.proposed)
        ]
        if values.current is None:
            lines.insert(0, "New library entry")
        return "\n".join(lines) if lines else "No changes"

    def _refresh_item_preview(self, item: WritebackItem) -> None:
        self._values = {
            **self._values,
            **self._ctrl.writeback_control.get_writeback_values_for_pane(
                self._tab_id, self._pane, session_id=item.session_id
            ),
        }
        fact = self._values[item.session_id]
        current_label, proposed_label, actual_path, checkbox = self._row_widgets[
            item.session_id
        ]
        current_label.setText(self._display_current(item, compact=True))
        current_label.setToolTip(self._display_current(item, compact=False))
        proposed_label.setText(self._display_proposed(item, compact=True))
        proposed_label.setToolTip(self._display_proposed(item, compact=False))
        actual_path.setText(fact.actual_path)
        self._refresh_item_applied_presentation(item, checkbox)

    def _refresh_cfg_item_preview(self, item: ModuleWriteback, *, valid: bool) -> None:
        if not valid:
            self._invalid_modules.add(item.session_id)
            current_label, proposed_label, actual_path, checkbox = self._row_widgets[
                item.session_id
            ]
            self._refresh_item_applied_presentation(item, checkbox)
            proposed_label.setText("Invalid configuration")
            current_label.setText("Unavailable (invalid configuration)")
            actual_path.setText("Unavailable (invalid configuration)")
            proposed_label.setToolTip("Invalid configuration")
            current_label.setToolTip("Unavailable (invalid configuration)")
        else:
            self._invalid_modules.discard(item.session_id)
            self._refresh_item_preview(item)
        proposed_label = self._row_widgets[item.session_id][1]
        self._foregrounds.set(proposed_label, accent_colour if valid else error_colour)
        self._refresh_apply_enabled()

    def _edit_item(self, item: WritebackItem) -> None:
        if isinstance(item, ParameterWriteback):
            self._edit_dialogs.edit_value(
                item,
                self._values[item.session_id],
                lambda: self._refresh_item_preview(item),
            )
        elif isinstance(item, ModuleWriteback):
            self._edit_dialogs.edit_module(
                item, lambda valid: self._refresh_cfg_item_preview(item, valid=valid)
            )

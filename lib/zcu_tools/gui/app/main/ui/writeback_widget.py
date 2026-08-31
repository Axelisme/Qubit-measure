from __future__ import annotations

import json
import logging
from collections.abc import Sequence
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

from zcu_tools.gui.app.main.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
    WritebackItem,
)
from zcu_tools.gui.app.main.ui.cfg_binding import make_value_source_input_enhancer
from zcu_tools.gui.widgets.cfg import CfgFormWidget

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller
    from zcu_tools.gui.app.main.services.writeback_control import WritebackPane

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
    table.setObjectName("writebackMatrixTable")
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # type: ignore[attr-defined]
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
    vh0 = table.verticalHeader()
    if vh0 is not None:
        vh0.setVisible(False)
    hh0 = table.horizontalHeader()
    if hh0 is not None:
        hh0.setVisible(False)
    for r in range(rows):
        for c in range(cols):
            val = matrix[r][c]  # type: ignore[index]
            if isinstance(val, float):
                txt = f"{val:.4f}"
            else:
                txt = str(val)
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # type: ignore[attr-defined]
            table.setItem(r, c, item)
    hh = table.horizontalHeader()
    if hh is not None:
        hh.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)  # type: ignore[attr-defined]
    vh = table.verticalHeader()
    if vh is not None:
        vh.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)  # type: ignore[attr-defined]
        vh.setDefaultSectionSize(22)
    total_h = rows * 22 + 4
    table.setFixedHeight(total_h)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # type: ignore[attr-defined]
    table.setMinimumWidth(120)
    return table


# ---------------------------------------------------------------------------
# Private row helper — owns responsive arrangement for one item
# ---------------------------------------------------------------------------


class _WritebackRow(QFrame):
    """Compact ledger row with responsive reflow (S1/S3).

    Owns checkbox, current → proposed labels, arrow, action button and
    optional inline matrix table. :meth:`set_narrow` moves those widgets
    between a single-line wide layout and a two-line narrow layout without
    duplicating widgets.
    """

    def __init__(
        self,
        cb: QCheckBox,
        current: QLabel,
        arrow: QLabel,
        proposed: QLabel,
        action: QWidget,
        table: QTableWidget | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("writebackRow")
        self.setFrameShape(QFrame.Shape.NoFrame)  # type: ignore[attr-defined]
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)  # type: ignore[attr-defined]
        self._cb = cb
        self._cur = current
        self._arrow = arrow
        self._prop = proposed
        self._btn = action
        self._table = table
        self._narrow = False
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(8, 4, 8, 4)
        self._outer.setSpacing(2)
        # initial wide layout
        self.set_narrow(False)

    def set_narrow(self, narrow: bool) -> None:
        if narrow == self._narrow and self._outer.count() != 0:
            return
        self._narrow = narrow
        # Clear outer without deleting the row's owned widgets.
        while self._outer.count():
            item = self._outer.takeAt(0)
            if item is None:
                continue
            w = item.widget()
            if w is not None:
                w.setParent(self)
                continue
            lay = item.layout()
            if lay is not None:
                while lay.count():
                    sub = lay.takeAt(0)
                    if sub is None:
                        continue
                    sw = sub.widget()
                    if sw is not None:
                        sw.setParent(self)
                        continue
                    inner = sub.layout()
                    if inner is not None:
                        while inner.count():
                            sj = inner.takeAt(0)
                            if sj is None:
                                continue
                            sjw = sj.widget()
                            if sjw is not None:
                                sjw.setParent(self)
                        inner.deleteLater()
                lay.deleteLater()
        if not narrow:
            primary = QHBoxLayout()
            primary.setSpacing(7)
            primary.addWidget(self._cb, 3)
            self._cur.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            primary.addWidget(self._cur, 2)
            primary.addWidget(self._arrow)
            self._prop.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            primary.addWidget(self._prop, 2)
            primary.addWidget(self._btn)
            self._outer.addLayout(primary)
            if self._table is not None:
                self._outer.addWidget(self._table)
        else:
            heading = QHBoxLayout()
            heading.setSpacing(7)
            heading.addWidget(self._cb)
            heading.addStretch(1)
            heading.addWidget(self._btn)
            self._outer.addLayout(heading)

            change = QHBoxLayout()
            change.setSpacing(7)
            self._cur.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            change.addWidget(self._cur, 1)
            change.addWidget(self._arrow)
            self._prop.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            change.addWidget(self._prop, 1)
            self._outer.addLayout(change)

            if self._table is not None:
                self._outer.addWidget(self._table)
        self.updateGeometry()
        self.update()


class WritebackWidget(QWidget):
    """Compact unified writeback ledger (S1–S3).

    Presentation is app-local; no data-model or wire contract is introduced.

    - Target-only checkbox labels with description tooltips.
    - Centered Current → Proposed columns on a shared-background,
      continuous-boundary panel.
    - Equal 56×26 Edit/Copy actions; non-scalar MetaDict values show a
      bounded summary (``3 × 3 matrix`` for matrices) and a compact
      read-only inline table with JSON Copy.
    - Width breakpoint near 450 px: wide rows single-line, narrow rows
      reflow to target/action above Current → Proposed.
    - Ledger is vertically scrollable without horizontal overflow;
      Apply Selected stays fixed at the bottom.
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
        layout.addWidget(self._hint)

        self._rows_container = QFrame()
        self._rows_container.setObjectName("writebackPanel")
        self._rows_container.setFrameShape(QFrame.Shape.NoFrame)  # type: ignore[attr-defined]
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        self._rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # type: ignore[attr-defined]

        self._scroll = QScrollArea()
        self._scroll.setObjectName("writebackScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)  # type: ignore[attr-defined]
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self._scroll.setWidget(self._rows_container)
        layout.addWidget(self._scroll, stretch=1)

        self._apply_btn = QPushButton("Apply Selected")
        self._apply_btn.setObjectName("writebackApply")
        self._apply_btn.setFixedHeight(30)
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self._apply_btn)

        self._row_widgets: dict[str, tuple[QLabel, QLabel, QCheckBox]] = {}
        self._rows: list[_WritebackRow] = []

        self._refresh_apply_enabled()

    def populate(self, items: Sequence[WritebackItem]) -> None:
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
        self._checks.clear()
        self._row_widgets.clear()
        self._rows.clear()

        for item in self._items:
            # Selection — target identity only (S1); description moves to tooltip.
            cb = QCheckBox(self._make_label_text(item))
            cb.setChecked(item.selected)
            cb.setToolTip(item.description)
            cb.setStyleSheet("font-weight: 700; background: transparent;")
            cb.stateChanged.connect(lambda _state, it=item: self._on_check_toggled(it))
            self._checks[item.session_id] = cb

            current_text = self._display_current(item)
            proposed_text = self._display_proposed(item)

            current_label = QLabel(current_text)
            current_label.setObjectName("writebackCurrent")
            current_label.setStyleSheet("color: #5e6b7d;")
            current_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            current_label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )  # type: ignore[attr-defined]
            current_label.setWordWrap(False)

            arrow = QLabel("\u2192")
            arrow.setObjectName("writebackArrow")
            arrow.setStyleSheet("color: #8a94a3; font-weight: 700;")
            arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            arrow.setFixedWidth(18)

            is_md = isinstance(item, MetaDictWriteback)
            is_nonscalar_md = is_md and not _is_scalar_md_value(item.proposed_value)  # type: ignore[attr-defined]
            is_matrix = (
                is_md and _is_matrix_value(item.proposed_value) and is_nonscalar_md
            )  # type: ignore[attr-defined]

            proposed_label = QLabel(proposed_text)
            if is_matrix:
                proposed_label.setObjectName("writebackProposedChip")
                proposed_label.setStyleSheet(
                    "background: #eaf2fd; color: #1f5fae; border-radius: 8px; padding: 3px 7px; font-weight: 700;"
                )
            else:
                proposed_label.setObjectName("writebackProposed")
                proposed_label.setStyleSheet("color: #1f5fae; font-weight: 700;")
            proposed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # type: ignore[attr-defined]
            proposed_label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )  # type: ignore[attr-defined]
            proposed_label.setWordWrap(False)

            # Action — equal 56×26 geometry (S1). Editable → Edit, non-scalar md → Copy.
            action: QWidget
            if self._is_editable(item):
                btn = QPushButton("Edit")
                btn.setFixedSize(56, 26)
                btn.clicked.connect(
                    lambda _=False, it=item, chk=cb: self._edit_item(it, chk)
                )
                action = btn
            elif isinstance(item, MetaDictWriteback) and not _is_scalar_md_value(
                item.proposed_value
            ):
                btn = QPushButton("Copy")
                btn.setFixedSize(56, 26)
                md_val = item.proposed_value  # type: ignore[attr-defined]
                btn.clicked.connect(lambda _=False, val=md_val: self._copy_value(val))
                action = btn
            else:
                placeholder = QLabel("")
                placeholder.setFixedSize(56, 26)
                placeholder.setStyleSheet("background: transparent;")
                action = placeholder

            table: QTableWidget | None = None
            if (
                isinstance(item, MetaDictWriteback)
                and not _is_scalar_md_value(item.proposed_value)
                and _should_show_table(item.proposed_value)
            ):  # type: ignore[attr-defined]
                table = _make_matrix_table(item.proposed_value)  # type: ignore[arg-type]

            row = _WritebackRow(cb, current_label, arrow, proposed_label, action, table)
            row.setToolTip(item.description)

            self._rows_layout.addWidget(row)
            self._rows.append(row)

            self._row_widgets[item.session_id] = (
                current_label,
                proposed_label,
                cb,
            )

        self._refresh_apply_enabled()
        self._update_responsive()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_responsive()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._update_responsive()

    def _update_responsive(self) -> None:
        is_narrow = self.width() < 450
        for row in self._rows:
            row.set_narrow(is_narrow)

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

    def _get_service_summaries(self, session_id: str) -> tuple[str | None, str | None]:
        """Fetch S2 summaries from the service-owned draft (app-local)."""
        try:
            getter = getattr(self._ctrl, "get_writeback_summaries_for_pane", None)
            if getter is None or not callable(getter):
                return (None, None)
            result = getter(self._tab_id, self._pane)  # type: ignore[call-arg]
            if isinstance(result, dict) and session_id in result:
                cur, prop = result[session_id]
                return cur, prop
        except Exception:
            pass
        return (None, None)

    def _display_current(self, item: WritebackItem) -> str:
        cur, _ = self._get_service_summaries(item.session_id)
        if cur is not None:
            return str(cur)
        if isinstance(item, MetaDictWriteback):
            return "\u2014"
        return "\u2014"

    def _display_proposed(self, item: WritebackItem) -> str:
        # Non-scalar MetaDict values use a bounded structural summary (S2)
        # so the ledger never widens; the full JSON is available via Copy.
        if isinstance(item, MetaDictWriteback) and not _is_scalar_md_value(
            item.proposed_value
        ):
            return _bounded_summary(item.proposed_value)
        _, prop = self._get_service_summaries(item.session_id)
        if prop is not None:
            return str(prop)
        if isinstance(item, MetaDictWriteback):
            return repr(item.proposed_value)
        if isinstance(item, (ModuleWriteback, WaveformWriteback)):
            return f"\u2192 {item.target_name}"
        return f"{item.target_name}"

    def _make_label_text(self, item: WritebackItem) -> str:
        # S1: target identity only; description lives in tooltip, proposed
        # value is shown in its own column and never duplicated here.
        return item.target_name

    def _edit_item(self, item: WritebackItem, cb: QCheckBox) -> None:
        if isinstance(item, MetaDictWriteback):
            self._edit_md_item(item, cb)
        elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
            self._edit_cfg_item(item, cb)

    def _edit_md_item(self, item: MetaDictWriteback, cb: QCheckBox) -> None:
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
                cb.setText(self._make_label_text(item))
                # Update ledger row from service-owned summary (S2)
                row_tuple_md = self._row_widgets.get(item.session_id)
                if row_tuple_md is not None:
                    _cur_md, proposed_label_md, _cb_md = row_tuple_md
                    proposed_label_md.setText(self._display_proposed(item))
                dialog.accept()
            except Exception as exc:
                QMessageBox.critical(dialog, "Validation Error", str(exc))

        save_btn.clicked.connect(save)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # type: ignore[attr-defined]
        dialog.open()

    def _edit_cfg_item(
        self,
        item: ModuleWriteback | WaveformWriteback,
        cb: QCheckBox,
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
            assert self._tab_id is not None
            self._ctrl.set_writeback_item_for_pane(
                self._tab_id,
                self._pane,
                item.session_id,
                target_name=text,
            )
            item.target_name = text

        name_edit.editingFinished.connect(_commit_name)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = CfgFormWidget(
            text_input_enhancer=make_value_source_input_enhancer(self._ctrl)
        )
        form_widget.attach(draft)
        scroll.setWidget(form_widget)
        layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        close_btn = QPushButton("Close")
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        close_btn.clicked.connect(dialog.accept)

        def _on_finished(*_: Any) -> None:
            _commit_name()
            form_widget.detach()
            cb.setText(self._make_label_text(item))
            # Refresh bounded summary after cfg edits (proposed may have changed)
            row_tuple = self._row_widgets.get(item.session_id)
            if row_tuple is not None:
                _cur_cfg, proposed_label_cfg, _cb_cfg = row_tuple
                # For cfg items, proposed_summary stays bounded; keep existing
                proposed_label_cfg.setText(self._display_proposed(item))

        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)  # type: ignore[attr-defined]
        dialog.finished.connect(_on_finished)
        dialog.open()


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

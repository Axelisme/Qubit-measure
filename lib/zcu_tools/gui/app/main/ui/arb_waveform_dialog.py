from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from qtpy.QtCore import Qt, QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.meta_tool import (
    ArbWaveformData,
    ArbWaveformError,
    FormulaRecipe,
    prepare_preview_series,
    render_formula_recipe,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.controller import Controller


_DATA_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# Evaluated once at import; from_raw is cheap (structure validation only, no render).
_DEFAULT_RECIPE: FormulaRecipe = FormulaRecipe.from_raw(
    {
        "segments": [
            {"duration": 0.2, "formula": "exp(-0.5*((t-0.2)/0.05)**2)"},
            {"duration": 0.6, "formula": "1"},
            {"duration": 0.2, "formula": "exp(-0.5*(t/0.05)**2)"},
        ],
        "normalize": "peak",
    }
)
_PREVIEW_DEBOUNCE_MS = 250


class _PreviewCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(7.0, 3.0), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._ax = self.figure.subplots()
        self._time: NDArray[np.float64] | None = None
        self._idata: NDArray[np.float64] | None = None
        self._qdata: NDArray[np.float64] | None = None
        self._abs_data: NDArray[np.float64] | None = None
        self._marker: Line2D | None = None
        self._i_line: Line2D | None = None
        self._q_line: Line2D | None = None
        self._abs_line: Line2D | None = None
        self._dragging = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.clear("No waveform selected")

    def clear(self, message: str = "") -> None:
        self._time = None
        self._idata = None
        self._qdata = None
        self._abs_data = None
        self._marker = None
        self._i_line = None
        self._q_line = None
        self._abs_line = None
        self._ax.clear()
        self._ax.set_xlabel("Time (us)")
        self._ax.set_ylabel("Normalized amplitude")
        self._ax.set_ylim(-1.05, 1.05)
        self._ax.grid(True, alpha=0.3)
        if message:
            self._ax.text(0.5, 0.5, message, transform=self._ax.transAxes, ha="center")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def plot(self, data_key: str, data: ArbWaveformData, *, normalize: bool) -> None:
        # Delegate normalize + I/Q/Abs computation to the domain helper (ADR-0034).
        series = prepare_preview_series(data, normalize=normalize)

        self._time = series.time
        self._idata = series.idata
        self._qdata = series.qdata
        self._abs_data = series.abs_data

        self._ax.clear()
        (self._i_line,) = self._ax.plot(series.time, series.idata, label="I:0.00")
        (self._q_line,) = self._ax.plot(series.time, series.qdata, label="Q:0.00")
        (self._abs_line,) = self._ax.plot(
            series.time, series.abs_data, label="Abs:0.00"
        )
        self._marker = self._ax.axvline(
            float(series.time[0]), color="black", linewidth=1.0
        )
        self._ax.set_title(data_key)
        self._ax.set_xlabel("Time (us)")
        self._ax.set_ylabel("Normalized amplitude" if normalize else "Amplitude")
        self._ax.set_xlim(float(series.time[0]), float(series.time[-1]))
        self._ax.set_ylim(self._data_ylim(series.idata, series.qdata, series.abs_data))
        self._ax.grid(True, alpha=0.3)
        self._move_marker(float(series.time[0]), draw=False)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _data_ylim(
        idata: NDArray[np.float64],
        qdata: NDArray[np.float64],
        abs_data: NDArray[np.float64],
    ) -> tuple[float, float]:
        low = min(
            float(np.min(idata)), float(np.min(qdata)), float(np.min(abs_data)), 0.0
        )
        high = max(
            float(np.max(idata)), float(np.max(qdata)), float(np.max(abs_data)), 0.0
        )
        if low == high:
            pad = max(abs(low) * 0.05, 0.05)
        else:
            pad = max((high - low) * 0.05, 0.05)
        return low - pad, high + pad

    def _on_press(self, event: object) -> None:
        if getattr(event, "inaxes", None) is not self._ax:
            return
        xdata = getattr(event, "xdata", None)
        if xdata is None:
            return
        self._dragging = True
        self._move_marker(float(xdata), draw=True)

    def _on_move(self, event: object) -> None:
        if not self._dragging or getattr(event, "inaxes", None) is not self._ax:
            return
        xdata = getattr(event, "xdata", None)
        if xdata is not None:
            self._move_marker(float(xdata), draw=True)

    def _on_release(self, event: object) -> None:
        if self._dragging:
            xdata = getattr(event, "xdata", None)
            if xdata is not None:
                self._move_marker(float(xdata), draw=True)
        self._dragging = False

    def _move_marker(self, x: float, *, draw: bool) -> None:
        if (
            self._time is None
            or self._idata is None
            or self._qdata is None
            or self._abs_data is None
            or self._marker is None
            or self._i_line is None
            or self._q_line is None
            or self._abs_line is None
        ):
            return
        left = float(self._time[0])
        right = float(self._time[-1])
        x = min(max(x, left), right)
        self._marker.set_xdata([x, x])
        i_val = float(np.interp(x, self._time, self._idata))
        q_val = float(np.interp(x, self._time, self._qdata))
        abs_val = float(np.interp(x, self._time, self._abs_data))
        self._i_line.set_label(f"I:{i_val:.2f}")
        self._q_line.set_label(f"Q:{q_val:.2f}")
        self._abs_line.set_label(f"Abs:{abs_val:.2f}")
        self._ax.legend(loc="upper right")
        if draw:
            self.canvas.draw_idle()


class ArbWaveformDialog(QDialog):
    """Qubit-scoped arbitrary waveform asset manager."""

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._current_data_key: str | None = None
        self._suppress_segment_change = False
        # Cheap path state: updated synchronously on every structure/data_key change.
        # _valid_recipe is set by _validate_structure (cheap, no render).
        self._valid_recipe: FormulaRecipe | None = None
        self._data_key_error: str | None = None
        self._structure_error: str | None = None
        # Deep path state: updated only by the debounce timer callback.
        self._valid_data: ArbWaveformData | None = None
        self._render_error: str | None = None
        # Debounce timer fires _render_preview (the expensive render path).
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._render_preview)

        self.setWindowTitle("Arbitrary Waveforms")
        self.setMinimumSize(980, 720)

        root = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addStretch()
        self._close_btn = QPushButton("Close")
        header.addWidget(self._close_btn)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Vertical)
        root.addWidget(splitter)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.addWidget(self._build_asset_group(), stretch=2)
        top_layout.addWidget(self._build_recipe_group(), stretch=3)
        splitter.addWidget(top)

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self._preview = _PreviewCanvas()
        preview_layout.addWidget(self._preview)
        splitter.addWidget(preview_group)
        splitter.setSizes([360, 360])

        self._close_btn.clicked.connect(self.reject)
        self.refresh()
        self._reset_draft()

    def refresh(self) -> None:
        selected = self._current_data_key
        infos = self._ctrl.list_arb_waveform_infos()
        self._asset_table.setRowCount(0)
        for row, info in enumerate(infos):
            self._asset_table.insertRow(row)
            values = (
                info.data_key,
                f"{info.duration:.2f}",
                datetime.fromtimestamp(info.mtime).strftime("%Y-%m-%d %H:%M:%S"),
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._asset_table.setItem(row, col, item)
        if selected:
            self._select_data_key(selected)
        self._revalidate_cheap()

    def _build_asset_group(self) -> QGroupBox:
        group = QGroupBox("Assets")
        layout = QVBoxLayout(group)

        self._asset_table = QTableWidget(0, 3)
        self._asset_table.setHorizontalHeaderLabels(["data_key", "duration", "mtime"])
        self._asset_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._asset_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._asset_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._asset_table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        header = self._asset_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._asset_table.itemSelectionChanged.connect(self._on_asset_selection_changed)
        layout.addWidget(self._asset_table)

        btns = QHBoxLayout()
        refresh_btn = QPushButton("Refresh")
        self._rename_btn = QPushButton("Rename")
        self._delete_btn = QPushButton("Delete")
        btns.addWidget(refresh_btn)
        btns.addWidget(self._rename_btn)
        btns.addWidget(self._delete_btn)
        layout.addLayout(btns)

        refresh_btn.clicked.connect(self.refresh)
        self._rename_btn.clicked.connect(self._rename_selected)
        self._delete_btn.clicked.connect(self._delete_selected)
        return group

    def _build_recipe_group(self) -> QGroupBox:
        group = QGroupBox("Formula Recipe")
        layout = QVBoxLayout(group)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("f(t) data_key:"))
        self._data_key_edit = QLineEdit()
        key_row.addWidget(self._data_key_edit, stretch=1)
        self._normalize_check = QCheckBox("Normalize")
        self._normalize_check.setChecked(True)
        key_row.addWidget(self._normalize_check)
        layout.addLayout(key_row)

        self._segment_table = QTableWidget(0, 2)
        self._segment_table.setHorizontalHeaderLabels(["duration (us)", "formula"])
        segment_header = self._segment_table.horizontalHeader()
        segment_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        segment_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._segment_table.cellChanged.connect(self._on_structure_changed)
        layout.addWidget(self._segment_table)

        segment_btns = QHBoxLayout()
        self._insert_before_btn = QPushButton("Insert Before")
        self._insert_after_btn = QPushButton("Insert After")
        self._delete_segment_btn = QPushButton("Delete Segment")
        self._save_btn = QPushButton("Save")
        segment_btns.addWidget(self._insert_before_btn)
        segment_btns.addWidget(self._insert_after_btn)
        segment_btns.addWidget(self._delete_segment_btn)
        segment_btns.addStretch()
        segment_btns.addWidget(self._save_btn)
        layout.addLayout(segment_btns)

        symbols = QLabel(
            "Supported: <b>t</b>=segment time (us), <b>T</b>=total time (us), "
            "<b>pi</b>, <b>e</b>, <b>I</b>=imaginary unit; functions: "
            "<b>sin</b>, <b>cos</b>, <b>tan</b>, <b>exp</b>, <b>sqrt</b>, "
            "<b>Abs</b>/<b>abs</b>, <b>erf</b>."
        )
        symbols.setTextFormat(Qt.TextFormat.RichText)
        symbols.setWordWrap(True)
        layout.addWidget(symbols)

        self._warning = QLabel()
        self._warning.setStyleSheet("color: red;")
        self._warning.setWordWrap(True)
        self._warning.setVisible(False)
        layout.addWidget(self._warning)

        # data_key edits only re-validate the key; structure/render state is unchanged.
        self._data_key_edit.textChanged.connect(self._on_data_key_changed)
        self._normalize_check.toggled.connect(self._on_structure_changed)
        self._insert_before_btn.clicked.connect(
            lambda: self._insert_segment(before=True)
        )
        self._insert_after_btn.clicked.connect(
            lambda: self._insert_segment(before=False)
        )
        self._delete_segment_btn.clicked.connect(self._delete_segment)
        self._save_btn.clicked.connect(self._save_recipe)
        return group

    # ------------------------------------------------------------------
    # Cheap validation primitives (no render_formula_recipe call)
    # ------------------------------------------------------------------

    def _validate_data_key(self) -> bool:
        data_key = self._data_key_edit.text().strip()
        if not _DATA_KEY_RE.fullmatch(data_key):
            self._data_key_error = "data_key must match ^[A-Za-z][A-Za-z0-9_]*$"
            return False
        self._data_key_error = None
        return True

    def _validate_structure(self) -> FormulaRecipe | None:
        """Construct FormulaRecipe from UI state (cheap: no render_formula_recipe).

        Sets _valid_recipe as a side effect so the cheap path always holds a
        fresh recipe object for save to use without triggering another render.
        """
        try:
            recipe = self._recipe_from_ui()
        except (ArbWaveformError, ValueError) as exc:
            self._structure_error = str(exc)
            self._valid_recipe = None
            return None
        self._structure_error = None
        self._valid_recipe = recipe
        return recipe

    def _apply_feedback(self) -> None:
        """Push consolidated error/warning to the UI; gate the Save button.

        Priority: data_key > structure > render (single warning label).
        Save is enabled only when all three error slots are None.
        """
        message = (
            self._data_key_error or self._structure_error or self._render_error or ""
        )
        self._set_warning(message)
        self._save_btn.setEnabled(
            self._data_key_error is None
            and self._structure_error is None
            and self._render_error is None
        )

    def _revalidate_cheap(self) -> None:
        """Re-run cheap validation only; does not touch the timer or render."""
        self._validate_data_key()
        self._validate_structure()
        self._apply_feedback()

    # ------------------------------------------------------------------
    # Edit event slots
    # ------------------------------------------------------------------

    def _on_structure_changed(self, *_: object) -> None:
        """Slot for cellChanged / normalize toggled.

        Runs cheap validation immediately; clears old render verdict (pending);
        restarts debounce timer for the expensive render path if structure is valid.
        """
        if self._suppress_segment_change:
            return
        recipe = self._validate_structure()
        # Structure changed → old render result is stale; mark pending.
        self._render_error = None
        self._apply_feedback()
        if recipe is None:
            # Broken structure: stop timer and clear deep-path cache.
            self._preview_timer.stop()
            self._valid_data = None
            return
        self._preview_timer.start(_PREVIEW_DEBOUNCE_MS)

    def _on_data_key_changed(self, *_: object) -> None:
        """Slot for data_key textChanged.

        Only re-validates the key; never touches the timer or renders.  data_key
        has no effect on the waveform shape, so triggering render here would be
        wasteful and confusing (preview title is not SSOT).
        """
        self._validate_data_key()
        self._apply_feedback()

    # ------------------------------------------------------------------
    # Debounced render (expensive path — only called by QTimer.timeout)
    # ------------------------------------------------------------------

    def _render_preview(self) -> None:
        """Expensive render triggered by debounce timer.

        Re-reads UI state (avoids relying on a snapshot taken at edit time).
        On failure: clears _valid_data, sets _render_error, disables Save.
        On success: sets _valid_data, clears _render_error, enables Save, plots preview.
        Invariant after settle: warning visible ⟺ Save disabled.
        """
        recipe = self._validate_structure()  # re-read; also updates _valid_recipe
        if recipe is None:
            self._valid_data = None
            self._render_error = None
            self._apply_feedback()
            return
        try:
            data = render_formula_recipe(recipe)
        except (ArbWaveformError, ValueError) as exc:
            # Render failure: no half-baked state — clear _valid_data.
            self._valid_data = None
            self._render_error = str(exc)
            self._apply_feedback()
            return
        self._valid_data = data
        self._render_error = None
        self._apply_feedback()
        self._preview.plot(
            self._data_key_edit.text().strip(),
            data,
            normalize=self._normalize_check.isChecked(),
        )

    # ------------------------------------------------------------------
    # Asset selection
    # ------------------------------------------------------------------

    def _selected_data_key(self) -> str | None:
        row = self._asset_table.currentRow()
        if row < 0:
            return None
        item = self._asset_table.item(row, 0)
        return item.text() if item is not None else None

    def _select_data_key(self, data_key: str) -> None:
        for row in range(self._asset_table.rowCount()):
            item = self._asset_table.item(row, 0)
            if item is not None and item.text() == data_key:
                self._asset_table.selectRow(row)
                return

    def _on_asset_selection_changed(self) -> None:
        data_key = self._selected_data_key()
        if data_key is None:
            return
        self._current_data_key = data_key
        self._data_key_edit.setText(data_key)
        try:
            data = self._ctrl.load_arb_waveform_data(data_key)
        except ArbWaveformError as exc:
            self._preview.clear(str(exc))
            self._set_warning(str(exc))
            return
        if data.recipe is None:
            # Asset has no formula recipe; seed the editor with a placeholder.
            self._set_segments(
                FormulaRecipe.from_raw(
                    {
                        "segments": [
                            {"duration": max(data.duration, 1e-6), "formula": "0"}
                        ],
                        "normalize": "peak",
                    }
                )
            )
        else:
            self._set_segments(data.recipe)
        self._revalidate_cheap()
        # Use already-loaded data directly; no debounce render needed.
        self._preview.plot(data_key, data, normalize=self._normalize_check.isChecked())
        self._valid_data = data
        self._render_error = None
        if data.recipe is None:
            # Override apply_feedback's message with the no-recipe notice.
            self._set_warning(
                "Selected asset has no formula recipe; saving will overwrite its data."
            )

    # ------------------------------------------------------------------
    # Draft management
    # ------------------------------------------------------------------

    def _reset_draft(self) -> None:
        self._current_data_key = None
        self._data_key_edit.setText(self._next_data_key())
        self._set_segments(_DEFAULT_RECIPE)
        self._preview.clear("Preview updates after a valid recipe")
        self._render_error = None
        self._on_structure_changed()

    def _set_segments(self, recipe: FormulaRecipe) -> None:
        """Populate segment table from a typed FormulaRecipe; suppresses cellChanged."""
        self._suppress_segment_change = True
        try:
            self._normalize_check.setChecked(recipe.normalize == "peak")
            self._segment_table.setRowCount(0)
            for segment in recipe.segments:
                # from_raw guarantees ≥1 segment; no empty-table fallback needed.
                self._append_segment(duration=segment.duration, formula=segment.formula)
        finally:
            self._suppress_segment_change = False

    def _append_segment(self, *, duration: float, formula: str) -> None:
        row = self._segment_table.rowCount()
        self._segment_table.insertRow(row)
        self._segment_table.setItem(row, 0, QTableWidgetItem(str(duration)))
        self._segment_table.setItem(row, 1, QTableWidgetItem(str(formula)))

    def _insert_segment(self, *, before: bool) -> None:
        row = self._segment_table.currentRow()
        if row < 0:
            row = self._segment_table.rowCount()
        elif not before:
            row += 1
        # Suppress cellChanged from setItem; fire a single _on_structure_changed after.
        self._suppress_segment_change = True
        try:
            self._segment_table.insertRow(row)
            self._segment_table.setItem(row, 0, QTableWidgetItem("1.0"))
            self._segment_table.setItem(row, 1, QTableWidgetItem("0"))
        finally:
            self._suppress_segment_change = False
        self._segment_table.selectRow(row)
        self._on_structure_changed()

    def _delete_segment(self) -> None:
        if self._segment_table.rowCount() <= 1:
            return
        row = self._segment_table.currentRow()
        if row < 0:
            row = self._segment_table.rowCount() - 1
        self._segment_table.removeRow(row)
        self._on_structure_changed()

    def _recipe_from_ui(self) -> FormulaRecipe:
        """Read the segment table and normalize checkbox into a typed FormulaRecipe.

        Passes duration as a string so _coerce_positive_float handles both numeric
        and textual input in a single validation point inside from_raw.
        """
        raw_segments: list[dict[str, object]] = []
        for row in range(self._segment_table.rowCount()):
            duration_item = self._segment_table.item(row, 0)
            formula_item = self._segment_table.item(row, 1)
            duration_text = duration_item.text().strip() if duration_item else ""
            formula = formula_item.text().strip() if formula_item else ""
            raw_segments.append({"duration": duration_text, "formula": formula})
        normalize = "peak" if self._normalize_check.isChecked() else "none"
        return FormulaRecipe.from_raw(
            {"segments": raw_segments, "normalize": normalize}
        )

    # ------------------------------------------------------------------
    # Save path (override design: render done once inside service)
    # ------------------------------------------------------------------

    def _save_recipe(self) -> None:
        data_key = self._data_key_edit.text().strip()
        if not _DATA_KEY_RE.fullmatch(data_key):
            return  # Save gate should have blocked; defensive return.
        recipe = self._valid_recipe  # cheap path keeps this fresh
        if recipe is None:
            return  # Save gate should have blocked; defensive return.
        exists = data_key in self._known_data_keys()
        if exists and data_key != self._current_data_key:
            answer = QMessageBox.question(
                self,
                "Overwrite waveform",
                f"Overwrite existing arbitrary waveform {data_key!r}?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        # Save try: service renders the recipe once and persists it.
        try:
            self._ctrl.set_arb_waveform(data_key, recipe, overwrite=exists)
        except Exception as exc:  # noqa: BLE001 — user-facing dialog boundary
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self._current_data_key = data_key
        # Reload try (separate): reload failure is distinct from save failure.
        try:
            data = self._ctrl.load_arb_waveform_data(data_key)
        except Exception as exc:  # noqa: BLE001 — user-facing dialog boundary
            QMessageBox.critical(self, "Reload failed", str(exc))
            self.refresh()
            self._select_data_key(data_key)
            return
        self._preview.plot(data_key, data, normalize=self._normalize_check.isChecked())
        self.refresh()
        self._select_data_key(data_key)

    # ------------------------------------------------------------------
    # Rename / delete
    # ------------------------------------------------------------------

    def _rename_selected(self) -> None:
        old_key = self._selected_data_key()
        if old_key is None:
            return
        new_key, accepted = QInputDialog.getText(
            self,
            "Rename waveform",
            "New data_key:",
            text=old_key,
        )
        if not accepted:
            return
        new_key = new_key.strip()
        if new_key == old_key:
            return
        if not _DATA_KEY_RE.fullmatch(new_key):
            QMessageBox.warning(
                self,
                "Invalid data_key",
                "data_key must match ^[A-Za-z][A-Za-z0-9_]*$",
            )
            return
        if new_key in self._known_data_keys():
            QMessageBox.warning(
                self,
                "Rename failed",
                f"Arbitrary waveform {new_key!r} already exists.",
            )
            return
        try:
            self._ctrl.rename_arb_waveform(old_key, new_key)
        except Exception as exc:  # noqa: BLE001 — user-facing dialog boundary
            QMessageBox.critical(self, "Rename failed", str(exc))
            return
        self._current_data_key = new_key
        self._data_key_edit.setText(new_key)
        self.refresh()
        self._select_data_key(new_key)

    def _delete_selected(self) -> None:
        data_key = self._selected_data_key()
        if data_key is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete waveform",
            f"Delete arbitrary waveform {data_key!r}?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._ctrl.delete_arb_waveform(data_key)
        except Exception as exc:  # noqa: BLE001 — user-facing dialog boundary
            QMessageBox.critical(self, "Delete failed", str(exc))
            return
        self._current_data_key = None
        self.refresh()
        self._reset_draft()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_warning(self, message: str) -> None:
        self._warning.setText(message)
        self._warning.setVisible(bool(message))

    def _next_data_key(self) -> str:
        existing = self._known_data_keys()
        index = 1
        while f"arb_data{index}" in existing:
            index += 1
        return f"arb_data{index}"

    def _known_data_keys(self) -> set[str]:
        return set(self._ctrl.list_arb_waveforms())

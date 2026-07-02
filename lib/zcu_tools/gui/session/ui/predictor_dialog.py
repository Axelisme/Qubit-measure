"""PredictorDialog — load/define a FluxoniumPredictor and predict frequencies.

Two ways to supply model parameters:
  1. Browse a params.json → populates the six editable spinboxes AND auto-applies
     (installs the predictor); on install failure the fields are still populated and
     the error is shown in the status.
  2. Type or tweak the spinboxes directly, then Apply → builds+installs in-memory
     via set_predictor_model_params.

Layout: left control column (model params group / predictor-active status /
tracked-transitions group / Remove+Close row) + right QTabWidget with three
PredictorCurveCanvas tabs:
  - "Frequency"  : f_ij transition-frequency curves (MHz)
  - "|n|"        : |<i|n|j>|   matrix-element curves (dimensionless)
  - "|phi|"      : |<i|phi|j>| matrix-element curves (dimensionless)

A click-follow-click flux-position marker is bidirectionally coupled to the
"Device value" spinbox.  Per-transition f / |n| / |phi| values are shown in the
table and updated (debounced) while the marker follows the cursor.  Both matrix
operators (n and phi) are always computed — there is no operator selector.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
from qtpy.QtGui import QCloseEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.session.ui.predictor_canvas import PredictorCurveCanvas
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox

if TYPE_CHECKING:
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

# Default display window in flux (Φ/Φ₀) units (plan spec).
_DEFAULT_FLUX_WINDOW: tuple[float, float] = (0.4, 1.1)
# Number of grid points for the curve computation.
_CURVE_GRID_N = 200
# Debounce delay (ms) before updating the per-transition frequency column.
_DEBOUNCE_MS = 150
# Default tracked transitions: from∈{0,1}, from<to<4.
_DEFAULT_TRANSITIONS: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
]

# Default spinbox values when no predictor is installed.
_DEFAULT_EJ: float = 4.0
_DEFAULT_EC: float = 1.0
_DEFAULT_EL: float = 1.0
_DEFAULT_FLUX_HALF: float = 0.0
_DEFAULT_FLUX_PERIOD: float = 0.005
_DEFAULT_FLUX_BIAS: float = 0.0

# Column indices for the 4-column tracked-transitions table.
# Both matrix operators (n and phi) are shown side by side — there is no selector.
_COL_TRANSITION = 0
_COL_FREQ = 1
_COL_MAG_N = 2
_COL_MAG_PHI = 3


class PredictorDialog(QDialog):
    """Modal dialog for loading a FluxoniumPredictor and predicting frequencies."""

    def __init__(
        self,
        predictor: PredictorControlPort,
        parent: QWidget | None = None,
        *,
        persistent_on_close: bool = False,
    ) -> None:
        super().__init__(parent)
        self._pred = predictor
        self._persistent_on_close = persistent_on_close
        self.setWindowTitle("Predictor")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(480)

        # Mutable list of currently tracked (from, to) transition pairs.
        # Modified only by _add_transition / _delete_transition / _on_remove_selected.
        self._tracked: list[tuple[int, int]] = list(_DEFAULT_TRANSITIONS)

        # Last computed results; None when predictor is not loaded or
        # tracked list is empty. Both matrix operators are always tracked.
        self._last_freq_result = None
        self._last_mat_n_result = None
        self._last_mat_phi_result = None

        # Root layout: left controls | right tab widget.
        root_layout = QHBoxLayout(self)

        # ── Left: controls column ─────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # ── Model params (editable EJ/EC/EL/flux anchors/flux_bias) ───────
        # Browse opens a file dialog and populates the six spinboxes (no install).
        # Apply builds+installs via set_predictor_model_params.
        model_group = QGroupBox("Model params")
        model_form = QFormLayout(model_group)

        # Energies in GHz; flux anchors and bias in device-value units. Wide
        # ranges + many decimals so exploratory inputs are accepted; the only hard
        # guard (flux_period != 0) lives in _on_apply_model_params.
        self._ej_spin = self._make_param_spin(_DEFAULT_EJ)
        model_form.addRow("EJ (GHz):", self._ej_spin)
        self._ec_spin = self._make_param_spin(_DEFAULT_EC)
        model_form.addRow("EC (GHz):", self._ec_spin)
        self._el_spin = self._make_param_spin(_DEFAULT_EL)
        model_form.addRow("EL (GHz):", self._el_spin)
        self._flux_half_spin = self._make_param_spin(_DEFAULT_FLUX_HALF)
        model_form.addRow("flux_half:", self._flux_half_spin)
        self._flux_period_spin = self._make_param_spin(_DEFAULT_FLUX_PERIOD)
        model_form.addRow("flux_period:", self._flux_period_spin)
        # flux_bias lives here (was in the old "Load predictor" group).
        self._flux_bias_spin = self._make_param_spin(_DEFAULT_FLUX_BIAS)
        model_form.addRow("flux_bias:", self._flux_bias_spin)

        model_btn_row = QHBoxLayout()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._on_browse_file)
        model_btn_row.addWidget(browse_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply_model_params)
        model_btn_row.addWidget(apply_btn)
        model_form.addRow("", model_btn_row)

        left_layout.addWidget(model_group)

        # ── Predictor active/inactive indicator ───────────────────────────
        # Simple presence indicator (active vs not installed), updated on
        # install (Apply/Browse), on clear, and on predictor-changed events.
        self._active_label = QLabel("")
        left_layout.addWidget(self._active_label)

        # ── Tracked transitions group (predict controls + table + remove) ──
        transitions_group = QGroupBox("Tracked transitions")
        transitions_layout = QVBoxLayout(transitions_group)

        # Controls row above the table: device-value spinbox + add-transition
        # controls.  All in a single horizontal bar so the table can take the
        # remaining vertical space.  No operator selector — both matrix operators
        # are shown as table columns.
        controls_row = QHBoxLayout()

        controls_row.addWidget(QLabel("Device value:"))
        self._predict_value_spin = TrimDoubleSpinBox()
        self._predict_value_spin.setRange(-1e6, 1e6)
        self._predict_value_spin.setDecimals(6)
        self._predict_value_spin.setValue(0.0)
        self._predict_value_spin.valueChanged.connect(self._on_spinbox_changed)
        controls_row.addWidget(self._predict_value_spin)

        controls_row.addSpacing(12)
        controls_row.addWidget(QLabel("from"))
        self._add_from_spin = QSpinBox()
        self._add_from_spin.setRange(0, 20)
        self._add_from_spin.setValue(0)
        controls_row.addWidget(self._add_from_spin)
        controls_row.addWidget(QLabel("to"))
        self._add_to_spin = QSpinBox()
        self._add_to_spin.setRange(0, 20)
        self._add_to_spin.setValue(1)
        controls_row.addWidget(self._add_to_spin)
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self._on_add_clicked)
        controls_row.addWidget(add_btn)

        controls_row.addStretch()
        transitions_layout.addLayout(controls_row)

        # 4-column table: Transition | f (MHz) | |n| | |phi|
        # Multi-row selection enabled; individual delete buttons are gone.
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Transition", "f (MHz)", "|n|", "|phi|"])
        header = self._table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectRows  # type: ignore[attr-defined]
        )
        self._table.setSelectionMode(
            QAbstractItemView.ExtendedSelection  # type: ignore[attr-defined]
        )
        self._table.setEditTriggers(
            QTableWidget.NoEditTriggers  # type: ignore[attr-defined]
        )
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        transitions_layout.addWidget(self._table)

        left_layout.addWidget(transitions_group)

        # ── status label ──────────────────────────────────────────────────
        self._status_label = QLabel("")
        left_layout.addWidget(self._status_label)

        # ── Remove + Close on one horizontal row (saves vertical space) ────
        # Remove removes ALL currently selected transition rows; Close closes
        # the dialog. A spacer pushes Close to the right edge.
        action_row = QHBoxLayout()
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_selected)
        action_row.addWidget(self._remove_btn)
        action_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._on_close_requested)
        action_row.addWidget(close_btn)
        left_layout.addLayout(action_row)

        left_layout.addStretch()

        # ── Right: QTabWidget with three canvases ─────────────────────────
        # Frequency, |n|, |phi|. All three share one marker value and one set of
        # follow/lock callbacks, so a click-follow on any tab drives the others.
        self._tab_widget = QTabWidget()

        self._freq_canvas = PredictorCurveCanvas(figsize=(6.0, 4.0))
        self._mat_n_canvas = PredictorCurveCanvas(figsize=(6.0, 4.0))
        self._mat_phi_canvas = PredictorCurveCanvas(figsize=(6.0, 4.0))

        # All canvases whose marker stays in sync with the device-value spinbox.
        self._all_canvases = (
            self._freq_canvas,
            self._mat_n_canvas,
            self._mat_phi_canvas,
        )
        for canvas in self._all_canvases:
            canvas.bind_callbacks(
                on_follow=self._on_canvas_follow,
                on_lock=self._on_canvas_lock,
            )

        self._tab_widget.addTab(self._freq_canvas, "Frequency")
        self._tab_widget.addTab(self._mat_n_canvas, "|n|")
        self._tab_widget.addTab(self._mat_phi_canvas, "|phi|")

        root_layout.addWidget(left_widget, stretch=0)
        root_layout.addWidget(self._tab_widget, stretch=1)

        # ── Debounce timer for per-row column updates ─────────────────────
        # Used both by the spinbox and by the marker follow-loop so cursor motion
        # updates the spinbox/table/markers at the debounce cadence, not per pixel.
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._update_value_columns)

        # ── Predictor info cache (flux_half, flux_period for conversions) ──
        self._flux_half: float | None = None
        self._flux_period: float | None = None

        # Populate table from default transitions before loading predictor.
        self._rebuild_table()

        # Pre-fill with current predictor state.
        info = predictor.get_predictor_info()
        if info is not None:
            self._flux_bias_spin.setValue(info["flux_bias"])
            self._flux_half = info["flux_half"]
            self._flux_period = info["flux_period"]
            self._populate_model_fields(
                info["EJ"],
                info["EC"],
                info["EL"],
                info["flux_half"],
                info["flux_period"],
            )
            self._set_status("Currently loaded", error=False)
            self._refresh_curves()
        self._update_active_label()

        # Facet subscription for live predictor state updates.
        self._unsubscribe_predictor_changed: Callable[[], None] | None = (
            predictor.on_predictor_changed(self._on_predictor_changed)
        )
        self.finished.connect(self._cleanup_subscription)
        self.destroyed.connect(self._cleanup_subscription)

    def reject(self) -> None:
        if self._persistent_on_close:
            self.hide()
            return
        super().reject()

    def closeEvent(self, a0: QCloseEvent | None) -> None:  # noqa: N802
        if self._persistent_on_close:
            if a0 is not None:
                a0.ignore()
            self.hide()
            return
        super().closeEvent(a0)

    def _on_close_requested(self) -> None:
        self.reject()

    # ------------------------------------------------------------------
    # Model-param widgets / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_param_spin(default: float = 0.0) -> TrimDoubleSpinBox:
        """A wide-range, high-precision spinbox for an editable model param.

        Range is intentionally permissive so exploratory inputs (any positive
        energy ratio, arbitrary flux anchors) are accepted; the only hard guard
        (flux_period != 0) lives in _on_apply_model_params, not here.
        """
        spin = TrimDoubleSpinBox()
        spin.setRange(-1e6, 1e6)
        spin.setDecimals(6)
        spin.setValue(default)
        return spin

    def _populate_model_fields(
        self, ej: float, ec: float, el: float, flux_half: float, flux_period: float
    ) -> None:
        """Set the five energy/flux-anchor spinboxes (no install side effect)."""
        self._ej_spin.setValue(ej)
        self._ec_spin.setValue(ec)
        self._el_spin.setValue(el)
        self._flux_half_spin.setValue(flux_half)
        self._flux_period_spin.setValue(flux_period)

    # ------------------------------------------------------------------
    # Affine helpers (value ↔ flux)
    # ------------------------------------------------------------------

    def _make_affine(
        self,
    ) -> tuple[Callable[[float], float] | None, Callable[[float], float] | None]:
        """Return (value_to_flux, flux_to_value) closures or (None, None) if no predictor.

        Mirrors FluxoniumPredictor.value_to_flux exactly:
          flux = (value + flux_bias - flux_half) / flux_period + 0.5
          value = (flux - 0.5) * flux_period + flux_half - flux_bias
        """
        if self._flux_half is None or self._flux_period is None:
            return None, None
        info = self._pred.get_predictor_info()
        if info is None:
            return None, None
        flux_bias = info["flux_bias"]
        flux_half = self._flux_half
        flux_period = self._flux_period

        def value_to_flux(v: float) -> float:
            return (v + flux_bias - flux_half) / flux_period + 0.5

        def flux_to_value(f: float) -> float:
            return (f - 0.5) * flux_period + flux_half - flux_bias

        return value_to_flux, flux_to_value

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _rebuild_table(self) -> None:
        """Rebuild the QTableWidget rows from self._tracked.

        Called after any add/delete operation.  Preserves the previous row
        selection if the transition is still present.
        """
        # Remember the currently selected transitions so we can restore them.
        selected = self._selected_transitions()

        # Disconnect selection signal while rebuilding to avoid spurious events.
        self._table.itemSelectionChanged.disconnect(self._on_selection_changed)
        self._table.setRowCount(0)

        for row_idx, (frm, to) in enumerate(self._tracked):
            self._table.insertRow(row_idx)

            # Column 0: transition label (e.g. "0→1")
            label_item = QTableWidgetItem(f"{frm}→{to}")
            self._table.setItem(row_idx, _COL_TRANSITION, label_item)

            # Column 1: freq placeholder — populated by _update_value_columns
            freq_item = QTableWidgetItem("—")
            self._table.setItem(row_idx, _COL_FREQ, freq_item)

            # Column 2: |n| placeholder — populated by _update_value_columns
            mag_n_item = QTableWidgetItem("—")
            self._table.setItem(row_idx, _COL_MAG_N, mag_n_item)

            # Column 3: |phi| placeholder — populated by _update_value_columns
            mag_phi_item = QTableWidgetItem("—")
            self._table.setItem(row_idx, _COL_MAG_PHI, mag_phi_item)

        self._table.resizeColumnsToContents()
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        # Restore selection for transitions still present.
        for transition in selected:
            if transition in self._tracked:
                row = self._tracked.index(transition)
                self._table.selectRow(row)

    def _selected_transitions(self) -> list[tuple[int, int]]:
        """Return the (from, to) tuples for all currently selected table rows."""
        selected_rows = {idx.row() for idx in self._table.selectedIndexes()}
        return [
            self._tracked[row]
            for row in sorted(selected_rows)
            if 0 <= row < len(self._tracked)
        ]

    def _selected_transition(self) -> tuple[int, int] | None:
        """Return the (from, to) tuple for the first selected row, or None.

        Used for single-transition highlight on the canvas.
        """
        transitions = self._selected_transitions()
        return transitions[0] if transitions else None

    def _update_value_columns(self) -> None:
        """Fill f / |n| / |phi| columns for every tracked transition at the marker.

        Called by the debounce timer.  Uses point-predict for each row so each value
        is as accurate as possible.  Any failure per-row shows "—".
        """
        from zcu_tools.gui.session.services.predictor import (
            PredictFreqRequest,
            PredictorNotLoaded,
        )

        marker_value = self._predict_value_spin.value()
        for row_idx, transition in enumerate(self._tracked):
            # Freq column: per-row scalar predict_freq (most accurate)
            try:
                freq = self._pred.predict_freq(
                    PredictFreqRequest(value=marker_value, transition=transition)
                )
                freq_text = f"{freq:.4f}"
            except (PredictorNotLoaded, ValueError):
                freq_text = "—"
            freq_item = self._table.item(row_idx, _COL_FREQ)
            if freq_item is not None:
                freq_item.setText(freq_text)

            # |n| and |phi| columns: one single-point matrix curve call per
            # operator (avoids the level<=1 cap in predict_matrix_element).
            n_item = self._table.item(row_idx, _COL_MAG_N)
            if n_item is not None:
                n_item.setText(self._point_matrix_text(marker_value, transition, "n"))
            phi_item = self._table.item(row_idx, _COL_MAG_PHI)
            if phi_item is not None:
                phi_item.setText(
                    self._point_matrix_text(marker_value, transition, "phi")
                )

    def _point_matrix_text(
        self, value: float, transition: tuple[int, int], operator: str
    ) -> str:
        """|<i|operator|j>| at a single device value, formatted, or "—" on failure."""
        from zcu_tools.gui.session.services.predictor import (
            PredictMatrixCurveRequest,
            PredictorNotLoaded,
        )

        try:
            mat_result = self._pred.predict_matrix_element_curve(
                PredictMatrixCurveRequest(
                    values=np.array([value], dtype=np.float64),
                    transitions=(transition,),
                    operator=operator,  # type: ignore[arg-type]
                )
            )
            return f"{mat_result.mags[0, 0]:.5f}"
        except (PredictorNotLoaded, ValueError):
            return "—"

    # ------------------------------------------------------------------
    # Curve refresh
    # ------------------------------------------------------------------

    def _refresh_curves(self) -> None:
        """Recompute and redraw the freq curve and both matrix-element curves."""
        from zcu_tools.gui.session.services.predictor import (
            PredictCurveRequest,
            PredictorNotLoaded,
        )

        if not self._tracked:
            for canvas in self._all_canvases:
                canvas.clear()
            return

        value_to_flux, flux_to_value = self._make_affine()
        if value_to_flux is None or flux_to_value is None:
            return

        # Build value grid from the default flux window.
        v_lo = flux_to_value(_DEFAULT_FLUX_WINDOW[0])
        v_hi = flux_to_value(_DEFAULT_FLUX_WINDOW[1])
        if v_lo > v_hi:
            v_lo, v_hi = v_hi, v_lo
        grid = np.linspace(v_lo, v_hi, _CURVE_GRID_N, dtype=np.float64)

        highlight = self._selected_transition()
        marker_value = self._predict_value_spin.value()

        # ── Frequency curves ──────────────────────────────────────────────
        try:
            freq_result = self._pred.predict_freq_curve(
                PredictCurveRequest(values=grid, transitions=tuple(self._tracked))
            )
            self._last_freq_result = freq_result
            self._freq_canvas.render_curves(
                values=freq_result.values,
                labels=freq_result.labels,
                series=freq_result.freqs_mhz,
                ylabel="Frequency (MHz)",
                highlight=highlight,
                marker_value=marker_value,
                flux_window=_DEFAULT_FLUX_WINDOW,
                value_to_flux=value_to_flux,
                flux_to_value=flux_to_value,
            )
        except (PredictorNotLoaded, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return

        # ── Matrix element curves (both operators, one tab each) ───────────
        if not self._render_matrix_curve(
            "n",
            self._mat_n_canvas,
            grid,
            highlight,
            marker_value,
            flux_to_value,
            value_to_flux,
        ):
            return
        if not self._render_matrix_curve(
            "phi",
            self._mat_phi_canvas,
            grid,
            highlight,
            marker_value,
            flux_to_value,
            value_to_flux,
        ):
            return

        # Refresh all value columns for the new marker position.
        self._update_value_columns()

    def _render_matrix_curve(
        self,
        operator: str,
        canvas: PredictorCurveCanvas,
        grid: np.ndarray,
        highlight: tuple[int, int] | None,
        marker_value: float,
        flux_to_value: Callable[[float], float],
        value_to_flux: Callable[[float], float],
    ) -> bool:
        """Compute+render one operator's matrix-element curve onto ``canvas``.

        Returns True on success; on an expected failure it sets the status and
        returns False so the caller can stop the refresh.
        """
        from zcu_tools.gui.session.services.predictor import (
            PredictMatrixCurveRequest,
            PredictorNotLoaded,
        )

        try:
            mat_result = self._pred.predict_matrix_element_curve(
                PredictMatrixCurveRequest(
                    values=grid,
                    transitions=tuple(self._tracked),
                    operator=operator,  # type: ignore[arg-type]
                )
            )
        except (PredictorNotLoaded, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return False

        if operator == "n":
            self._last_mat_n_result = mat_result
        else:
            self._last_mat_phi_result = mat_result
        canvas.render_curves(
            values=mat_result.values,
            labels=mat_result.labels,
            series=mat_result.mags,
            ylabel=f"|<i|{operator}|j>|",
            highlight=highlight,
            marker_value=marker_value,
            flux_window=_DEFAULT_FLUX_WINDOW,
            value_to_flux=value_to_flux,
            flux_to_value=flux_to_value,
        )
        return True

    # ------------------------------------------------------------------
    # Add / delete transitions
    # ------------------------------------------------------------------

    def _add_transition(self, frm: int, to: int) -> bool:
        """Validate and append (frm, to) to self._tracked.

        Returns True on success, False if the transition is invalid or duplicate.
        Validation: frm >= 0, to >= 0, frm < to, and not already tracked.
        """
        if frm < 0 or to < 0:
            self._set_status("Transition levels must be non-negative.", error=True)
            return False
        if frm >= to:
            self._set_status(
                f"Invalid transition {frm}→{to}: 'from' must be < 'to'.", error=True
            )
            return False
        if (frm, to) in self._tracked:
            self._set_status(f"Transition {frm}→{to} is already tracked.", error=True)
            return False
        self._tracked.append((frm, to))
        return True

    def _delete_transition(self, transition: tuple[int, int]) -> None:
        """Remove ``transition`` from self._tracked and refresh."""
        if transition not in self._tracked:
            return
        self._tracked.remove(transition)
        self._rebuild_table()
        self._refresh_curves()

    def _on_add_clicked(self) -> None:
        """Validate and add the transition from the add-row spinboxes."""
        frm = self._add_from_spin.value()
        to = self._add_to_spin.value()
        if self._add_transition(frm, to):
            self._rebuild_table()
            self._refresh_curves()

    def _on_remove_selected(self) -> None:
        """Remove ALL currently selected transitions and refresh."""
        to_remove = self._selected_transitions()
        if not to_remove:
            return
        for transition in to_remove:
            if transition in self._tracked:
                self._tracked.remove(transition)
        self._rebuild_table()
        self._refresh_curves()

    # ------------------------------------------------------------------
    # Spinbox / canvas bidirectional coupling
    # ------------------------------------------------------------------

    def _sync_markers(self, value: float) -> None:
        """Move the marker on every canvas to ``value`` (no recompute)."""
        for canvas in self._all_canvases:
            canvas.set_marker(value)

    def _on_spinbox_changed(self, value: float) -> None:
        """Spinbox changed → move all canvas markers; schedule debounced column update."""
        self._sync_markers(value)
        self._debounce_timer.start()

    def _on_canvas_follow(self, value: float) -> None:
        """Marker following the cursor → update spinbox + all markers; debounce recompute.

        Fired on every motion event while the canvas is in follow mode. The
        spinbox value is updated immediately (with signals blocked to avoid a
        re-entrant column update), the markers track the cursor, but the heavy
        recompute (table f/|n|/|phi|) is debounced so motion does not recompute
        on every pixel.
        """
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        self._sync_markers(value)
        self._debounce_timer.start()

    def _on_canvas_lock(self, value: float) -> None:
        """Marker locked (second click) → final immediate recompute of the columns."""
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        self._sync_markers(value)
        # Lock triggers an immediate update (cancel any pending debounce first).
        self._debounce_timer.stop()
        self._update_value_columns()

    def _on_selection_changed(self) -> None:
        """Table row selection changed → restyle highlighted curve on all canvases."""
        transition = self._selected_transition()
        for canvas in self._all_canvases:
            canvas.set_highlight(transition)

    # ------------------------------------------------------------------
    # Browse / Apply handlers
    # ------------------------------------------------------------------

    def _on_browse_file(self) -> None:
        """Open a file dialog to pick a params.json, populate the fields, auto-apply.

        Browse first fills the six spinboxes from the file, then installs the
        predictor via the same path as Apply.  If the install fails (e.g.
        flux_period == 0 or a service error), the fields stay populated and the
        error is shown in the status — Browse never crashes on a bad install.
        """
        from zcu_tools.gui.session.services.predictor import (
            PredictorLoadError,
            read_fluxdep_fit_params,
        )

        path, _ = QFileDialog.getOpenFileName(
            self, "Select params.json", "", "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            params = read_fluxdep_fit_params(path)
        except PredictorLoadError as exc:
            self._set_status(str(exc), error=True)
            return
        self._populate_model_fields(
            params.EJ, params.EC, params.EL, params.flux_half, params.flux_period
        )
        logger.info("PredictorDialog: loaded params into fields from %r", path)
        # Auto-apply: install immediately. Failure leaves fields populated and
        # surfaces the error in the status (install path sets its own status).
        self._install_from_fields()

    def _on_apply_model_params(self) -> None:
        """Build+install a predictor from the current editable fields."""
        self._install_from_fields()

    def _install_from_fields(self) -> bool:
        """Build+install a predictor from the current editable fields.

        Shared install path for both Apply and Browse-auto-apply.  Only
        flux_period == 0 is guarded (it makes the value<->flux affine singular);
        everything else is allowed so trial ratios work.  The bus
        PredictorChangedPayload that install emits drives the state refresh.
        Returns True on success, False on a guarded/expected failure (status set).
        """
        from zcu_tools.gui.session.services.predictor import (
            PredictorLoadError,
            SetModelParamsRequest,
        )

        flux_period = self._flux_period_spin.value()
        if flux_period == 0.0:
            self._set_status(
                "flux_period must be non-zero (value↔flux mapping).", error=True
            )
            return False
        req = SetModelParamsRequest(
            EJ=self._ej_spin.value(),
            EC=self._ec_spin.value(),
            EL=self._el_spin.value(),
            flux_half=self._flux_half_spin.value(),
            flux_period=flux_period,
            flux_bias=self._flux_bias_spin.value(),
        )
        try:
            self._pred.set_predictor_model_params(req)
        except PredictorLoadError as exc:
            self._set_status(str(exc), error=True)
            return False
        self._set_status("Model params applied", error=False)
        self._update_active_label()
        logger.info("PredictorDialog: applied model params %r", req)
        return True

    # ------------------------------------------------------------------
    # Bus subscription
    # ------------------------------------------------------------------

    def _cleanup_subscription(self, *_args: object) -> None:
        unsubscribe = self._unsubscribe_predictor_changed
        if unsubscribe is None:
            return
        self._unsubscribe_predictor_changed = None
        unsubscribe()

    def _on_predictor_changed(self, payload: object) -> None:
        del payload
        info = self._pred.get_predictor_info()
        if info is not None:
            self._flux_bias_spin.setValue(info["flux_bias"])
            self._flux_half = info["flux_half"]
            self._flux_period = info["flux_period"]
            self._populate_model_fields(
                info["EJ"],
                info["EC"],
                info["EL"],
                info["flux_half"],
                info["flux_period"],
            )
            self._set_status("Currently loaded", error=False)
            self._refresh_curves()
        else:
            self._flux_half = None
            self._flux_period = None
            self._set_status("Not loaded", error=False)
            for canvas in self._all_canvases:
                canvas.clear()
            for row_idx in range(self._table.rowCount()):
                for col in (_COL_FREQ, _COL_MAG_N, _COL_MAG_PHI):
                    item = self._table.item(row_idx, col)
                    if item is not None:
                        item.setText("—")
        self._update_active_label()

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")

    def _update_active_label(self) -> None:
        """Reflect whether a predictor is currently installed (active/inactive).

        Presence is derived from get_predictor_info (None ⇒ not installed). This
        is a plain indicator, NOT a per-param read-back.
        """
        active = self._pred.get_predictor_info() is not None
        if active:
            self._active_label.setText("Predictor: active")
            self._active_label.setStyleSheet("color: green;")
        else:
            self._active_label.setText("Predictor: not installed")
            self._active_label.setStyleSheet("color: gray;")

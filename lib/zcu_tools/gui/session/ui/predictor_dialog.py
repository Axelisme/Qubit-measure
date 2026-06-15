"""PredictorDialog — load/define a FluxoniumPredictor and predict frequencies.

Two ways to supply model parameters:
  1. Browse a params.json → populates the six editable spinboxes (no install).
  2. Type or tweak the spinboxes directly, then Apply → builds+installs in-memory
     via set_predictor_model_params.

Layout: left control column (model params group / tracked-transitions group /
status) + right QTabWidget with two PredictorCurveCanvas tabs:
  - "Frequency"       : f_ij transition-frequency curves (MHz)
  - "Matrix element"  : |<i|op|j>| curves (dimensionless)

A draggable flux-position marker is bidirectionally coupled to the "Flux value (A)"
spinbox.  Per-transition Freq and |M| values are shown in the table and updated on
marker movement (debounced).  An "operator" combobox (n / phi) selects which matrix
element operator is used; changing it re-computes the matrix-element curves and
refreshes the |M| column.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
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

if TYPE_CHECKING:
    from zcu_tools.gui.session.controller_port import SessionControllerPort

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

# Column indices for the 3-column tracked-transitions table.
_COL_TRANSITION = 0
_COL_FREQ = 1
_COL_MAG = 2


class PredictorDialog(QDialog):
    """Modal dialog for loading a FluxoniumPredictor and predicting frequencies.

    Shared session dialog: depends only on ``SessionControllerPort`` (the
    load/clear/predict + predictor-info + bus surface), so both measure and
    autofluxdep open it with their own Controller.
    """

    def __init__(
        self, controller: SessionControllerPort, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._ctrl = controller
        self.setWindowTitle("Predictor")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(480)

        # Mutable list of currently tracked (from, to) transition pairs.
        # Modified only by _add_transition / _delete_transition / _on_remove_selected.
        self._tracked: list[tuple[int, int]] = list(_DEFAULT_TRANSITIONS)

        # Currently selected matrix-element operator ("n" or "phi").
        self._operator: str = "n"

        # Last computed results; None when predictor is not loaded or
        # tracked list is empty.
        self._last_freq_result = None
        self._last_mat_result = None

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

        # ── Tracked transitions group (predict controls + table + remove) ──
        transitions_group = QGroupBox("Tracked transitions")
        transitions_layout = QVBoxLayout(transitions_group)

        # Controls row above the table: flux value spinbox + operator selector
        # + add-transition controls.  All in a single horizontal bar so the
        # table can take the remaining vertical space.
        controls_row = QHBoxLayout()

        controls_row.addWidget(QLabel("Flux value (A):"))
        self._predict_value_spin = QDoubleSpinBox()
        self._predict_value_spin.setRange(-1e6, 1e6)
        self._predict_value_spin.setDecimals(6)
        self._predict_value_spin.setValue(0.0)
        self._predict_value_spin.valueChanged.connect(self._on_spinbox_changed)
        controls_row.addWidget(self._predict_value_spin)

        controls_row.addSpacing(12)
        controls_row.addWidget(QLabel("operator:"))
        self._operator_combo = QComboBox()
        self._operator_combo.addItems(["n", "phi"])
        self._operator_combo.setCurrentText("n")
        self._operator_combo.currentTextChanged.connect(self._on_operator_changed)
        controls_row.addWidget(self._operator_combo)

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

        # 3-column table: Transition | f (MHz) | |M|
        # Multi-row selection enabled; individual delete buttons are gone.
        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Transition", "f (MHz)", "|M|"])
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

        # Remove button below the table — removes ALL currently selected rows.
        remove_row = QHBoxLayout()
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_selected)
        remove_row.addWidget(self._remove_btn)
        remove_row.addStretch()
        transitions_layout.addLayout(remove_row)

        left_layout.addWidget(transitions_group)

        # ── status label ──────────────────────────────────────────────────
        self._status_label = QLabel("")
        left_layout.addWidget(self._status_label)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)  # type: ignore[attr-defined]
        btn_box.rejected.connect(self.reject)
        left_layout.addWidget(btn_box)

        left_layout.addStretch()

        # ── Right: QTabWidget with two canvases ───────────────────────────
        self._tab_widget = QTabWidget()

        self._freq_canvas = PredictorCurveCanvas(figsize=(6.0, 4.0))
        self._freq_canvas.bind_callbacks(
            on_drag=self._on_canvas_drag,
            on_drop=self._on_canvas_drop,
        )

        self._mat_canvas = PredictorCurveCanvas(figsize=(6.0, 4.0))
        self._mat_canvas.bind_callbacks(
            on_drag=self._on_canvas_drag,
            on_drop=self._on_canvas_drop,
        )

        self._tab_widget.addTab(self._freq_canvas, "Frequency")
        self._tab_widget.addTab(self._mat_canvas, "Matrix element")

        root_layout.addWidget(left_widget, stretch=0)
        root_layout.addWidget(self._tab_widget, stretch=1)

        # ── Debounce timer for per-row column updates ─────────────────────
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
        info = controller.get_predictor_info()
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

        # EventBus subscription for live predictor state updates.
        from zcu_tools.gui.session.events import PredictorChangedPayload

        self._bus_subscribed = False
        bus = controller.get_bus()
        bus.subscribe(PredictorChangedPayload, self._on_predictor_changed)
        self._bus_subscribed = True
        self.finished.connect(self._cleanup_bus)
        self.destroyed.connect(self._cleanup_bus)

    # ------------------------------------------------------------------
    # Model-param widgets / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_param_spin(default: float = 0.0) -> QDoubleSpinBox:
        """A wide-range, high-precision spinbox for an editable model param.

        Range is intentionally permissive so exploratory inputs (any positive
        energy ratio, arbitrary flux anchors) are accepted; the only hard guard
        (flux_period != 0) lives in _on_apply_model_params, not here.
        """
        spin = QDoubleSpinBox()
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
        info = self._ctrl.get_predictor_info()
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

        Called after any add/delete/operator-change operation.  Preserves the
        previous row selection if the transition is still present.
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

            # Column 2: |M| placeholder — populated by _update_value_columns
            mag_item = QTableWidgetItem("—")
            self._table.setItem(row_idx, _COL_MAG, mag_item)

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
        """Fill Freq and |M| columns for every tracked transition at the current marker.

        Called by the debounce timer.  Uses point-predict for each row so each value
        is as accurate as possible.  Any failure per-row shows "—".
        """
        from zcu_tools.gui.session.services.predictor import (
            PredictFreqRequest,
            PredictMatrixCurveRequest,
            PredictorNotLoaded,
        )

        marker_value = self._predict_value_spin.value()
        for row_idx, transition in enumerate(self._tracked):
            # Freq column: per-row scalar predict_freq (most accurate)
            try:
                freq = self._ctrl.predict_freq(
                    PredictFreqRequest(value=marker_value, transition=transition)
                )
                freq_text = f"{freq:.4f}"
            except (PredictorNotLoaded, ValueError):
                freq_text = "—"
            freq_item = self._table.item(row_idx, _COL_FREQ)
            if freq_item is not None:
                freq_item.setText(freq_text)

            # |M| column: single-point matrix curve call (avoids level<=1 cap
            # in predict_matrix_element; handles any arbitrary levels).
            try:
                mat_req = PredictMatrixCurveRequest(
                    values=np.array([marker_value], dtype=np.float64),
                    transitions=(transition,),
                    operator=self._operator,  # type: ignore[arg-type]
                )
                mat_result = self._ctrl.predict_matrix_element_curve(mat_req)
                mag_text = f"{mat_result.mags[0, 0]:.5f}"
            except (PredictorNotLoaded, ValueError):
                mag_text = "—"
            mag_item = self._table.item(row_idx, _COL_MAG)
            if mag_item is not None:
                mag_item.setText(mag_text)

    # ------------------------------------------------------------------
    # Curve refresh
    # ------------------------------------------------------------------

    def _refresh_curves(self) -> None:
        """Recompute and redraw both freq and matrix-element curves."""
        from zcu_tools.gui.session.services.predictor import (
            PredictCurveRequest,
            PredictMatrixCurveRequest,
            PredictorNotLoaded,
        )

        if not self._tracked:
            self._freq_canvas.clear()
            self._mat_canvas.clear()
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
            freq_result = self._ctrl.predict_freq_curve(
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

        # ── Matrix element curves ─────────────────────────────────────────
        try:
            mat_result = self._ctrl.predict_matrix_element_curve(
                PredictMatrixCurveRequest(
                    values=grid,
                    transitions=tuple(self._tracked),
                    operator=self._operator,  # type: ignore[arg-type]
                )
            )
            self._last_mat_result = mat_result
            op_label = self._operator
            self._mat_canvas.render_curves(
                values=mat_result.values,
                labels=mat_result.labels,
                series=mat_result.mags,
                ylabel=f"|<i|{op_label}|j>|",
                highlight=highlight,
                marker_value=marker_value,
                flux_window=_DEFAULT_FLUX_WINDOW,
                value_to_flux=value_to_flux,
                flux_to_value=flux_to_value,
            )
        except (PredictorNotLoaded, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return

        # Refresh both value columns for the new marker position.
        self._update_value_columns()

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
    # Operator change
    # ------------------------------------------------------------------

    def _on_operator_changed(self, operator: str) -> None:
        """Operator combobox changed → re-run matrix curves + rebuild |M| column."""
        if operator not in ("n", "phi"):
            return  # guard against spurious signals; only accept valid values
        self._operator = operator
        self._rebuild_table()
        self._refresh_curves()

    # ------------------------------------------------------------------
    # Spinbox / canvas bidirectional coupling
    # ------------------------------------------------------------------

    def _on_spinbox_changed(self, value: float) -> None:
        """Spinbox changed → update both canvas markers; schedule debounced column update."""
        self._freq_canvas.set_marker(value)
        self._mat_canvas.set_marker(value)
        self._debounce_timer.start()

    def _on_canvas_drag(self, value: float) -> None:
        """Canvas drag → update spinbox (blockSignals to avoid loop) + both markers."""
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        # Update the other canvas's marker position too (they share one marker value).
        self._freq_canvas.set_marker(value)
        self._mat_canvas.set_marker(value)

    def _on_canvas_drop(self, value: float) -> None:
        """Canvas drop → update spinbox and trigger column recompute immediately."""
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        # Sync the other canvas's marker (it didn't receive the drag events).
        self._freq_canvas.set_marker(value)
        self._mat_canvas.set_marker(value)
        # Drop triggers an immediate update (cancel any pending debounce first).
        self._debounce_timer.stop()
        self._update_value_columns()

    def _on_selection_changed(self) -> None:
        """Table row selection changed → restyle highlighted curve on both canvases."""
        transition = self._selected_transition()
        self._freq_canvas.set_highlight(transition)
        self._mat_canvas.set_highlight(transition)

    # ------------------------------------------------------------------
    # Browse / Apply handlers
    # ------------------------------------------------------------------

    def _on_browse_file(self) -> None:
        """Open a file dialog to pick a params.json and populate the six spinboxes.

        Does NOT install a predictor — the user tweaks then clicks Apply.
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
        self._set_status("Loaded params into fields — click Apply to use", error=False)
        logger.info("PredictorDialog: loaded params into fields from %r", path)

    def _on_apply_model_params(self) -> None:
        """Build+install a predictor from the current editable fields.

        Only flux_period==0 is guarded (it makes the value<->flux affine
        singular); everything else is allowed so trial ratios work. The bus
        PredictorChangedPayload that install emits drives the state refresh.
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
            return
        req = SetModelParamsRequest(
            EJ=self._ej_spin.value(),
            EC=self._ec_spin.value(),
            EL=self._el_spin.value(),
            flux_half=self._flux_half_spin.value(),
            flux_period=flux_period,
            flux_bias=self._flux_bias_spin.value(),
        )
        try:
            self._ctrl.set_predictor_model_params(req)
        except PredictorLoadError as exc:
            self._set_status(str(exc), error=True)
            return
        self._set_status("Model params applied", error=False)
        logger.info("PredictorDialog: applied model params %r", req)

    # ------------------------------------------------------------------
    # Bus subscription
    # ------------------------------------------------------------------

    def _cleanup_bus(self, *_args: object) -> None:
        if not self._bus_subscribed:
            return
        from zcu_tools.gui.session.events import PredictorChangedPayload

        self._ctrl.get_bus().unsubscribe(
            PredictorChangedPayload, self._on_predictor_changed
        )
        self._bus_subscribed = False

    def _on_predictor_changed(self, payload: object) -> None:
        del payload
        info = self._ctrl.get_predictor_info()
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
            self._freq_canvas.clear()
            self._mat_canvas.clear()
            for row_idx in range(self._table.rowCount()):
                freq_item = self._table.item(row_idx, _COL_FREQ)
                if freq_item is not None:
                    freq_item.setText("—")
                mag_item = self._table.item(row_idx, _COL_MAG)
                if mag_item is not None:
                    mag_item.setText("—")

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")

"""PredictorDialog — load FluxoniumPredictor from params.json and predict frequencies.

Layout: left control column (load / predict / status) + right PredictorCurveCanvas.
The canvas shows f_01, f_02, f_03, f_04 transition curves vs device value (A), with
the selected transition highlighted and a draggable flux-marker that is bidirectionally
coupled to the "Flux value (A)" spinbox.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

from qtpy.QtCore import QTimer  # type: ignore[attr-defined]
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
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
# Debounce delay (ms) before updating the single-point prediction label.
_DEBOUNCE_MS = 150
# Default transitions to display on the canvas.
_DEFAULT_TRANSITIONS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (0, 4))


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
        self.setMinimumWidth(900)
        self.setMinimumHeight(480)

        # Root layout: left controls | right canvas.
        root_layout = QHBoxLayout(self)

        # ── Left: controls column ─────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # ── Load predictor ────────────────────────────────────────────────
        load_group = QGroupBox("Load predictor")
        load_form = QFormLayout(load_group)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("/path/to/params.json")
        path_row.addWidget(self._path_edit)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_file)
        path_row.addWidget(browse_btn)
        load_form.addRow("params.json:", path_row)

        self._flux_bias_spin = QDoubleSpinBox()
        self._flux_bias_spin.setRange(-1e6, 1e6)
        self._flux_bias_spin.setDecimals(6)
        self._flux_bias_spin.setValue(0.0)
        load_form.addRow("flux_bias:", self._flux_bias_spin)

        load_btn_row = QHBoxLayout()
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._on_accepted)
        load_btn_row.addWidget(load_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        load_btn_row.addWidget(clear_btn)
        load_form.addRow("", load_btn_row)

        left_layout.addWidget(load_group)

        # ── Predict frequency ─────────────────────────────────────────────
        predict_group = QGroupBox("Predict frequency")
        predict_form = QFormLayout(predict_group)

        self._predict_value_spin = QDoubleSpinBox()
        self._predict_value_spin.setRange(-1e6, 1e6)
        self._predict_value_spin.setDecimals(6)
        self._predict_value_spin.setValue(0.0)
        self._predict_value_spin.valueChanged.connect(self._on_spinbox_changed)
        predict_form.addRow("Flux value (A):", self._predict_value_spin)

        transition_row = QHBoxLayout()
        self._from_spin = QSpinBox()
        self._from_spin.setRange(0, 20)
        self._from_spin.setValue(0)
        self._to_spin = QSpinBox()
        self._to_spin.setRange(0, 20)
        self._to_spin.setValue(1)
        self._from_spin.valueChanged.connect(self._on_transition_changed)
        self._to_spin.valueChanged.connect(self._on_transition_changed)
        transition_row.addWidget(QLabel("from"))
        transition_row.addWidget(self._from_spin)
        transition_row.addWidget(QLabel("to"))
        transition_row.addWidget(self._to_spin)
        predict_form.addRow("Transition:", transition_row)

        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self._on_predict_clicked)
        predict_form.addRow("", predict_btn)

        self._predict_result_label = QLabel("—")
        self._predict_result_label.setStyleSheet("font-weight: bold;")
        predict_form.addRow("Result:", self._predict_result_label)

        left_layout.addWidget(predict_group)

        # ── status label ──────────────────────────────────────────────────
        self._status_label = QLabel("")
        left_layout.addWidget(self._status_label)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)  # type: ignore[attr-defined]
        btn_box.rejected.connect(self.reject)
        left_layout.addWidget(btn_box)

        left_layout.addStretch()

        # ── Right: canvas ─────────────────────────────────────────────────
        self._canvas = PredictorCurveCanvas(figsize=(6.0, 4.0))
        self._canvas.bind_callbacks(
            on_drag=self._on_canvas_drag,
            on_drop=self._on_canvas_drop,
        )

        root_layout.addWidget(left_widget, stretch=0)
        root_layout.addWidget(self._canvas, stretch=1)

        # ── Debounce timer for the single-point predict label ─────────────
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._on_predict_clicked)

        # ── Predictor info cache (flux_half, flux_period for conversions) ──
        self._flux_half: float | None = None
        self._flux_period: float | None = None

        # pre-fill with current predictor state
        info = controller.get_predictor_info()
        if info is not None:
            if info["path"] is not None:
                self._path_edit.setText(info["path"])
            self._flux_bias_spin.setValue(info["flux_bias"])
            self._flux_half = info["flux_half"]
            self._flux_period = info["flux_period"]
            self._set_status("Currently loaded", error=False)
            self._refresh_curves()

        # EventBus subscription for live predictor state updates
        from zcu_tools.gui.session.events import PredictorChangedPayload

        self._bus_subscribed = False
        bus = controller.get_bus()
        bus.subscribe(PredictorChangedPayload, self._on_predictor_changed)
        self._bus_subscribed = True
        self.finished.connect(self._cleanup_bus)
        self.destroyed.connect(self._cleanup_bus)

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
    # Curve refresh
    # ------------------------------------------------------------------

    def _refresh_curves(self) -> None:
        """Recompute and redraw all transition curves."""
        from zcu_tools.gui.session.services.connection import (
            PredictCurveRequest,
            PredictorNotLoaded,
        )

        value_to_flux, flux_to_value = self._make_affine()
        if value_to_flux is None or flux_to_value is None:
            return

        # Build value grid from the default flux window.
        v_lo = flux_to_value(_DEFAULT_FLUX_WINDOW[0])
        v_hi = flux_to_value(_DEFAULT_FLUX_WINDOW[1])
        if v_lo > v_hi:
            v_lo, v_hi = v_hi, v_lo
        grid = np.linspace(v_lo, v_hi, _CURVE_GRID_N, dtype=np.float64)

        try:
            result = self._ctrl.predict_freq_curve(
                PredictCurveRequest(values=grid, transitions=_DEFAULT_TRANSITIONS)
            )
        except (PredictorNotLoaded, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return

        highlight = (self._from_spin.value(), self._to_spin.value())
        marker_value = self._predict_value_spin.value()

        self._canvas.render_curves(
            result,
            highlight=highlight,
            marker_value=marker_value,
            flux_window=_DEFAULT_FLUX_WINDOW,
            value_to_flux=value_to_flux,
            flux_to_value=flux_to_value,
        )

    # ------------------------------------------------------------------
    # Spinbox / canvas bidirectional coupling
    # ------------------------------------------------------------------

    def _on_spinbox_changed(self, value: float) -> None:
        """Spinbox changed → update marker; schedule debounced label update."""
        self._canvas.set_marker(value)
        self._debounce_timer.start()

    def _on_canvas_drag(self, value: float) -> None:
        """Canvas drag → update spinbox (blockSignals to avoid loop) + visual marker."""
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        # Canvas already moved the marker in set_marker; nothing else needed.

    def _on_canvas_drop(self, value: float) -> None:
        """Canvas drop → update spinbox and trigger label recompute immediately."""
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        # Drop triggers an immediate predict (cancel any pending debounce first).
        self._debounce_timer.stop()
        self._on_predict_clicked()

    def _on_transition_changed(self) -> None:
        """Transition spinbox changed → restyle existing curve artists; no recompute."""
        self._canvas.set_highlight((self._from_spin.value(), self._to_spin.value()))

    # ------------------------------------------------------------------
    # File / load / clear / predict handlers
    # ------------------------------------------------------------------

    def _on_browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select params.json", "", "JSON files (*.json);;All files (*)"
        )
        if path:
            self._path_edit.setText(path)

    def _on_accepted(self) -> None:
        from zcu_tools.gui.session.services.connection import (
            LoadPredictorRequest,
            PredictorLoadError,
        )

        path = self._path_edit.text().strip()
        flux_bias = self._flux_bias_spin.value()
        try:
            self._ctrl.load_predictor(
                LoadPredictorRequest(path=path, flux_bias=flux_bias)
            )
        except PredictorLoadError as exc:
            self._set_status(str(exc), error=True)
            return
        self._set_status("Predictor loaded", error=False)
        logger.info("PredictorDialog: loaded path=%r", path)

    def _on_clear(self) -> None:
        self._ctrl.clear_predictor()
        self._predict_result_label.setText("—")
        self._set_status("Predictor cleared")
        self._flux_half = None
        self._flux_period = None
        self._canvas.clear()
        logger.info("PredictorDialog: predictor cleared")

    def _on_predict_clicked(self) -> None:
        from zcu_tools.gui.session.services.connection import (
            PredictFreqRequest,
            PredictorNotLoaded,
        )

        value = self._predict_value_spin.value()
        from_lvl = self._from_spin.value()
        to_lvl = self._to_spin.value()
        transition = (from_lvl, to_lvl)
        try:
            freq = self._ctrl.predict_freq(
                PredictFreqRequest(value=value, transition=transition)
            )
        except PredictorNotLoaded as exc:
            self._set_status(str(exc), error=True)
            return
        self._predict_result_label.setText(f"{freq:.4f} MHz")
        self._set_status(
            f"Predicted ({from_lvl},{to_lvl}) @ {value:.6g}: {freq:.4f} MHz"
        )
        logger.info(
            "PredictorDialog: predict value=%r transition=%r → %.4f MHz",
            value,
            transition,
            freq,
        )

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
            if info["path"] is not None:
                self._path_edit.setText(info["path"])
            self._flux_bias_spin.setValue(info["flux_bias"])
            self._flux_half = info["flux_half"]
            self._flux_period = info["flux_period"]
            self._set_status("Currently loaded", error=False)
            self._refresh_curves()
        else:
            self._flux_half = None
            self._flux_period = None
            self._set_status("Not loaded", error=False)
            self._canvas.clear()

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")

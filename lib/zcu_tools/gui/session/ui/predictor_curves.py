"""Predictor curve tabs, query presentation and cross-canvas interaction state."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from zcu_tools.analysis.flux import FluxFrame, flux_from_device_values
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.services.predictor import PredictMatrixCurveRequest
from zcu_tools.gui.session.ui.predictor_canvas import CurveView, PredictorCurveCanvas
from zcu_tools.simulate.parameters import ModelParams
from zcu_tools.simulate.predictor import MatrixPredictor, PredictorInputError

if TYPE_CHECKING:
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

logger = logging.getLogger(__name__)

_DEFAULT_FLUX_WINDOW: tuple[float, float] = (0.4, 1.1)
_CURVE_GRID_N = 200


@dataclass(frozen=True)
class CurveCoordinates:
    """Native scan coordinates shared by the frequency and matrix plots.

    frame holds Phi0 offset and signed native periods/units. axis_device names
    one frame device. device_values fixes every other device in its native unit.
    """

    frame: FluxFrame
    axis_device: str
    device_values: Mapping[str, float]


class PredictorCurveTabs(QTabWidget):
    """Own three canvases/toolbars and shared marker, highlight and x-range state.

    predictor supplies explicit-parameter queries, never an installed model.
    Callbacks execute synchronously on the Qt owner thread. on_follow/on_lock
    receive native device values, respectively for debounced motion and immediate
    settlement; the caller coordinates its device input and table. on_status
    receives (message, is_error). on_clear_values clears the caller's table when
    invalid physics clears these plots. on_matrix_capability receives whether
    matrix columns should be shown after successful frequency rendering.
    Callback exceptions are not relayed through Qt signals or discarded.
    Inherited currentChanged(int) reports the selected tab for UI preferences.
    """

    def __init__(
        self,
        predictor: PredictorControlPort,
        *,
        on_follow: Callable[[float], None],
        on_lock: Callable[[float], None],
        on_status: Callable[[str, bool], None],
        on_clear_values: Callable[[], None],
        on_matrix_capability: Callable[[bool], None],
    ) -> None:
        super().__init__()
        self._pred = predictor
        self._on_follow = on_follow
        self._on_lock = on_lock
        self._on_status = on_status
        self._on_clear_values = on_clear_values
        self._on_matrix_capability = on_matrix_capability
        self._synced_xlim: tuple[float, float] | None = None
        self._syncing_xlim = False
        self._matrix_supported = True
        self._build_tabs()

    def _build_tabs(self) -> None:
        # ── Right: QTabWidget with three canvases ─────────────────────────
        # Frequency, |n|, |phi|. All three share one marker value and one set of
        # follow/lock callbacks, so a click-follow on any tab drives the others.
        # Each tab owns a native Matplotlib toolbar; x-ranges stay synced across
        # tabs while y-ranges remain per-tab because their units differ.

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
                on_follow=self._on_follow,
                on_lock=self._on_lock,
            )
            canvas.bind_xlim_changed(self._on_canvas_xlim_changed)

        self._freq_tab, self._freq_toolbar = self._make_canvas_tab(self._freq_canvas)
        self._mat_n_tab, self._mat_n_toolbar = self._make_canvas_tab(self._mat_n_canvas)
        self._mat_phi_tab, self._mat_phi_toolbar = self._make_canvas_tab(
            self._mat_phi_canvas
        )
        self._all_toolbars = (
            self._freq_toolbar,
            self._mat_n_toolbar,
            self._mat_phi_toolbar,
        )

        self.addTab(self._freq_tab, "Frequency")
        self.addTab(self._mat_n_tab, "|n|")
        self.addTab(self._mat_phi_tab, "|phi|")

    @staticmethod
    def _make_canvas_tab(
        canvas: PredictorCurveCanvas,
    ) -> tuple[QWidget, NavigationToolbar2QT]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = NavigationToolbar2QT(canvas.canvas, page)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, stretch=1)
        return page, toolbar

    def clear_curves(self) -> None:
        """Clear all plots and their shared user x-range; leave tab choice unchanged."""
        self._synced_xlim = None
        for canvas in self._all_canvases:
            canvas.clear()

    def set_interaction_enabled(self, *, enabled: bool) -> None:
        """Enable/disable marker interaction on every canvas without locking tabs."""
        for canvas in self._all_canvases:
            canvas.set_interaction_enabled(enabled=enabled)

    def set_marker(self, value: float) -> None:
        """Move the marker on every canvas to ``value`` (no recompute)."""
        for canvas in self._all_canvases:
            canvas.set_marker(value)

    def set_highlights(self, highlights: tuple[tuple[int, int], ...]) -> None:
        """Restyle these transition pairs on every plot without querying the model."""
        for canvas in self._all_canvases:
            canvas.set_highlights(highlights)

    def refresh_curves(
        self,
        params: ModelParams,
        transitions: tuple[tuple[int, int], ...],
        highlights: tuple[tuple[int, int], ...],
        marker_value: float,
        *,
        coordinates: CurveCoordinates,
    ) -> bool:
        """Render queries for explicit trial inputs, transitions and highlighted rows.

        Native marker_value is shared by all tabs. Returns True only after all
        supported curves render. Empty transitions clear plots only. Invalid
        physics clears plots/xlim and calls on_clear_values before on_status.
        Query failures are displayed (unexpected failures are logged) and return
        False. Successful frequency rendering updates matrix capability before
        querying matrix curves. Other callback/rendering failures propagate.
        """
        from zcu_tools.gui.session.services.predictor import (
            PredictCurveRequest,
        )

        if not transitions:
            for canvas in self._all_canvases:
                canvas.clear()
            return False

        try:
            model = params.build_predictor()
        except PredictorInputError as exc:
            self.clear_curves()
            self._on_clear_values()
            self._set_status(str(exc), error=True)
            return False
        self._matrix_supported = isinstance(model, MatrixPredictor)
        frame, axis, fixed = (
            coordinates.frame,
            coordinates.axis_device,
            coordinates.device_values,
        )

        # Matplotlib secondary-axis transforms also supply NumPy arrays. Keep
        # the transforms linear after the public flux owner projects the origin.
        origin = flux_from_device_values(frame, {**fixed, axis: 0.0})
        period = frame.devices[axis].period

        def value_to_flux(value: float) -> float:
            return origin + value / period

        def flux_to_value(flux: float) -> float:
            return (flux - origin) * period

        # Build value grid from the default flux window.
        v_lo = flux_to_value(_DEFAULT_FLUX_WINDOW[0])
        v_hi = flux_to_value(_DEFAULT_FLUX_WINDOW[1])
        if v_lo > v_hi:
            v_lo, v_hi = v_hi, v_lo
        grid = np.linspace(v_lo, v_hi, _CURVE_GRID_N, dtype=np.float64)

        view = CurveView(
            highlights=highlights,
            marker_value=marker_value,
            flux_window=_DEFAULT_FLUX_WINDOW,
            value_to_flux=value_to_flux,
            flux_to_value=flux_to_value,
            value_label=f"{axis} ({frame.devices[axis].unit})",
        )

        # ── Frequency curves ──────────────────────────────────────────────
        try:
            freq_result = self._pred.predict_freq_curve(
                params,
                PredictCurveRequest(frame, axis, grid, fixed, tuple(transitions)),
            )
            self._freq_canvas.render_curves(
                values=freq_result.values,
                labels=freq_result.labels,
                series=freq_result.freqs_mhz,
                ylabel="Frequency (MHz)",
                view=view,
            )
        except (FailedPreconditionError, PredictorInputError) as exc:
            self._set_status(str(exc), error=True)
            return False
        except Exception as exc:
            logger.exception("predictor frequency curve refresh failed")
            self._set_status(str(exc), error=True)
            return False

        # ── Matrix element curves (both operators, one tab each) ───────────
        self._sync_matrix_capability()
        matrix_canvases: tuple[
            tuple[Literal["n", "phi"], PredictorCurveCanvas], ...
        ] = (
            ("n", self._mat_n_canvas),
            ("phi", self._mat_phi_canvas),
        )
        if self._matrix_supported and not all(
            self._render_matrix_curve(
                params,
                PredictMatrixCurveRequest(
                    frame, axis, grid, fixed, transitions, operator
                ),
                canvas,
                view,
            )
            for operator, canvas in matrix_canvases
        ):
            return False

        self._apply_synced_xlim_to_all()

        # Refresh all value columns for the new marker position.
        return True

    def _sync_matrix_capability(self) -> None:
        """Update matrix tab availability and notify the table column owner."""
        supported = self._matrix_supported
        for index, canvas in (
            (1, self._mat_n_canvas),
            (2, self._mat_phi_canvas),
        ):
            self.setTabEnabled(index, supported)
            if not supported:
                canvas.clear()
        self._on_matrix_capability(supported)
        if not supported and self.currentIndex() != 0:
            self.setCurrentIndex(0)

    def _render_matrix_curve(
        self,
        params: ModelParams,
        request: PredictMatrixCurveRequest,
        canvas: PredictorCurveCanvas,
        view: CurveView,
    ) -> bool:
        """Compute+render one operator's matrix-element curve onto ``canvas``.

        Returns True on success; on an expected failure it sets the status and
        returns False so the caller can stop the refresh.
        """
        try:
            mat_result = self._pred.predict_matrix_element_curve(
                params,
                request,
            )
        except (FailedPreconditionError, PredictorInputError) as exc:
            self._set_status(str(exc), error=True)
            return False
        except Exception as exc:
            logger.exception(
                "predictor matrix curve refresh failed: operator=%s", request.operator
            )
            self._set_status(str(exc), error=True)
            return False

        canvas.render_curves(
            values=mat_result.values,
            labels=mat_result.labels,
            series=mat_result.mags,
            ylabel=f"|<i|{request.operator}|j>|",
            view=view,
        )
        return True

    def _on_canvas_xlim_changed(
        self,
        source: PredictorCurveCanvas,
        xlim: tuple[float, float],
    ) -> None:
        """Propagate one canvas' x-range to every predictor tab."""
        if self._syncing_xlim:
            return
        self._synced_xlim = (float(xlim[0]), float(xlim[1]))
        self._syncing_xlim = True
        try:
            for canvas in self._all_canvases:
                if canvas is source:
                    continue
                canvas.apply_xlim(self._synced_xlim, notify=False)
        finally:
            self._syncing_xlim = False

    def _apply_synced_xlim_to_all(self) -> None:
        if self._synced_xlim is None:
            return
        self._syncing_xlim = True
        try:
            for canvas in self._all_canvases:
                canvas.apply_xlim(self._synced_xlim, notify=False)
        finally:
            self._syncing_xlim = False

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._on_status(message, error)

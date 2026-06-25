"""SelectorWidget — cross-spectrum joint-point-cloud filtering.

Port of the notebook's InteractiveSelector: overlays every spectrum's heatmap as
the background, scatters the joint point cloud (assembled from all spectra's
selected points), and lets a circular brush select/erase points across the whole
collection. A distance threshold downsamples the kept points (``downsample_points``,
reused verbatim). The transient brush circle uses a Qt QTimer (the notebook used a
threading.Timer).
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.patches import Ellipse
from numpy.typing import NDArray
from qtpy.QtCore import (  # type: ignore[attr-defined]
    Qt,
    QTimer,
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from zcu_tools.analysis.fluxdep import (
    cast2real_and_norm,
    downsample_points,
    points_in_normalized_brush,
)
from zcu_tools.gui.background import BackgroundRunner
from zcu_tools.notebook.persistance import SpectrumResult

from .base import InteractiveMplWidget

_SCALE = 1000


class SelectorWidget(InteractiveMplWidget):
    """Brush selection over the joint point cloud of several spectra."""

    def __init__(
        self,
        spectrums: dict[str, SpectrumResult],
        min_distance: float = 0.0,
        brush_width: float = 0.05,
        parent: QWidget | None = None,
    ) -> None:
        # Controls on the LEFT to match the Search / Show tabs (the cross-spectrum
        # filter lives in the same Analyze panel).
        super().__init__(parent, controls_side="left")
        self._spectrums = spectrums
        self._s_fluxs = np.concatenate(
            [s["points"]["fluxs"] for s in spectrums.values()]
        )
        self._s_freqs = np.concatenate(
            [s["points"]["freqs"] for s in spectrums.values()]
        )
        # Start with EVERYTHING selected (do not inherit the previous brush
        # selection — removed points would be hard to bring back). The stable
        # downsample threshold IS inherited via min_distance.
        self._selected = np.ones_like(self._s_fluxs, dtype=bool)
        self._init_min_distance = min_distance
        self._filter_mask = np.ones_like(self._selected, dtype=bool)
        self._temp_circle: Ellipse | None = None
        self._temp_timer: QTimer | None = None

        # Off-main-thread downsample with instant-cancel by generation (O(N²)).
        self._runner = BackgroundRunner(self)
        self._generation = 0
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._launch_worker)

        self._compute_bounds()
        self._build_controls(brush_width)
        self._init_plots()
        # Initial render is synchronous so the figure is complete on first show
        # (the debounced apply_filter only runs later, on interaction).
        self._run_filter()

    # --- bounds + controls ----------------------------------------------

    def _compute_bounds(self) -> None:
        spects = [s["spectrum"] for s in self._spectrums.values()]
        sp_fluxs = np.concatenate([s["fluxs"] for s in spects])
        sp_freqs = np.concatenate([s["freqs"] for s in spects])
        self._flux_bound = (
            float(min(np.nanmin(sp_fluxs), self._s_fluxs.min())),
            float(max(np.nanmax(sp_fluxs), self._s_fluxs.max())),
        )
        self._freq_bound = (
            float(min(np.nanmin(sp_freqs), self._s_freqs.min())),
            float(max(np.nanmax(sp_freqs), self._s_freqs.max())),
        )

    def _slider(self, lo: float, hi: float, val: float) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setMinimum(int(lo * _SCALE))
        s.setMaximum(int(hi * _SCALE))
        s.setValue(int(val * _SCALE))
        return s

    def _build_controls(self, brush_width: float) -> None:
        self.controls_layout.addWidget(QLabel("Brush width"))
        self._width = self._slider(0.01, 0.1, brush_width)
        self.controls_layout.addWidget(self._width)

        self.controls_layout.addWidget(QLabel("Min distance"))
        self._thresh = self._slider(0.0, 0.1, self._init_min_distance)
        self._thresh.valueChanged.connect(lambda _v: self.apply_filter())
        self.controls_layout.addWidget(self._thresh)

        self._operation = QComboBox()
        self._operation.addItems(["Select", "Erase"])
        self.controls_layout.addWidget(self._operation)

        perform_all = QPushButton("Perform on all")
        perform_all.clicked.connect(self._on_perform_all)
        self.controls_layout.addWidget(perform_all)

        # "Apply" (not "Finish"): the cross-spectrum filter is not a terminal
        # pipeline step — it commits the current selection to State for Search.
        self.apply_button = self.add_finish_button("Apply")
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.controls_layout.addWidget(self.status_label)

    def _width_val(self) -> float:
        return self._width.value() / _SCALE

    def _thresh_val(self) -> float:
        return self._thresh.value() / _SCALE

    def _operation_select(self) -> bool:
        return self._operation.currentText() == "Select"

    # --- plotting --------------------------------------------------------

    def _init_plots(self) -> None:
        self._ax = self.figure.add_subplot(1, 1, 1)
        for name, spect in self._spectrums.items():
            t0 = time.perf_counter()
            signals = spect["spectrum"]["signals"]
            flux_mask = np.any(~np.isnan(signals), axis=1)
            freq_mask = np.any(~np.isnan(signals), axis=0)
            signals = signals[flux_mask, :][:, freq_mask]
            sp_fluxs = spect["spectrum"]["fluxs"][flux_mask]
            sp_freqs = spect["spectrum"]["freqs"][freq_mask]
            if sp_fluxs.size == 0 or sp_freqs.size == 0:
                continue
            # Contrast boost on the REAL magnitude (cast2real_and_norm already
            # takes abs): a fractional power on the complex array is ~30x slower
            # for no benefit (it gets abs'd anyway).
            real_signals = cast2real_and_norm(signals) ** 1.5
            logger.debug(
                "selector background %r: cast %.0fms, shape=%s",
                name,
                (time.perf_counter() - t0) * 1000,
                real_signals.shape,
            )
            self._ax.imshow(
                real_signals.T,
                aspect="auto",
                origin="lower",
                interpolation="antialiased",  # downsample large images (was "none")
                extent=(sp_fluxs[0], sp_fluxs[-1], sp_freqs[0], sp_freqs[-1]),
                cmap="gray_r",  # neutral grayscale so the coloured points stand out
            )
        # Kept points red, dropped points a saturated blue: red/blue is a
        # warm/cool complementary pair, so both stand out against the gray_r
        # background and against each other (the earlier faint blue-grey had too
        # little contrast with the red). Kept points are also drawn larger than
        # dropped so the selection reads at a glance. Two separate scatters so the
        # kept points draw ON TOP of the dropped ones (a single scatter draws in
        # index order, letting dropped points cover kept ones).
        self._scatter_dropped = self._ax.scatter([], [], c="#1f77ff", s=6, zorder=2)
        self._scatter_kept = self._ax.scatter([], [], c="#e02020", s=18, zorder=3)
        self._update_scatters()
        self._ax.set_xlim(*self._flux_bound)
        self._ax.set_ylim(*self._freq_bound)
        self._ax.set_xlabel("Flux")
        self._ax.set_ylabel("Frequency (GHz)")

    def _cur_selected(self) -> NDArray[np.bool_]:
        cur = np.zeros_like(self._selected, dtype=bool)
        cur[np.where(self._selected)[0][self._filter_mask]] = True
        return cur

    def _update_scatters(self) -> None:
        """Put kept points in the top scatter, dropped in the bottom one."""
        kept = self._cur_selected()

        def _offsets(mask: NDArray[np.bool_]) -> NDArray[np.float64]:
            if not mask.any():
                return np.empty((0, 2), dtype=np.float64)
            return np.column_stack((self._s_fluxs[mask], self._s_freqs[mask]))

        self._scatter_kept.set_offsets(_offsets(kept))
        self._scatter_dropped.set_offsets(_offsets(~kept))

    def _selected_normalised(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        flux_span = self._flux_bound[1] - self._flux_bound[0]
        freq_span = self._freq_bound[1] - self._freq_bound[0]
        return (
            self._s_fluxs[self._selected] / flux_span,
            self._s_freqs[self._selected] / freq_span,
        )

    def apply_filter(self) -> None:
        """Re-apply the downsample filter, debounced + off-main (interactive use).

        The public entry for slider / brush changes: it just schedules the
        synchronous worker (``_run_filter``) after a debounce so rapid changes
        coalesce and the O(N²) downsample never stalls the UI. The first render
        calls ``_run_filter`` directly (below) so the figure is complete at once.
        """
        self._generation += 1
        self._debounce.start()

    def _run_filter(self) -> None:
        """Compute the downsample filter synchronously and redraw at once.

        The core both paths share: the debounced ``apply_filter`` runs it on a
        worker thread; the initial render calls it directly so the figure shows
        its final content immediately (rather than blank until a tab switch
        forced a repaint).
        """
        self._generation += 1
        sel_x, sel_y = self._selected_normalised()
        mask = downsample_points(sel_x, sel_y, self._thresh_val())
        self._set_filter_mask(mask)

    def _launch_worker(self) -> None:
        # Capture the generation + a parameter snapshot in the closure; the
        # staleness check (cancellation by discarding stale-generation results)
        # stays in this widget, not the runner.
        generation = self._generation
        sel_x, sel_y = self._selected_normalised()
        threshold = self._thresh_val()

        def _compute() -> NDArray[np.bool_]:
            return downsample_points(sel_x, sel_y, threshold)

        self._runner.submit(
            _compute,
            on_done=lambda mask, g=generation: self._on_worker_done(g, mask),
            on_error=self._on_worker_error,
            run_in_pool=True,
        )

    def _on_worker_error(self, exc: Exception) -> None:
        # A downsample failing is non-fatal (a transient parameter combo); log and
        # keep the prior filter on screen rather than crashing the selector.
        logger.exception("downsample worker failed", exc_info=exc)

    def _on_worker_done(self, generation: int, filter_mask: NDArray[np.bool_]) -> None:
        if generation != self._generation:
            return  # superseded by a newer change — drop the stale result
        self._set_filter_mask(filter_mask)

    def _set_filter_mask(self, filter_mask: NDArray[np.bool_]) -> None:
        """Record the downsample mask, re-split kept/dropped scatters, and redraw."""
        self._filter_mask = filter_mask
        self._update_scatters()
        self.redraw()

    # --- interaction -----------------------------------------------------

    def on_press(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None:
            return
        x, y, width = float(event.xdata), float(event.ydata), self._width_val()
        toggle = points_in_normalized_brush(
            self._s_fluxs,
            self._s_freqs,
            x=x,
            y=y,
            width=width,
            x_bound=self._flux_bound,
            y_bound=self._freq_bound,
        )
        self._selected[toggle] = self._operation_select()
        self._show_temp_circle(x, y, width)
        self.apply_filter()

    def _on_perform_all(self) -> None:
        self._selected[:] = self._operation_select()
        self.apply_filter()

    def _show_temp_circle(self, x: float, y: float, width: float) -> None:
        if self._temp_circle is not None:
            self._temp_circle.remove()
            self._temp_circle = None
        if self._temp_timer is not None:
            self._temp_timer.stop()
            self._temp_timer = None

        x_range = self._flux_bound[1] - self._flux_bound[0]
        y_range = self._freq_bound[1] - self._freq_bound[0]
        color = "yellow" if self._operation_select() else "black"
        self._temp_circle = Ellipse(
            (x, y),
            width=width * x_range * 2,
            height=width * y_range * 2,
            angle=0,
            fill=False,
            color=color,
            linestyle="--",
            linewidth=1,
        )
        self._ax.add_patch(self._temp_circle)

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._remove_temp_circle)
        self._temp_timer = timer
        timer.start(1000)

    def _remove_temp_circle(self) -> None:
        if self._temp_circle is not None:
            self._temp_circle.remove()
            self._temp_circle = None
            self.redraw()
        self._temp_timer = None

    # --- result ----------------------------------------------------------

    def quiesce(self) -> None:
        """Stop the debounce timer and join any in-flight pool worker.

        Call this before ``deleteLater()`` (e.g. from ``AnalyzePanelWidget._refresh_filter_tab``
        or the host window's ``closeEvent``) to prevent a pending ``QMetaCallEvent``
        from being dispatched onto a freed C++ object after the widget is destroyed.
        """
        self._debounce.stop()
        self._runner.quiesce()

    def get_result(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        # Finish is terminal: compute the final downsample synchronously from the
        # current parameters (a pending worker may not have run yet).
        sel_x, sel_y = self._selected_normalised()
        self._filter_mask = downsample_points(sel_x, sel_y, self._thresh_val())
        cur = self._cur_selected()
        return self._s_fluxs[cur], self._s_freqs[cur], cur

    def min_distance(self) -> float:
        """The current downsample threshold (remembered for the next session)."""
        return self._thresh_val()

"""SelectorWidget — cross-spectrum joint-point-cloud filtering.

Port of the notebook's InteractiveSelector: overlays every spectrum's heatmap as
the background, scatters the joint point cloud (assembled from all spectra's
selected points), and lets a circular brush select/erase points across the whole
collection. A distance threshold downsamples the kept points (``downsample_points``,
reused verbatim). The transient brush circle uses a Qt QTimer (the notebook used a
threading.Timer).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.patches import Ellipse
from numpy.typing import NDArray
from qtpy.QtCore import (  # type: ignore[attr-defined]
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,  # type: ignore[attr-defined]
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from zcu_tools.fluxdep_gui.ui.interactive.base import InteractiveMplWidget
from zcu_tools.notebook.analysis.fluxdep.processing import (
    cast2real_and_norm,
    downsample_points,
)
from zcu_tools.notebook.persistance import SpectrumResult

_SCALE = 1000


class _DownsampleSignals(QObject):
    done = Signal(int, object)  # (generation, filter_mask)


class _DownsampleWorker(QRunnable):
    """Off-main-thread downsample_points for one parameter set.

    O(N²), so it stalls the UI on large point clouds (measured ~1.3 s at 5000
    points). Generation-stamped: a result whose generation is stale (a newer
    parameter change happened) is dropped by the widget — the "instant cancel".
    """

    def __init__(
        self,
        generation: int,
        sel_x: NDArray[np.float64],
        sel_y: NDArray[np.float64],
        threshold: float,
        sink: _DownsampleSignals,
    ) -> None:
        super().__init__()
        self._gen = generation
        self._sel_x = sel_x
        self._sel_y = sel_y
        self._threshold = threshold
        self._sink = sink

    def run(self) -> None:
        mask = downsample_points(self._sel_x, self._sel_y, self._threshold)
        self._sink.done.emit(self._gen, mask)


class SelectorWidget(InteractiveMplWidget):
    """Brush selection over the joint point cloud of several spectra."""

    def __init__(
        self,
        spectrums: dict[str, SpectrumResult],
        min_distance: float = 0.0,
        brush_width: float = 0.05,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
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
        self._temp_circle: Optional[Ellipse] = None
        self._temp_timer: Optional[QTimer] = None

        # Off-main-thread downsample with instant-cancel by generation (O(N²)).
        self._pool = QThreadPool.globalInstance() or QThreadPool(self)
        self._worker_signals = _DownsampleSignals()
        self._worker_signals.done.connect(self._on_worker_done)
        self._generation = 0
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)
        self._debounce.timeout.connect(self._launch_worker)

        self._compute_bounds()
        self._build_controls(brush_width)
        self._init_plots()
        self.apply_filter()

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

        self.add_finish_button()

    def _width_val(self) -> float:
        return self._width.value() / _SCALE

    def _thresh_val(self) -> float:
        return self._thresh.value() / _SCALE

    def _operation_select(self) -> bool:
        return self._operation.currentText() == "Select"

    # --- plotting --------------------------------------------------------

    def _init_plots(self) -> None:
        self._ax = self.figure.add_subplot(1, 1, 1)
        for spect in self._spectrums.values():
            signals = spect["spectrum"]["signals"] ** 1.5  # improve contrast
            flux_mask = np.any(~np.isnan(signals), axis=1)
            freq_mask = np.any(~np.isnan(signals), axis=0)
            signals = signals[flux_mask, :][:, freq_mask]
            real_signals = cast2real_and_norm(signals)
            sp_fluxs = spect["spectrum"]["fluxs"][flux_mask]
            sp_freqs = spect["spectrum"]["freqs"][freq_mask]
            if sp_fluxs.size == 0 or sp_freqs.size == 0:
                continue
            self._ax.imshow(
                real_signals.T,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=(sp_fluxs[0], sp_fluxs[-1], sp_freqs[0], sp_freqs[-1]),
                cmap="gray_r",  # neutral grayscale so the coloured points stand out
            )
        self._scatter = self._ax.scatter(
            self._s_fluxs,
            self._s_freqs,
            c=self._selected.astype(float),
            s=2,
            vmin=0,
            vmax=1,
        )
        self._ax.set_xlim(*self._flux_bound)
        self._ax.set_ylim(*self._freq_bound)
        self._ax.set_xlabel("Flux")
        self._ax.set_ylabel("Frequency (GHz)")

    def _cur_selected(self) -> NDArray[np.bool_]:
        cur = np.zeros_like(self._selected, dtype=bool)
        cur[np.where(self._selected)[0][self._filter_mask]] = True
        return cur

    def _selected_normalised(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        flux_span = self._flux_bound[1] - self._flux_bound[0]
        freq_span = self._freq_bound[1] - self._freq_bound[0]
        return (
            self._s_fluxs[self._selected] / flux_span,
            self._s_freqs[self._selected] / freq_span,
        )

    def apply_filter(self) -> None:
        """Schedule a (debounced, off-main-thread) downsample re-computation."""
        self._generation += 1
        self._debounce.start()

    def _launch_worker(self) -> None:
        sel_x, sel_y = self._selected_normalised()
        worker = _DownsampleWorker(
            self._generation, sel_x, sel_y, self._thresh_val(), self._worker_signals
        )
        self._pool.start(worker)

    def _on_worker_done(self, generation: int, filter_mask: NDArray[np.bool_]) -> None:
        if generation != self._generation:
            return  # superseded by a newer change — drop the stale result
        self._filter_mask = filter_mask
        self._scatter.set_array(self._cur_selected().astype(float))
        self.redraw()

    # --- interaction -----------------------------------------------------

    def on_press(self, event: MouseEvent) -> None:
        if event.xdata is None or event.ydata is None:
            return
        x, y, width = float(event.xdata), float(event.ydata), self._width_val()
        x_d = np.abs(self._s_fluxs - x) / (self._flux_bound[1] - self._flux_bound[0])
        y_d = np.abs(self._s_freqs - y) / (self._freq_bound[1] - self._freq_bound[0])
        toggle = x_d**2 + y_d**2 <= width**2
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

        self._temp_timer = QTimer(self)
        self._temp_timer.setSingleShot(True)
        self._temp_timer.timeout.connect(self._remove_temp_circle)
        self._temp_timer.start(1000)

    def _remove_temp_circle(self) -> None:
        if self._temp_circle is not None:
            self._temp_circle.remove()
            self._temp_circle = None
            self.redraw()
        self._temp_timer = None

    # --- result ----------------------------------------------------------

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

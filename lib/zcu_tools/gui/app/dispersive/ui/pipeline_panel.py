"""PipelinePanelWidget — the dispersive single-flow analysis panel.

Layout: the step controls sit in compact rows at the top (steps 1/2/3 side by side,
step 4 on its own row), and a single tabbed figure area below holds the plots
(Preprocess diagnostic / Tune) so each gets the full width and height instead of
competing for vertical space.

The pipeline (each step enabled only once the prior completes — fast-fail UX):

1. **Project & inputs** — load the fluxonium fit from params.json (the hard gate).
2. **Load one-tone** — browse + transpose a raw one-tone hdf5.
3. **Preprocess** — run the (off-main) signal pipeline → the Preprocess tab.
4. **Tune g / r_f** — drag the g / r_f sliders (or click "Auto tune" to let a scipy
   optimiser pre-fit them against the dropped sample-flux lines), then "Use these
   g/r_f" runs the predictor off-main, draws the Tune tab, and records the result (the
   accepted values are the final fit). The button is disabled while computing.
5. **Export** — write the dispersive section back to params.json.

The heavy work (preprocess, predict, auto-tune) runs off-main via the shared
``BackgroundRunner`` (pool strategy), calling only the pure ``compute_*`` /
``predict_*`` / ``auto_tune`` controller methods and returning plain data; the
runner's ``on_done`` callback — on the Qt main thread — records State and draws the
figure (the worker never touches Qt widgets or pyplot, per ADR-0017). The figures
live on local ``FigureCanvasQTAgg`` widgets in the tabs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.services.viz import (
    SampleArtists,
    add_sample_line,
    move_sample_line,
    remove_sample_line,
    render_tune_figure,
    set_dispersion_lines,
    update_bare_line,
    update_sample_dots,
)
from zcu_tools.gui.app.dispersive.state import PreprocessResult
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.widgets.project_dialog import ProjectDialog

from .error_messages import friendly_fit_message, friendly_io_message
from .tune_canvas import TuneCanvasWidget

logger = logging.getLogger(__name__)

# The r_f slider has this many ticks across the data's frequency span, so its
# precision is (freq span) / _RF_TICKS independent of the span's width.
_RF_TICKS = 300

# The g slider spans a fixed 0..200 MHz in 1 MHz ticks (default 50 MHz).
_G_MIN_MHZ = 0
_G_MAX_MHZ = 200
_G_DEFAULT_MHZ = 50

# Sample-dot recompute is debounced this long after the last g / r_f slider move
# (the line itself still moves live on every tick).
_SLIDER_DEBOUNCE_MS = 150


@dataclass(frozen=True)
class _TuneParams:
    g: float
    bare_rf: float


@dataclass(frozen=True)
class _TuneData:
    rf_0: np.ndarray
    rf_1: np.ndarray
    t_fluxs: np.ndarray
    g: float
    bare_rf: float


@dataclass(frozen=True)
class _AutoTuneParams:
    sample_fluxs: np.ndarray
    g0: float  # GHz
    bare_rf0: float  # GHz
    g_bounds: tuple[float, float]  # GHz
    rf_bounds: tuple[float, float]  # GHz


@dataclass(frozen=True)
class _AutoTuneResult:
    g: float  # GHz
    bare_rf: float  # GHz


class PipelinePanelWidget(QWidget):
    """The dispersive single-flow analysis panel."""

    def __init__(self, ctrl: Controller, parent: QWidget | None = None) -> None:
        from qtpy.QtCore import QTimer  # type: ignore[attr-defined]

        super().__init__(parent)
        self._ctrl = ctrl
        self._runner = BackgroundRunner(self)
        self._tune_artists = None
        # Points at whichever step's busy bar is currently active (preprocess /
        # predict). Both run a black-box compute, so the bars are indeterminate.
        self._active_progress: QProgressBar | None = None

        # Debounce the sample-dot recompute while a g / r_f slider is being dragged:
        # the line moves live on every tick, but the (heavier) per-flux prediction
        # only fires _SLIDER_DEBOUNCE_MS after the last move (when the user pauses).
        self._dot_debounce = QTimer(self)
        self._dot_debounce.setSingleShot(True)
        self._dot_debounce.setInterval(_SLIDER_DEBOUNCE_MS)
        self._dot_debounce.timeout.connect(self._on_dot_debounce_fired)

        self._build_ui()
        self._sync_enabled()

    # --- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        # Compact control rows at the top: steps 1/2/3 side by side, step 4 on its
        # own row (it has more controls, and now owns Export too). The plots go in
        # the shared tabbed area below, so the figures get the full space.
        top_row = QHBoxLayout()
        top_row.addWidget(self._build_inputs_section(), stretch=1)
        top_row.addWidget(self._build_load_section(), stretch=1)
        top_row.addWidget(self._build_preprocess_section(), stretch=1)
        root.addLayout(top_row)

        # Shared info bar between the step 1/2/3 row and step 4 (keeps those boxes
        # short by moving their loaded-state text here).
        root.addWidget(self._build_info_row())
        root.addWidget(self._build_tune_section())
        root.addWidget(self._build_figure_tabs(), stretch=1)

    def _build_inputs_section(self) -> QWidget:
        # Buttons only — the loaded-inputs text shows in the shared info row below
        # (built by _build_info_row), so this box stays short.
        box = QGroupBox("1. Project & fit inputs")
        layout = QVBoxLayout(box)
        row = QHBoxLayout()
        project_btn = QPushButton("Project…")
        project_btn.clicked.connect(self._on_project)
        load_btn = QPushButton("Load params.json")
        load_btn.clicked.connect(self._on_load_inputs)
        row.addWidget(project_btn)
        row.addWidget(load_btn)
        row.addStretch(1)
        layout.addLayout(row)
        return box

    def _build_load_section(self) -> QWidget:
        # Button only — the loaded one-tone status shows in the shared info row below.
        self._load_box = QGroupBox("2. Load one-tone spectrum")
        layout = QVBoxLayout(self._load_box)
        row = QHBoxLayout()
        # The load dialog carries the file picker + transpose toggle + live preview,
        # so the user can judge the axis orientation before loading.
        self._browse_btn = QPushButton("Load one-tone…")
        self._browse_btn.clicked.connect(self._on_load_onetone)
        row.addWidget(self._browse_btn)
        row.addStretch(1)
        layout.addLayout(row)
        return self._load_box

    def _build_preprocess_section(self) -> QWidget:
        self._preprocess_box = QGroupBox("3. Preprocess")
        layout = QVBoxLayout(self._preprocess_box)
        self._preprocess_btn = QPushButton("Run preprocessing")
        self._preprocess_btn.clicked.connect(self._on_preprocess)
        layout.addWidget(self._preprocess_btn)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        layout.addStretch(1)
        return self._preprocess_box

    def _build_info_row(self) -> QWidget:
        """A shared single-line info bar (between steps 1/2/3 and step 4) that shows
        the loaded fit inputs and one-tone status, so the step boxes above stay short."""
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 0, 4, 0)
        self._inputs_label = QLabel("Load the fluxonium fit (fluxdep_fit) first.")
        self._inputs_label.setWordWrap(True)
        self._onetone_label = QLabel("(no one-tone loaded)")
        layout.addWidget(self._inputs_label, stretch=3)
        layout.addWidget(self._onetone_label, stretch=1)
        return bar

    def _build_tune_section(self) -> QWidget:
        # The tune figure lives in the shared tab area below. Dragging a slider moves
        # its line + refreshes the sample dots live (cheap single-point predicts); only
        # "Use these g/r_f" runs the full off-main predictor, draws the dispersion
        # lines, and records the result (the manual tuning IS the final fit). Export
        # (writing the dispersive section to params.json) lives at the bottom-right of
        # this section — it enables once a tuning has been accepted.
        self._tune_box = QGroupBox("4. Tune g / r_f  (drag to match, accept, export)")
        outer = QVBoxLayout(self._tune_box)

        # Two sliders stacked vertically at the top: g, then r_f.
        self._g_slider, self._g_label = self._build_g_slider()
        outer.addLayout(self._slider_row("g (MHz)", self._g_slider, self._g_label))
        self._rf_slider, self._rf_label = self._build_rf_slider()
        outer.addLayout(self._slider_row("r_f (MHz)", self._rf_slider, self._rf_label))

        # Bottom row: action buttons on the left, Export pushed to the right.
        bottom = QHBoxLayout()
        self._add_sample_btn = QPushButton("Add sample flux")
        self._add_sample_btn.clicked.connect(self._on_add_sample)
        self._clear_samples_btn = QPushButton("Clear samples")
        self._clear_samples_btn.clicked.connect(self._on_clear_samples)
        # Auto tune optimises g / r_f against the sample lines; only meaningful once
        # at least one sample line exists, so it is enabled/disabled in _sync_enabled.
        self._auto_tune_btn = QPushButton("Auto tune")
        self._auto_tune_btn.clicked.connect(self._on_auto_tune)
        self._tune_btn = QPushButton("Use these g / r_f")
        self._tune_btn.clicked.connect(self._on_tune)
        bottom.addWidget(self._add_sample_btn)
        bottom.addWidget(self._clear_samples_btn)
        bottom.addWidget(self._auto_tune_btn)
        bottom.addWidget(self._tune_btn)
        bottom.addStretch(1)
        self._export_label = QLabel("")
        bottom.addWidget(self._export_label)
        self._export_btn = QPushButton("Export to params.json")
        self._export_btn.clicked.connect(self._on_export)
        bottom.addWidget(self._export_btn)
        outer.addLayout(bottom)

        self._tune_progress = QProgressBar()
        self._tune_progress.setVisible(False)
        outer.addWidget(self._tune_progress)
        return self._tune_box

    def _slider_row(self, name: str, slider: QWidget, value_label: QLabel):
        """A horizontal row: a fixed-width name label, the slider, and its value."""
        row = QHBoxLayout()
        label = QLabel(name)
        label.setFixedWidth(60)
        row.addWidget(label)
        row.addWidget(slider, stretch=1)
        value_label.setFixedWidth(80)
        row.addWidget(value_label)
        return row

    def _build_g_slider(self):
        """The g slider: 0..200 MHz in 1 MHz ticks (default 50). Live-refreshes dots."""
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]
        from qtpy.QtWidgets import QSlider  # type: ignore[attr-defined]

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(_G_MIN_MHZ, _G_MAX_MHZ)  # 1 tick = 1 MHz
        slider.setValue(_G_DEFAULT_MHZ)
        slider.valueChanged.connect(self._on_g_slider)
        return slider, QLabel(f"{_G_DEFAULT_MHZ}.0 MHz")

    def _build_rf_slider(self):
        """The r_f slider: a FIXED 0..RF_TICKS tick range across the data's freq span,
        so its precision is always (freq span)/RF_TICKS. The GHz value is mapped from
        the tick (see _rf_ghz); the span/default are set in _init_tune_view."""
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]
        from qtpy.QtWidgets import QSlider  # type: ignore[attr-defined]

        self._rf_lo_ghz = 5.0  # data freq span, set in _init_tune_view
        self._rf_hi_ghz = 6.0
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, _RF_TICKS)
        slider.setValue(_RF_TICKS // 2)
        slider.valueChanged.connect(self._on_rf_slider)
        return slider, QLabel("")

    def _build_figure_tabs(self) -> QWidget:
        """The shared tabbed figure area: Preprocess / Tune."""
        from qtpy.QtWidgets import QTabWidget  # type: ignore[attr-defined]

        self._tabs = QTabWidget()

        # Preprocess diagnostic (3-panel, vertical).
        self._diag_figure = Figure(figsize=(6, 8))
        self._diag_canvas = FigureCanvasQTAgg(self._diag_figure)
        self._tabs.addTab(self._diag_canvas, "Preprocess")

        # Tune figure (g/r_f lines over the norm-phase image — the final result).
        self._tune_canvas = TuneCanvasWidget()
        self._tabs.addTab(self._tune_canvas, "Tune")
        return self._tabs

    # --- enable/disable gating ------------------------------------------

    def _sync_enabled(self) -> None:
        st = self._ctrl.state
        has_inputs = st.fit_inputs is not None
        has_onetone = st.onetone is not None
        has_preprocess = st.preprocess is not None
        has_result = st.disp_fit.has_result
        self._load_box.setEnabled(has_inputs)
        self._preprocess_box.setEnabled(has_onetone)
        self._tune_box.setEnabled(has_preprocess)
        # Export (now inside step 4) only enables once a tuning has been accepted.
        self._export_btn.setEnabled(has_result)
        self._sync_auto_tune_enabled()

    def _sync_auto_tune_enabled(self) -> None:
        """Auto tune needs sample-flux lines to define its objective; enable it only
        when at least one exists (sample lines live in the artists, not State, so this
        is called on add/clear too, not just on _sync_enabled)."""
        has_samples = (
            self._tune_artists is not None and len(self._tune_artists.samples) > 0
        )
        self._auto_tune_btn.setEnabled(has_samples)

    # --- section 1: inputs ----------------------------------------------

    def _on_project(self) -> None:
        # Non-blocking dialog: exec() would freeze the Qt event loop and stall the
        # read-only control socket, so open() the dialog and act on its result via
        # the ``accepted`` signal (it only fires on Accept; Reject runs nothing).
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]

        dialog = ProjectDialog(
            self._ctrl.state.project,
            self,
            db_label="One-tone dir",
            project_root=self._ctrl.get_project_root(),
        )
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_accepted() -> None:
            self._ctrl.setup_project(dialog.result_project())

        dialog.accepted.connect(_on_accepted)
        self._project_dialog = dialog  # hold a reference so it is not GC'd while open
        dialog.finished.connect(
            lambda _result=0: setattr(self, "_project_dialog", None)
        )
        dialog.open()

    def _on_load_inputs(self) -> None:
        from .paths import params_dir

        start = params_dir(self._ctrl.state.project)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select params.json", start, filter="JSON (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            self._ctrl.load_fit_inputs(path)
        except Exception as exc:  # noqa: BLE001 — surface to the panel
            self._warn("Load inputs failed", friendly_io_message("Load", path, exc))
            return
        inputs = self._ctrl.state.fit_inputs
        assert inputs is not None
        ej, ec, el = inputs.params
        self._inputs_label.setText(
            f"EJ/EC/EL = {ej:.3f}/{ec:.3f}/{el:.3f} GHz · "
            f"flux_half={inputs.flux_half:.4g} period={inputs.flux_period:.4g} · "
            f"bare_rf seed={inputs.bare_rf_seed:.4f} GHz"
        )
        self._sync_enabled()

    # --- section 2: load onetone ----------------------------------------

    def _on_load_onetone(self) -> None:
        # Non-blocking dialog: exec() would freeze the Qt event loop and stall the
        # read-only control socket, so open() the dialog and act on its result via
        # the ``accepted`` signal (it only fires on Accept; Reject runs nothing).
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]

        from .load_dialog import LoadOnetoneDialog
        from .paths import raw_onetone_dir

        start = raw_onetone_dir(self._ctrl.state.project)
        dialog = LoadOnetoneDialog(self, start_dir=start)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        def _on_accepted() -> None:
            req = dialog.result_request()
            if req is None:
                return
            try:
                name = self._ctrl.load_onetone(
                    req.filepath, transpose_axes=req.transpose_axes
                )
            except Exception as exc:  # noqa: BLE001 — surface to the panel
                self._warn(
                    "Load one-tone failed",
                    friendly_io_message("Load", req.filepath, exc),
                )
                return
            self._onetone_label.setText(f"Loaded: {name}")
            self._sync_enabled()

        dialog.accepted.connect(_on_accepted)
        self._onetone_dialog = dialog  # hold a reference so it is not GC'd while open
        dialog.finished.connect(
            lambda _result=0: setattr(self, "_onetone_dialog", None)
        )
        dialog.open()

    # --- section 3: preprocess ------------------------------------------

    def _on_preprocess(self) -> None:
        self._preprocess_btn.setEnabled(False)
        self._begin_progress(
            self._progress
        )  # busy/indeterminate (numba is a black box)
        # Pure ``compute_preprocess`` off-main (numba edelay kernel, a single
        # black-box call); no routing/pbar scope, so enter=None.
        self._runner.submit(
            self._ctrl.compute_preprocess,
            on_done=self._on_preprocess_done,
            on_error=self._on_preprocess_failed,
            run_in_pool=True,
        )

    def _on_preprocess_done(self, result: PreprocessResult) -> None:
        self._ctrl.record_preprocess(result)
        self._preprocess_btn.setEnabled(True)
        self._end_progress()
        self._render_diagnostic(result)
        self._init_tune_view(result)
        self._show_tab(self._diag_canvas)
        self._sync_enabled()

    def _init_tune_view(self, result: PreprocessResult) -> None:
        """Set up the Tune tab from a fresh preprocessing result.

        The r_f slider spans the data's frequency range over a fixed _RF_TICKS ticks
        (so its precision is span/_RF_TICKS); its default tick is the one nearest the
        median peak frequency. The tune figure shows the norm-phase background + the
        r_f line right away (no dispersion lines until "Use these g/r_f" predicts).
        """
        self._rf_lo_ghz = float(result.sp_freqs.min())
        self._rf_hi_ghz = float(result.sp_freqs.max())
        self._rf_slider.blockSignals(True)
        self._rf_slider.setValue(self._rf_tick_for(result.median_rf))
        self._rf_slider.blockSignals(False)
        self._rf_label.setText(f"{1e3 * self._rf_ghz():.1f} MHz")
        self._tune_artists = render_tune_figure(
            self._tune_canvas.figure,
            result.sp_fluxs,
            result.sp_freqs,
            result.norm_phases,
            self._rf_ghz(),
        )
        # A fresh background drops any prior sample lines; re-arm drag on the new
        # artists so dragging targets them (move on drag, recompute on drop).
        self._tune_canvas.bind_drag(
            self._tune_artists, self._on_sample_drag, self._on_sample_drop
        )
        self._tune_canvas.redraw()

    def _rf_tick_for(self, rf_ghz: float) -> int:
        """The slider tick nearest a GHz r_f value (clamped into range)."""
        span = self._rf_hi_ghz - self._rf_lo_ghz
        if span <= 0.0:
            return 0
        frac = (rf_ghz - self._rf_lo_ghz) / span
        return min(max(int(round(frac * _RF_TICKS)), 0), _RF_TICKS)

    def _on_preprocess_failed(self, exc: Exception) -> None:
        logger.exception("preprocess worker failed", exc_info=exc)
        self._preprocess_btn.setEnabled(True)
        self._end_progress()
        self._warn("Preprocess failed", friendly_fit_message("Preprocess", exc))

    def _render_diagnostic(self, result: PreprocessResult) -> None:
        """The 3-panel diagnostic (signal magnitude / edelay / norm phases)."""
        st = self._ctrl.state
        assert st.onetone is not None
        raw = st.onetone.raw
        fig = self._diag_figure
        fig.clear()
        # add_subplot (vs subplots) gives a precise Axes type to pyright, avoiding
        # the subplots() overload's "not iterable" unpacking error. Stacked
        # vertically so the three share the flux x-axis at a glance.
        ax1 = fig.add_subplot(3, 1, 1)
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
        extent = (
            float(result.sp_fluxs[0]),
            float(result.sp_fluxs[-1]),
            float(result.sp_freqs[0]),
            float(result.sp_freqs[-1]),
        )
        ax1.set_title("Signal magnitude")
        ax1.imshow(
            np.abs(raw["signals"]).T,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="none",
        )
        ax2.set_title("Fitted edelay")
        ax2.plot(result.sp_fluxs, result.edelays)
        ax2.axhline(result.edelay, color="k", linestyle="--")
        ax2.grid(True)
        ax3.set_title("Normalized phases")
        ax3.imshow(
            result.norm_phases.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="none",
        )
        fig.tight_layout()
        self._diag_canvas.draw_idle()

    # --- section 4: tune (r_f slider live; predict on "Use these g/r_f") -

    def _rf_ghz(self) -> float:
        """The r_f slider value in GHz (tick mapped into the data's freq span)."""
        frac = self._rf_slider.value() / _RF_TICKS
        return self._rf_lo_ghz + frac * (self._rf_hi_ghz - self._rf_lo_ghz)

    def _g_mhz(self) -> float:
        """The g slider value in MHz (1 tick = 1 MHz)."""
        return float(self._g_slider.value())

    def _on_rf_slider(self, _: int) -> None:
        """Live: move the r_f line as the slider drags; debounce the dot recompute."""
        rf_ghz = self._rf_ghz()
        self._rf_label.setText(f"{1e3 * rf_ghz:.1f} MHz")
        if self._tune_artists is not None:
            update_bare_line(self._tune_artists, rf_ghz)  # cheap, every tick
            self._tune_canvas.redraw()
            self._dot_debounce.start()  # recompute dots only when the user pauses

    def _on_g_slider(self, value: int) -> None:
        """Live: update the g label; debounce the (heavier) sample-dot recompute."""
        self._g_label.setText(f"{value}.0 MHz")
        if self._tune_artists is not None:
            self._dot_debounce.start()

    def _on_dot_debounce_fired(self) -> None:
        """Recompute all sample dots once the g / r_f slider has settled."""
        if self._tune_artists is not None:
            self._refresh_sample_dots()
            self._tune_canvas.redraw()

    # --- sample-flux lines (draggable; live single-point dots) -----------

    def _on_add_sample(self) -> None:
        """Drop a new sample-flux line at the center of the flux axis + its dots."""
        if self._tune_artists is None or self._ctrl.state.preprocess is None:
            return
        sp_fluxs = self._ctrl.state.preprocess.sp_fluxs
        flux = float(0.5 * (sp_fluxs[0] + sp_fluxs[-1]))
        add_sample_line(self._tune_artists, flux)
        self._refresh_sample_dots()
        self._tune_canvas.redraw()
        self._sync_auto_tune_enabled()  # first sample enables Auto tune

    def _on_clear_samples(self) -> None:
        """Remove every sample line + its dots in place (keeps the dispersion lines)."""
        if self._tune_artists is None:
            return
        for sample in list(self._tune_artists.samples):
            remove_sample_line(self._tune_artists, sample)
        self._tune_canvas.redraw()
        self._sync_auto_tune_enabled()  # no samples left disables Auto tune

    def _on_sample_drag(self, sample: SampleArtists, flux: float) -> None:
        """Motion callback: move the grabbed line VISUALLY only (no compute while
        dragging — the dot is recomputed on release in ``_on_sample_drop``)."""
        if self._tune_artists is None or self._ctrl.state.preprocess is None:
            return
        sp_fluxs = self._ctrl.state.preprocess.sp_fluxs
        flux = float(np.clip(flux, sp_fluxs[0], sp_fluxs[-1]))
        move_sample_line(sample, flux)
        self._tune_canvas.redraw()

    def _on_sample_drop(self, sample: SampleArtists) -> None:
        """Release callback: recompute the dropped line's dot now the drag has ended."""
        if self._tune_artists is None or self._ctrl.state.preprocess is None:
            return
        self._compute_sample_dots([sample])
        self._tune_canvas.redraw()

    def _refresh_sample_dots(self) -> None:
        """Recompute every sample line's ground/excited dots for the current g/r_f."""
        if self._tune_artists is not None and self._tune_artists.samples:
            self._compute_sample_dots(self._tune_artists.samples)

    def _compute_sample_dots(self, samples: list[SampleArtists]) -> None:
        """Predict + set the dots for the given sample lines (current g / r_f slider).

        A single batched single-point predict over the sample fluxs (a few ms). The
        compute reads State (fit_inputs.params) but does not write it, so it is safe
        on the main thread. Failures (e.g. labeling fallback raising) are swallowed
        with a warning — a bad sample dot must not break the tuning UI.
        """
        if self._tune_artists is None or not samples:
            return
        g = 1e-3 * self._g_mhz()
        bare_rf = self._rf_ghz()
        fluxs = np.array([s.flux for s in samples], dtype=np.float64)
        try:
            rf_0, rf_1 = self._ctrl.predict_sample_points(fluxs, g, bare_rf)
        except Exception:  # noqa: BLE001 — a sample dot must not break tuning
            logger.exception("sample-point prediction failed")
            return
        for sample, r0, r1 in zip(samples, rf_0, rf_1):
            update_sample_dots(self._tune_artists, sample, float(r0), float(r1))

    def _on_tune(self) -> None:
        """Run the predictor off-main for the current g + r_f slider, draw + record.

        The manual tuning IS the final fit: the done slot records the accepted
        g/bare_rf and draws the dispersion lines. The button is disabled while the
        prediction is in flight, so a new run cannot start until it finishes.
        """
        if self._ctrl.state.preprocess is None:
            return
        params = _TuneParams(
            g=1e-3 * self._g_mhz(),
            bare_rf=self._rf_ghz(),
        )
        self._tune_btn.setEnabled(False)
        self._begin_progress(self._tune_progress)
        ctrl = self._ctrl

        def _predict() -> _TuneData:
            # The LRU-cached predictor reads State but does not write it, so it is
            # safe off-main; no routing/pbar scope, so enter=None.
            rf_0, rf_1 = ctrl.predict_dispersive(params.g, params.bare_rf, return_dim=2)
            t_fluxs = ctrl.predict_flux_axis()
            return _TuneData(rf_0, rf_1, t_fluxs, params.g, params.bare_rf)

        self._runner.submit(
            _predict,
            on_done=self._on_tune_done,
            on_error=self._on_tune_failed,
            run_in_pool=True,
        )

    def _on_tune_done(self, data: _TuneData) -> None:
        self._tune_btn.setEnabled(True)
        self._end_progress()
        if self._ctrl.state.preprocess is None or self._tune_artists is None:
            return
        # The accepted tuning IS the result: record g/bare_rf, then draw the
        # dispersion lines over the (already-drawn) background + r_f line.
        self._ctrl.set_manual_fit(data.g, data.bare_rf)
        set_dispersion_lines(
            self._tune_artists,
            data.t_fluxs,
            data.rf_0,
            data.rf_1,
            data.g,
            data.bare_rf,
        )
        self._tune_canvas.redraw()
        self._show_tab(self._tune_canvas)
        self._sync_enabled()

    def _on_tune_failed(self, exc: Exception) -> None:
        logger.exception("predict worker failed", exc_info=exc)
        self._tune_btn.setEnabled(True)
        self._end_progress()
        self._warn("Tune failed", friendly_fit_message("Tune", exc))

    # --- auto tune (scipy optimiser over the sample lines) ---------------

    def _on_auto_tune(self) -> None:
        """Optimise g / r_f against the sample lines off-main; write back the sliders.

        The slow iterative scipy search runs on a worker; the done slot sets the g/r_f
        sliders to the result and refreshes the sample dots (it does NOT accept the
        fit — the user still presses "Use these g/r_f"). The button is disabled while
        running so only one optimisation is in flight.
        """
        if self._tune_artists is None or self._ctrl.state.preprocess is None:
            return
        sample_fluxs = np.array(
            [s.flux for s in self._tune_artists.samples], dtype=np.float64
        )
        if sample_fluxs.size == 0:
            return
        params = _AutoTuneParams(
            sample_fluxs=sample_fluxs,
            g0=1e-3 * self._g_mhz(),
            bare_rf0=self._rf_ghz(),
            g_bounds=(1e-3 * _G_MIN_MHZ, 1e-3 * _G_MAX_MHZ),
            rf_bounds=(self._rf_lo_ghz, self._rf_hi_ghz),
        )
        self._auto_tune_btn.setEnabled(False)
        self._begin_progress(self._tune_progress)
        ctrl = self._ctrl

        def _optimise() -> _AutoTuneResult:
            # The slow iterative scipy search reads State but does not write it, so
            # it is safe off-main; no routing/pbar scope, so enter=None.
            g, bare_rf = ctrl.auto_tune(
                params.sample_fluxs,
                params.g0,
                params.bare_rf0,
                params.g_bounds,
                params.rf_bounds,
            )
            return _AutoTuneResult(g, bare_rf)

        self._runner.submit(
            _optimise,
            on_done=self._on_auto_tune_done,
            on_error=self._on_auto_tune_failed,
            run_in_pool=True,
        )

    def _on_auto_tune_done(self, result: _AutoTuneResult) -> None:
        self._end_progress()
        self._sync_auto_tune_enabled()  # re-enable (if samples still present)
        # Write the optimised values back into the sliders (their valueChanged slots
        # move the lines + refresh the sample dots). Does NOT accept the fit.
        self._g_slider.setValue(int(round(1e3 * result.g)))
        self._rf_slider.setValue(self._rf_tick_for(result.bare_rf))

    def _on_auto_tune_failed(self, exc: Exception) -> None:
        logger.exception("auto-tune worker failed", exc_info=exc)
        self._end_progress()
        self._sync_auto_tune_enabled()
        self._warn("Auto tune failed", friendly_fit_message("Auto tune", exc))

    # --- tabs + export --------------------------------------------------

    def _show_tab(self, widget: QWidget) -> None:
        idx = self._tabs.indexOf(widget)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def _on_export(self) -> None:
        try:
            path = self._ctrl.export_params()
        except Exception as exc:  # noqa: BLE001
            self._warn(
                "Export failed", friendly_io_message("Export", "params.json", exc)
            )
            return
        self._export_label.setText(f"Exported → {os.path.basename(path)}")

    # --- progress + messages --------------------------------------------

    def _begin_progress(self, bar: QProgressBar) -> None:
        """Show ``bar`` as a busy/indeterminate spinner (the compute is a black box)."""
        self._active_progress = bar
        bar.setVisible(True)
        bar.setRange(0, 0)

    def _end_progress(self) -> None:
        """Hide the active progress bar."""
        if self._active_progress is not None:
            self._active_progress.setVisible(False)
        self._active_progress = None

    def _warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

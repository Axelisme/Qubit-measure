"""PipelinePanelWidget — the dispersive single-flow analysis panel.

A vertical stack of sections, each enabled only once the prior step completes (a
linear pipeline, fast-fail UX — no skipping):

1. **Project & inputs** — load the fluxonium fit from params.json (the hard gate).
2. **Load one-tone** — browse + transpose a raw one-tone hdf5.
3. **Preprocess** — run the (off-main) signal pipeline + a 3-panel diagnostic.
4. **Tune g / r_f** — sliders drive the LRU-cached predictor; the display-only
   canvas redraws in place on the main thread.
5. **Auto-fit** — the (off-main) scipy fit, recorded on the main thread.
6. **Result** — the two product figures (dispersive-with-onetone + chi-shift).
7. **Export** — write the dispersive section back to params.json.

The two heavy steps (preprocess, auto-fit) run on a ``QThreadPool`` worker that calls
only the pure ``compute_*`` controller methods; the worker's ``done`` slot — on the
Qt main thread — is the sole caller of ``record_*`` (the State writers), upholding
the main-thread State invariant. The tuning / result figures are drawn on local
``FigureCanvasQTAgg`` widgets on the main thread (no routed backend, no marshalling).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import (  # type: ignore[attr-defined]
    QObject,
    QRunnable,
    QThreadPool,
    Signal,  # type: ignore[attr-defined]
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.app.dispersive.controller import Controller
from zcu_tools.gui.app.dispersive.services.fit import AutoFitResult
from zcu_tools.gui.app.dispersive.services.viz import (
    render_dispersive_shift,
    render_dispersive_with_onetone,
    render_tune_figure,
    update_tune_lines,
)
from zcu_tools.gui.app.dispersive.state import PreprocessResult

from .error_messages import friendly_fit_message, friendly_io_message
from .gui_pbar import GuiProgressBarChannel
from .project_dialog import ProjectDialog
from .tune_canvas import TuneCanvasWidget

logger = logging.getLogger(__name__)

_N_TUNE_FLUX = 501  # result-figure flux resolution (notebook cell 12)


class _WorkerSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _PreprocessWorker(QRunnable):
    """Runs the pure ``compute_preprocess`` off the main thread (joblib edelay fit)."""

    def __init__(self, ctrl: Controller, pbar_factory) -> None:
        super().__init__()
        self.signals = _WorkerSignals()
        self._ctrl = ctrl
        self._pbar_factory = pbar_factory

    def run(self) -> None:
        try:
            from zcu_tools.progress_bar import use_pbar_factory

            with use_pbar_factory(self._pbar_factory):
                result = self._ctrl.compute_preprocess()
        except Exception as exc:  # noqa: BLE001 — surface to the panel
            logger.exception("preprocess worker failed")
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(result)


class _AutoFitWorker(QRunnable):
    """Runs the pure ``compute_autofit`` off the main thread (scipy + scqubits)."""

    def __init__(self, ctrl: Controller, pbar_factory) -> None:
        super().__init__()
        self.signals = _WorkerSignals()
        self._ctrl = ctrl
        self._pbar_factory = pbar_factory

    def run(self) -> None:
        try:
            result = self._ctrl.compute_autofit(pbar_factory=self._pbar_factory)
        except Exception as exc:  # noqa: BLE001 — surface to the panel
            logger.exception("auto-fit worker failed")
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(result)


class PipelinePanelWidget(QWidget):
    """The dispersive single-flow analysis panel."""

    def __init__(self, ctrl: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._pool = QThreadPool.globalInstance() or QThreadPool(self)
        self._channel = GuiProgressBarChannel()
        self._channel.progress.connect(self._on_progress)
        self._tune_artists = None
        # The progress signal is shared by both workers (only one runs at a time);
        # this points at the bar of whichever worker is currently active, so the
        # preprocess (step 3) and auto-fit (step 5) bars stay independent.
        self._active_progress: Optional[QProgressBar] = None

        self._build_ui()
        self._sync_enabled()

    # --- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        # Steps 1 & 2 are small (a couple of buttons + a label each), so place them
        # side by side instead of two full-width rows.
        top_row = QHBoxLayout()
        top_row.addWidget(self._build_inputs_section(), stretch=1)
        top_row.addWidget(self._build_load_section(), stretch=1)
        root.addLayout(top_row)
        root.addWidget(self._build_preprocess_section())
        root.addWidget(self._build_tune_section(), stretch=1)
        root.addWidget(self._build_autofit_section())
        root.addWidget(self._build_result_section(), stretch=1)
        root.addWidget(self._build_export_section())

    def _build_inputs_section(self) -> QWidget:
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
        self._inputs_label = QLabel("Load the fluxonium fit (fluxdep_fit) first.")
        self._inputs_label.setWordWrap(True)
        layout.addWidget(self._inputs_label)
        return box

    def _build_load_section(self) -> QWidget:
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
        self._onetone_label = QLabel("(no one-tone loaded)")
        layout.addWidget(self._onetone_label)
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
        self._diag_figure = Figure(figsize=(5, 9))
        self._diag_canvas = FigureCanvasQTAgg(self._diag_figure)
        layout.addWidget(self._diag_canvas)
        return self._preprocess_box

    def _build_tune_section(self) -> QWidget:
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]
        from qtpy.QtWidgets import QSplitter  # type: ignore[attr-defined]

        self._tune_box = QGroupBox("4. Tune g / r_f")
        outer = QVBoxLayout(self._tune_box)

        controls = QGroupBox("Parameters")
        form = QFormLayout(controls)
        self._g_spin = self._mhz_spin(60.0, 0.0, 200.0)
        self._rf_spin = self._mhz_spin(5300.0, 1000.0, 12000.0)
        self._g_spin.valueChanged.connect(self._on_tune_changed)
        self._rf_spin.valueChanged.connect(self._on_tune_changed)
        form.addRow("g (MHz)", self._g_spin)
        form.addRow("r_f (MHz)", self._rf_spin)
        self._qub_dim = self._int_spin(15)
        self._qub_cutoff = self._int_spin(30)
        self._res_dim = self._int_spin(4)
        self._step = self._int_spin(1)
        for spin in (self._qub_dim, self._qub_cutoff, self._res_dim, self._step):
            spin.valueChanged.connect(self._on_tune_changed)
        form.addRow("qub_dim", self._qub_dim)
        form.addRow("qub_cutoff", self._qub_cutoff)
        form.addRow("res_dim", self._res_dim)
        form.addRow("step", self._step)
        finish = QPushButton("Use these g / r_f")
        finish.clicked.connect(self._on_finish_tune)
        form.addRow(finish)
        controls.setMaximumWidth(320)

        self._tune_canvas = TuneCanvasWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(self._tune_canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter)
        return self._tune_box

    def _build_autofit_section(self) -> QWidget:
        self._autofit_box = QGroupBox("5. Auto-fit g")
        layout = QVBoxLayout(self._autofit_box)
        row = QHBoxLayout()
        self._fit_bare_rf = QCheckBox("also fit bare_rf (±2 MHz)")
        self._g_lo = self._mhz_spin(0.0, 0.0, 200.0)
        self._g_hi = self._mhz_spin(200.0, 0.0, 200.0)
        row.addWidget(QLabel("g bound (MHz)"))
        row.addWidget(self._g_lo)
        row.addWidget(QLabel("–"))
        row.addWidget(self._g_hi)
        row.addWidget(self._fit_bare_rf)
        row.addStretch(1)
        layout.addLayout(row)
        self._autofit_btn = QPushButton("Auto-fit g")
        self._autofit_btn.clicked.connect(self._on_autofit)
        layout.addWidget(self._autofit_btn)
        self._autofit_progress = QProgressBar()
        self._autofit_progress.setVisible(False)
        layout.addWidget(self._autofit_progress)
        self._autofit_label = QLabel("")
        layout.addWidget(self._autofit_label)
        return self._autofit_box

    def _build_result_section(self) -> QWidget:
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]
        from qtpy.QtWidgets import QSplitter  # type: ignore[attr-defined]

        self._result_box = QGroupBox("6. Result")
        outer = QVBoxLayout(self._result_box)
        redraw = QPushButton("Render result figures")
        redraw.clicked.connect(self._on_render_result)
        outer.addWidget(redraw)
        self._overlay_figure = Figure(figsize=(5, 4))
        self._overlay_canvas = FigureCanvasQTAgg(self._overlay_figure)
        self._chi_figure = Figure(figsize=(5, 4))
        self._chi_canvas = FigureCanvasQTAgg(self._chi_figure)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._overlay_canvas)
        splitter.addWidget(self._chi_canvas)
        outer.addWidget(splitter)
        return self._result_box

    def _build_export_section(self) -> QWidget:
        self._export_box = QGroupBox("7. Export")
        layout = QVBoxLayout(self._export_box)
        self._export_btn = QPushButton("Export to params.json")
        self._export_btn.clicked.connect(self._on_export)
        layout.addWidget(self._export_btn)
        self._export_label = QLabel("")
        layout.addWidget(self._export_label)
        return self._export_box

    # --- widget helpers --------------------------------------------------

    def _mhz_spin(self, value: float, lo: float, hi: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(1)
        box.setSingleStep(1.0)
        box.setKeyboardTracking(False)  # emit on commit, not per keystroke
        box.setValue(value)
        return box

    def _int_spin(self, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(1, 200)
        box.setKeyboardTracking(False)
        box.setValue(value)
        return box

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
        self._autofit_box.setEnabled(has_preprocess)
        self._result_box.setEnabled(has_result)
        self._export_box.setEnabled(has_result)

    # --- section 1: inputs ----------------------------------------------

    def _on_project(self) -> None:
        dialog = ProjectDialog(self._ctrl.state.project, self)
        if dialog.exec():
            self._ctrl.setup_project(dialog.result_project())

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
        self._rf_spin.setValue(1e3 * inputs.bare_rf_seed)
        self._sync_enabled()

    # --- section 2: load onetone ----------------------------------------

    def _on_load_onetone(self) -> None:
        from .load_dialog import LoadOnetoneDialog
        from .paths import raw_onetone_dir

        start = raw_onetone_dir(self._ctrl.state.project)
        dialog = LoadOnetoneDialog(self, start_dir=start)
        if not dialog.exec():
            return
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

    # --- section 3: preprocess ------------------------------------------

    def _on_preprocess(self) -> None:
        self._preprocess_btn.setEnabled(False)
        self._begin_progress(self._progress)
        worker = _PreprocessWorker(self._ctrl, self._channel.factory())
        worker.signals.done.connect(self._on_preprocess_done)
        worker.signals.failed.connect(self._on_preprocess_failed)
        self._pool.start(worker)

    def _on_preprocess_done(self, result: PreprocessResult) -> None:
        self._ctrl.record_preprocess(result)
        self._preprocess_btn.setEnabled(True)
        self._end_progress()
        self._render_diagnostic(result)
        self._sync_enabled()
        self._refresh_tune()

    def _on_preprocess_failed(self, message: str) -> None:
        self._preprocess_btn.setEnabled(True)
        self._end_progress()
        self._warn("Preprocess failed", friendly_fit_message("Preprocess", message))

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

    # --- section 4: tune ------------------------------------------------

    def _refresh_tune(self) -> None:
        """(Re)build the tuning figure from the current sliders + preprocess."""
        st = self._ctrl.state
        if st.preprocess is None:
            return
        pred = self._predict_tune()
        if pred is None:
            return
        rf_0, rf_1, t_fluxs, g, bare_rf = pred
        self._tune_artists = render_tune_figure(
            self._tune_canvas.figure,
            st.preprocess.sp_fluxs,
            st.preprocess.sp_freqs,
            st.preprocess.norm_phases,
            t_fluxs,
            rf_0,
            rf_1,
            g,
            bare_rf,
        )
        self._tune_canvas.redraw()

    def _on_tune_changed(self, *_args) -> None:
        if self._ctrl.state.preprocess is None:
            return
        pred = self._predict_tune()
        if pred is None:
            return
        if self._tune_artists is None:
            self._refresh_tune()
            return
        rf_0, rf_1, t_fluxs, g, bare_rf = pred
        update_tune_lines(self._tune_artists, t_fluxs, rf_0, rf_1, g, bare_rf)
        self._tune_canvas.redraw()

    def _predict_tune(self):
        """Run the LRU-cached predictor for the current slider values.

        Returns ``(rf_0, rf_1, t_fluxs, g_GHz, bare_rf_GHz)``, or ``None`` on a
        prediction error (surfaced to the label, not raised — slider moves must
        not crash the panel).
        """
        g = 1e-3 * self._g_spin.value()
        bare_rf = 1e-3 * self._rf_spin.value()
        step = self._step.value()
        try:
            rf_0, rf_1 = self._ctrl.predict_dispersive(
                g,
                bare_rf,
                qub_dim=self._qub_dim.value(),
                qub_cutoff=self._qub_cutoff.value(),
                res_dim=self._res_dim.value(),
                step=step,
                return_dim=2,
            )
            t_fluxs = self._ctrl.predict_flux_axis(step)
        except Exception as exc:  # noqa: BLE001 — keep the panel alive
            logger.exception("predict_dispersive failed")
            self._autofit_label.setText(f"Prediction error: {exc}")
            return None
        return rf_0, rf_1, t_fluxs, g, bare_rf

    def _on_finish_tune(self) -> None:
        g = 1e-3 * self._g_spin.value()
        bare_rf = 1e-3 * self._rf_spin.value()
        self._ctrl.set_manual_fit(g, bare_rf)
        self._autofit_label.setText(
            f"Using g = {1e3 * g:.1f} MHz, r_f = {1e3 * bare_rf:.1f} MHz"
        )
        self._sync_enabled()

    # --- section 5: auto-fit --------------------------------------------

    def _on_autofit(self) -> None:
        try:
            self._ctrl.set_disp_params(
                g_bound=(1e-3 * self._g_lo.value(), 1e-3 * self._g_hi.value()),
                fit_bare_rf=self._fit_bare_rf.isChecked(),
                qub_dim=self._qub_dim.value(),
                qub_cutoff=self._qub_cutoff.value(),
                res_dim=self._res_dim.value(),
                step=self._step.value(),
            )
        except Exception as exc:  # noqa: BLE001
            self._warn("Auto-fit failed", friendly_fit_message("Auto-fit", exc))
            return
        self._autofit_btn.setEnabled(False)
        self._begin_progress(self._autofit_progress)
        self._autofit_label.setText("Fitting…")
        worker = _AutoFitWorker(self._ctrl, self._channel.factory())
        worker.signals.done.connect(self._on_autofit_done)
        worker.signals.failed.connect(self._on_autofit_failed)
        self._pool.start(worker)

    def _on_autofit_done(self, result: AutoFitResult) -> None:
        self._ctrl.record_autofit_result(result)
        self._autofit_btn.setEnabled(True)
        self._end_progress()
        fit = self._ctrl.state.disp_fit
        assert fit.g is not None and fit.bare_rf is not None
        self._g_spin.setValue(1e3 * fit.g)
        self._rf_spin.setValue(1e3 * fit.bare_rf)
        self._autofit_label.setText(
            f"g = {1e3 * fit.g:.1f} MHz, r_f = {1e3 * fit.bare_rf:.1f} MHz"
        )
        self._sync_enabled()
        self._refresh_tune()

    def _on_autofit_failed(self, message: str) -> None:
        self._autofit_btn.setEnabled(True)
        self._end_progress()
        self._autofit_label.setText("Auto-fit failed.")
        self._warn("Auto-fit failed", friendly_fit_message("Auto-fit", message))

    # --- section 6: result ----------------------------------------------

    def _on_render_result(self) -> None:
        st = self._ctrl.state
        fit = st.disp_fit
        if not fit.has_result or st.preprocess is None or st.fit_inputs is None:
            return
        assert fit.g is not None and fit.bare_rf is not None
        pp = st.preprocess
        t_fluxs = np.linspace(
            float(pp.sp_fluxs.min()), float(pp.sp_fluxs.max()), _N_TUNE_FLUX
        ).astype(np.float64)
        try:
            plot_rfs = self._ctrl.predict_dispersive(
                fit.g, fit.bare_rf, res_dim=3, return_dim=3, step=1
            )
            # predict_dispersive returns over sp_fluxs[::step]; for the product
            # figure use the full axis directly via a fresh prediction grid.
            render_dispersive_with_onetone(
                self._overlay_figure,
                fit.bare_rf,
                fit.g,
                self._ctrl.predict_flux_axis(1),
                plot_rfs,
                pp.sp_fluxs,
                pp.sp_freqs,
                pp.norm_phases,
            )
            render_dispersive_shift(
                self._chi_figure,
                st.fit_inputs.params,
                t_fluxs,
                fit.bare_rf,
                fit.g,
            )
        except Exception as exc:  # noqa: BLE001
            self._warn("Render failed", friendly_fit_message("Render", exc))
            return
        self._overlay_canvas.draw_idle()
        self._chi_canvas.draw_idle()

    # --- section 7: export ----------------------------------------------

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
        """Show ``bar`` and route the shared progress signal to it."""
        self._active_progress = bar
        bar.setVisible(True)
        bar.setRange(0, 0)

    def _end_progress(self) -> None:
        """Hide the active progress bar and stop routing to it."""
        if self._active_progress is not None:
            self._active_progress.setVisible(False)
        self._active_progress = None

    def _on_progress(self, n: float, total: float, desc: str) -> None:
        bar = self._active_progress
        if bar is None:
            return
        if total > 0:
            bar.setRange(0, int(total))
            bar.setValue(int(n))
        else:
            bar.setRange(0, 0)

    def _warn(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

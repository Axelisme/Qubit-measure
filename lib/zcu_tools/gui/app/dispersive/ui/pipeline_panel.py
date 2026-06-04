"""PipelinePanelWidget — the dispersive single-flow analysis panel.

Layout: the step controls sit in compact rows at the top (steps 1/2/3 side by side,
step 4 on its own row), and a single tabbed figure area below holds the plots
(Preprocess diagnostic / Tune) so each gets the full width and height instead of
competing for vertical space.

The pipeline (each step enabled only once the prior completes — fast-fail UX):

1. **Project & inputs** — load the fluxonium fit from params.json (the hard gate).
2. **Load one-tone** — browse + transpose a raw one-tone hdf5.
3. **Preprocess** — run the (off-main) signal pipeline → the Preprocess tab.
4. **Tune g / r_f** — edit the spinboxes, then "Use these g/r_f" runs the predictor
   off-main, draws the Tune tab, and records the result (the manual tuning IS the
   final fit — there is no separate auto-fit). The button is disabled while computing.
5. **Export** — write the dispersive section back to params.json.

The heavy work (preprocess, predict) runs on a ``QThreadPool`` worker that calls only
the pure ``compute_*`` / ``predict_*`` controller methods and returns plain data; the
worker's ``done`` slot — on the Qt main thread — records State and draws the figure
(the worker never touches Qt widgets or pyplot, per ADR-0017). The figures live on
local ``FigureCanvasQTAgg`` widgets in the tabs.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
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
    QDoubleSpinBox,
    QFileDialog,
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
from zcu_tools.gui.app.dispersive.services.viz import render_tune_figure
from zcu_tools.gui.app.dispersive.state import PreprocessResult

from .error_messages import friendly_fit_message, friendly_io_message
from .gui_pbar import GuiProgressBarChannel
from .project_dialog import ProjectDialog
from .tune_canvas import TuneCanvasWidget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _TuneParams:
    g: float
    bare_rf: float
    res_dim: int
    step: int


@dataclass(frozen=True)
class _TuneData:
    rf_0: np.ndarray
    rf_1: np.ndarray
    t_fluxs: np.ndarray
    g: float
    bare_rf: float
    res_dim: int
    step: int


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


class _PredictWorker(QRunnable):
    """Runs the LRU-cached predictor off the main thread for the tuning figure.

    Returns ``_TuneData`` (the rf arrays + axis + the g/bare_rf used); the main-thread
    ``done`` slot draws it and records the result. The predictor reads State but does
    not write it, so it is safe off-main.
    """

    def __init__(self, ctrl: Controller, params: "_TuneParams") -> None:
        super().__init__()
        self.signals = _WorkerSignals()
        self._ctrl = ctrl
        self._p = params

    def run(self) -> None:
        try:
            p = self._p
            rf_0, rf_1 = self._ctrl.predict_dispersive(
                p.g, p.bare_rf, res_dim=p.res_dim, step=p.step, return_dim=2
            )
            t_fluxs = self._ctrl.predict_flux_axis(p.step)
        except Exception as exc:  # noqa: BLE001 — surface to the panel
            logger.exception("predict worker failed")
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(
            _TuneData(rf_0, rf_1, t_fluxs, p.g, p.bare_rf, p.res_dim, p.step)
        )


class PipelinePanelWidget(QWidget):
    """The dispersive single-flow analysis panel."""

    def __init__(self, ctrl: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._pool = QThreadPool.globalInstance() or QThreadPool(self)
        self._channel = GuiProgressBarChannel()
        self._channel.progress.connect(self._on_progress)
        self._tune_artists = None
        # The progress signal is shared by the workers (only one runs at a time);
        # this points at the bar of whichever worker is currently active.
        self._active_progress: Optional[QProgressBar] = None

        self._build_ui()
        self._sync_enabled()

    # --- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        # Compact control rows at the top: steps 1/2/3 side by side, step 4 on its
        # own row (it has more controls). The plots go in the shared tabbed area
        # below, so the figures get the full space.
        top_row = QHBoxLayout()
        top_row.addWidget(self._build_inputs_section(), stretch=1)
        top_row.addWidget(self._build_load_section(), stretch=1)
        top_row.addWidget(self._build_preprocess_section(), stretch=1)
        root.addLayout(top_row)

        root.addWidget(self._build_tune_section())
        root.addWidget(self._build_figure_tabs(), stretch=1)
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
        layout.addStretch(1)
        return self._preprocess_box

    def _build_tune_section(self) -> QWidget:
        # Controls only — the tune figure lives in the shared tab area. Editing a
        # spinbox does NOT compute; only "Use these g/r_f" runs the (off-main)
        # predictor, draws the figure, and records the result (the manual tuning IS
        # the final fit). The button is disabled while a prediction is in flight.
        self._tune_box = QGroupBox("4. Tune g / r_f  (drag to match, then accept)")
        outer = QVBoxLayout(self._tune_box)
        row = QHBoxLayout()
        self._g_spin = self._mhz_spin(60.0, 0.0, 200.0)
        self._rf_spin = self._mhz_spin(5300.0, 1000.0, 12000.0)
        self._res_dim = self._int_spin(4)
        self._step = self._int_spin(1)
        row.addWidget(QLabel("g (MHz)"))
        row.addWidget(self._g_spin)
        row.addWidget(QLabel("r_f (MHz)"))
        row.addWidget(self._rf_spin)
        row.addWidget(QLabel("res_dim"))
        row.addWidget(self._res_dim)
        row.addWidget(QLabel("step"))
        row.addWidget(self._step)
        self._tune_btn = QPushButton("Use these g / r_f")
        self._tune_btn.clicked.connect(self._on_tune)
        row.addWidget(self._tune_btn)
        row.addStretch(1)
        outer.addLayout(row)
        self._tune_progress = QProgressBar()
        self._tune_progress.setVisible(False)
        outer.addWidget(self._tune_progress)
        return self._tune_box

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

    def _build_export_section(self) -> QWidget:
        self._export_box = QGroupBox("6. Export")
        layout = QHBoxLayout(self._export_box)
        self._export_btn = QPushButton("Export to params.json")
        self._export_btn.clicked.connect(self._on_export)
        layout.addWidget(self._export_btn)
        self._export_label = QLabel("")
        layout.addWidget(self._export_label, stretch=1)
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
        self._show_tab(self._diag_canvas)
        self._sync_enabled()

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

    # --- section 4: tune (button-triggered, off-main predict) -----------

    def _on_tune(self) -> None:
        """Run the predictor off-main for the current spinbox values, draw + record.

        The manual tuning IS the final fit: the done slot records the accepted
        g/bare_rf and draws the figure. The button is disabled while the prediction
        is in flight, so a new run cannot start until the previous one finishes.
        """
        if self._ctrl.state.preprocess is None:
            return
        params = _TuneParams(
            g=1e-3 * self._g_spin.value(),
            bare_rf=1e-3 * self._rf_spin.value(),
            res_dim=self._res_dim.value(),
            step=self._step.value(),
        )
        self._tune_btn.setEnabled(False)
        self._begin_progress(self._tune_progress)
        worker = _PredictWorker(self._ctrl, params)
        worker.signals.done.connect(self._on_tune_done)
        worker.signals.failed.connect(self._on_tune_failed)
        self._pool.start(worker)

    def _on_tune_done(self, data: _TuneData) -> None:
        st = self._ctrl.state
        self._tune_btn.setEnabled(True)
        self._end_progress()
        if st.preprocess is None:
            return
        # The accepted tuning IS the result: record g/bare_rf (+ its resolution),
        # then draw the tune figure.
        self._ctrl.set_manual_fit(
            data.g, data.bare_rf, res_dim=data.res_dim, step=data.step
        )
        self._tune_artists = render_tune_figure(
            self._tune_canvas.figure,
            st.preprocess.sp_fluxs,
            st.preprocess.sp_freqs,
            st.preprocess.norm_phases,
            data.t_fluxs,
            data.rf_0,
            data.rf_1,
            data.g,
            data.bare_rf,
        )
        self._tune_canvas.redraw()
        self._show_tab(self._tune_canvas)
        self._sync_enabled()

    def _on_tune_failed(self, message: str) -> None:
        self._tune_btn.setEnabled(True)
        self._end_progress()
        self._warn("Tune failed", friendly_fit_message("Tune", message))

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

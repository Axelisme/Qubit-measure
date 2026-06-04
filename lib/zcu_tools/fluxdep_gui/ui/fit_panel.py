"""FitPanelWidget — the v2 database-search fit panel.

Shown in the editing area (via the "Fit spectrum…" button). A structured form
collects the search inputs; a worker thread runs ``search_in_database`` (it
releases the GIL in numba, so a thread keeps the UI live without process IPC);
the result is the best (EJ, EC, EL) plus a matplotlib visualisation and the
search's native per-parameter diagnostic figure. An Export button writes
``params.json``.

The search is NOT cancellable (a single deterministic sweep) — the panel just
disables the Search button and shows progress while it runs. Progress is fed from
the worker through a ``GuiProgressBarChannel`` (queued Qt signal) so the notebook
search's ``make_pbar`` calls drive a real Qt progress bar.
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
    Qt,  # type: ignore[attr-defined]
    QThreadPool,
    Signal,  # type: ignore[attr-defined]
)
from qtpy.QtWidgets import (  # type: ignore[attr-defined]
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QWidget,
)

from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.services.fit import SearchResult
from zcu_tools.fluxdep_gui.services.viz import render_fit_figure
from zcu_tools.fluxdep_gui.ui.gui_pbar import GuiProgressBarChannel
from zcu_tools.fluxdep_gui.ui.plot_host import FigureContainer, set_current_container
from zcu_tools.fluxdep_gui.ui.transitions_form import TransitionsForm
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux

logger = logging.getLogger(__name__)

_N_SIM_FLUX = 1000  # simulated flux grid resolution for the visualisation

# (EJb, ECb, ELb) bound presets — the notebook's named fitting ranges (GHz).
# Selecting a preset fills the bound spin boxes; transitions are independent.
_BOUND_PRESETS: dict[str, tuple[tuple[float, float], ...]] = {
    "general": ((2.0, 15.0), (0.2, 2.0), (0.1, 2.0)),
    "integer": ((3.0, 6.0), (0.8, 2.0), (0.08, 0.2)),
    "all": ((1.0, 20.0), (0.1, 4.0), (0.01, 3.0)),
}


def _bound_spinbox(value: float) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(0.0, 1000.0)
    box.setDecimals(3)
    box.setSingleStep(0.1)
    box.setValue(value)
    return box


class _SearchSignals(QObject):
    # SearchResult on success; (message) on error.
    done = Signal(object)
    failed = Signal(str)


class _SearchWorker(QRunnable):
    """Runs the PURE ``compute_search`` off the main thread (it releases the GIL).

    ``compute_search`` reads State up front but writes nothing, so it is safe to
    run on a worker thread (per the main-thread State invariant). The result is
    emitted to the main thread, where ``_on_search_done`` records it on State via
    ``record_search_result``. The numba kernel releases the GIL, so a thread (not
    a process) keeps the UI live without paying the cost of pickling the large
    signal arrays across a process boundary.
    """

    def __init__(self, ctrl: Controller, pbar_factory) -> None:
        super().__init__()
        self.signals = _SearchSignals()
        self._ctrl = ctrl
        self._pbar_factory = pbar_factory

    def run(self) -> None:
        try:
            result = self._ctrl.compute_search(
                pbar_factory=self._pbar_factory, plot=True
            )
        except Exception as exc:  # noqa: BLE001 — surface to the panel
            logger.exception("search worker failed")
            self.signals.failed.emit(str(exc))
            return
        self.signals.done.emit(result)


class FitPanelWidget(QWidget):
    """Structured search-parameter form + result visualisation + export."""

    def __init__(self, ctrl: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._pool = QThreadPool.globalInstance() or QThreadPool(self)
        self._channel = GuiProgressBarChannel()
        self._channel.progress.connect(self._on_progress)

        self._build_ui()
        self._load_from_state()

    # --- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # Left: the structured form (width adjustable via the splitter).
        form_box = QGroupBox("Search parameters")
        form = QFormLayout(form_box)

        self._db_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse_db)
        db_row = QHBoxLayout()
        db_row.addWidget(self._db_edit, stretch=1)
        db_row.addWidget(browse)
        db_holder = QWidget()
        db_holder.setLayout(db_row)
        form.addRow("Database", db_holder)

        # Preset selects a named (EJb, ECb, ELb) range → fills the bound boxes.
        self._preset = QComboBox()
        self._preset.addItems(sorted(_BOUND_PRESETS))
        self._preset.activated.connect(self._on_preset_selected)  # only on user pick
        form.addRow("Bounds preset", self._preset)

        self._ej_lo, self._ej_hi = _bound_spinbox(2.0), _bound_spinbox(15.0)
        self._ec_lo, self._ec_hi = _bound_spinbox(0.2), _bound_spinbox(2.0)
        self._el_lo, self._el_hi = _bound_spinbox(0.1), _bound_spinbox(2.0)
        form.addRow("EJ bounds", self._bound_row(self._ej_lo, self._ej_hi))
        form.addRow("EC bounds", self._bound_row(self._ec_lo, self._ec_hi))
        form.addRow("EL bounds", self._bound_row(self._el_lo, self._el_hi))

        self._r_f = _bound_spinbox(0.0)
        self._sample_f = _bound_spinbox(0.0)
        form.addRow("r_f (GHz)", self._r_f)
        form.addRow("sample_f (GHz)", self._sample_f)

        self._transitions_form = TransitionsForm()
        form.addRow(QLabel("Transitions"))
        form.addRow(self._transitions_form)

        self._search_btn = QPushButton("Search database")
        self._search_btn.clicked.connect(self._on_search)
        form.addRow(self._search_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        form.addRow(self._progress)

        self._export_btn = QPushButton("Export params.json")
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setEnabled(False)
        form.addRow(self._export_btn)

        self._status = QLabel("")
        form.addRow(self._status)

        # Right: two figures in tabs (each gets the full space). Tab 1 is our fit
        # visualisation; tab 2 is the search's native diagnostic figure — a
        # FigureContainer the embedded matplotlib backend routes
        # search_in_database(plot=True)'s pyplot figure into (so fitting.py just
        # uses pyplot and the figure lands here).
        self._tabs = QTabWidget()

        self._fit_figure = Figure(figsize=(6, 4))
        self._fit_canvas = FigureCanvasQTAgg(self._fit_figure)
        self._tabs.addTab(self._fit_canvas, "Fit")

        self._diag_stack = QStackedWidget()
        diag_placeholder = QLabel("Search to see the diagnostic plot.")
        diag_placeholder.setEnabled(False)
        self._diag_stack.addWidget(diag_placeholder)
        self._diag_container = FigureContainer(self._diag_stack, diag_placeholder)
        self._tabs.addTab(self._diag_stack, "Diagnostic")

        # A draggable splitter lets the user resize the form vs the figures.
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(form_box)
        splitter.addWidget(self._tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 640])
        root.addWidget(splitter)

    def _bound_row(self, lo: QDoubleSpinBox, hi: QDoubleSpinBox) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(lo)
        row.addWidget(QLabel("–"))
        row.addWidget(hi)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    def _load_from_state(self) -> None:
        fit = self._ctrl.state.fit
        if fit.database_path:
            self._db_edit.setText(fit.database_path)
        self._ej_lo.setValue(fit.EJb[0])
        self._ej_hi.setValue(fit.EJb[1])
        self._ec_lo.setValue(fit.ECb[0])
        self._ec_hi.setValue(fit.ECb[1])
        self._el_lo.setValue(fit.ELb[0])
        self._el_hi.setValue(fit.ELb[1])
        self._r_f.setValue(fit.r_f)
        self._sample_f.setValue(fit.sample_f)
        self._transitions_form.set_transitions(fit.transitions)
        self._export_btn.setEnabled(fit.has_result)

    # --- form → State ----------------------------------------------------

    def _commit_params(self) -> None:
        """Read the form into State (fast-fails on a malformed transition field)."""
        transitions = self._transitions_form.get_transitions()
        r_f = self._r_f.value()
        sample_f = self._sample_f.value()
        if r_f:
            transitions["r_f"] = r_f
        if sample_f:
            transitions["sample_f"] = sample_f
        self._ctrl.set_fit_params(
            database_path=self._db_edit.text().strip(),
            EJb=(self._ej_lo.value(), self._ej_hi.value()),
            ECb=(self._ec_lo.value(), self._ec_hi.value()),
            ELb=(self._el_lo.value(), self._el_hi.value()),
            transitions=transitions,
            r_f=r_f,
            sample_f=sample_f,
        )

    # --- actions ---------------------------------------------------------

    def _on_preset_selected(self, _index: int) -> None:
        """User picked a bounds preset → fill the EJ/EC/EL bound spin boxes."""
        preset = _BOUND_PRESETS.get(self._preset.currentText())
        if preset is None:
            return
        (ej, ec, el) = preset
        self._ej_lo.setValue(ej[0])
        self._ej_hi.setValue(ej[1])
        self._ec_lo.setValue(ec[0])
        self._ec_hi.setValue(ec[1])
        self._el_lo.setValue(el[0])
        self._el_hi.setValue(el[1])

    def _on_browse_db(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select database", filter="HDF5 (*.h5 *.hdf5);;All files (*)"
        )
        if path:
            self._db_edit.setText(path)

    def _on_search(self) -> None:
        # Block early on an empty / missing database path (don't run the worker
        # just to fast-fail with a traceback).
        db_path = self._db_edit.text().strip()
        if not db_path:
            self._status.setText("Select a database first.")
            return
        if not os.path.isfile(db_path):
            self._status.setText(f"Database not found: {db_path}")
            return
        try:
            self._commit_params()
        except ValueError as exc:
            self._status.setText(f"Invalid parameters: {exc}")
            return
        self._search_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # busy until the first progress tick
        self._status.setText("Searching…")

        # Route the search's pyplot diagnostic figure into our container. Set on
        # the main thread before the worker; cleared when it finishes (the worker
        # thread reads this module-level value during search_in_database).
        # Close any pyplot figure left over from a previous search first: pyplot's
        # global figure stack (Gcf) keeps every plt.figure() that is never closed,
        # so a second search's plt.show() would otherwise act on a stale, already-
        # detached figure and raise "not attached". Closing only drops pyplot's
        # reference; the embedded canvas already lives in the container.
        import matplotlib.pyplot as plt

        plt.close("all")
        self._diag_container.clear()
        set_current_container(self._diag_container)

        worker = _SearchWorker(self._ctrl, self._channel.factory())
        worker.signals.done.connect(self._on_search_done)  # type: ignore[arg-type]
        worker.signals.failed.connect(self._on_search_failed)
        self._pool.start(worker)

    def _on_progress(self, n: float, total: float, desc: str) -> None:
        if total > 0:
            self._progress.setRange(0, int(total))
            self._progress.setValue(int(n))
        else:
            self._progress.setRange(0, 0)
        if desc:
            self._status.setText(desc)

    def _on_search_done(self, result: SearchResult) -> None:
        # Main thread: record the worker's pure result onto State (the search
        # itself ran off-main and touched no State), then render. The diagnostic
        # figure is already embedded in the container (the backend routed the
        # search's pyplot figure there), so we only stop routing now.
        set_current_container(None)
        self._search_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._ctrl.record_search_result(result)
        EJ, EC, EL = result.params
        self._status.setText(f"EJ={EJ:.3f}  EC={EC:.3f}  EL={EL:.3f}")
        self._export_btn.setEnabled(True)
        self._render_result(result.params)
        # surface the freshly embedded diagnostic figure
        self._tabs.setCurrentWidget(self._diag_stack)

    def _on_search_failed(self, message: str) -> None:
        set_current_container(None)
        self._search_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText(f"Search failed: {message}")

    def _on_export(self) -> None:
        try:
            path = self._ctrl.export_params()
        except Exception as exc:  # noqa: BLE001 — surface to the panel
            self._status.setText(f"Export failed: {exc}")
            return
        self._status.setText(f"Exported → {path}")

    # --- rendering -------------------------------------------------------

    def _render_result(self, params: tuple) -> None:
        """Draw the fit visualisation (background + sim lines + points + ...)."""
        spectrums = self._ctrl.state.spectrums
        s_fluxs, s_freqs = self._ctrl.selected_pointcloud()
        if s_fluxs.size == 0:
            return
        flux_lo, flux_hi = float(np.min(s_fluxs)), float(np.max(s_fluxs))
        if flux_hi <= flux_lo:
            flux_lo, flux_hi = flux_lo - 0.1, flux_hi + 0.1
        t_fluxs = np.linspace(flux_lo, flux_hi, _N_SIM_FLUX)
        _, energies = calculate_energy_vs_flux(
            params, t_fluxs, cutoff=40, evals_count=15
        )

        fit = self._ctrl.state.fit
        aligned = next((e for e in spectrums.values() if e.aligned), None)
        render_fit_figure(
            self._fit_figure,
            spectrums,
            t_fluxs,
            energies,
            fit.transitions,
            s_fluxs,
            s_freqs,
            r_f=fit.r_f,
            sample_f=fit.sample_f,
            flux_half=aligned.flux_half if aligned else None,
            flux_period=aligned.flux_period if aligned else None,
            title=f"EJ/EC/EL = ({params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f})",
        )
        self._fit_canvas.draw_idle()

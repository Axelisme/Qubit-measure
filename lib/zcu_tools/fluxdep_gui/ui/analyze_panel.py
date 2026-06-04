"""AnalyzePanelWidget — the v2 analysis panel: Filter / Search / Show tabs.

One panel (a MainWindow singleton) gathers the three post-selection analysis
steps that used to be separate buttons:

- **Filter**: the cross-spectrum selector over the joint point cloud.
- **Search**: the database-search form + the search's native diagnostic figure.
- **Show**: the fit visualisation (background / simulation lines / points /
  constant-freq lines), with display tools — y/x axis limits (default matching
  the notebook's ``auto_derive_limits``), r_f / sample_f reference-line toggles,
  and a transition display subset.

The search runs on a worker thread; its pyplot diagnostic figure is routed into
a FigureContainer by the embedded matplotlib backend (see ``mpl_backend`` /
``plot_host``). pyplot's global figure stack is cleared before each search so a
stale figure cannot resurface.
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
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.services.fit import SearchResult
from zcu_tools.fluxdep_gui.services.viz import derive_auto_limits, render_fit_figure
from zcu_tools.fluxdep_gui.ui.error_messages import friendly_fit_message
from zcu_tools.fluxdep_gui.ui.gui_pbar import GuiProgressBarChannel
from zcu_tools.fluxdep_gui.ui.plot_host import FigureContainer, set_current_container
from zcu_tools.fluxdep_gui.ui.transitions_form import TransitionsForm
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux

logger = logging.getLogger(__name__)

_N_SIM_FLUX = 1000  # simulated flux grid resolution for the visualisation

# (EJb, ECb, ELb) bound presets — the notebook's named fitting ranges (GHz).
_BOUND_PRESETS: dict[str, tuple[tuple[float, float], ...]] = {
    "general": ((2.0, 15.0), (0.2, 2.0), (0.1, 2.0)),
    "integer": ((3.0, 6.0), (0.8, 2.0), (0.08, 0.2)),
    "all": ((1.0, 20.0), (0.1, 4.0), (0.01, 3.0)),
}


def _bound_spinbox(value: float, hi: float = 1000.0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(-hi, hi)
    box.setDecimals(3)
    box.setSingleStep(0.1)
    box.setValue(value)
    return box


def _freq_edit() -> QLineEdit:
    """A nullable frequency field: blank means unset, else a float."""
    from qtpy.QtGui import QDoubleValidator  # type: ignore[attr-defined]

    edit = QLineEdit()
    edit.setPlaceholderText("(unset)")
    edit.setValidator(QDoubleValidator())
    return edit


def _parse_freq(edit: QLineEdit) -> "float | None":
    """The field's value as float, or None when blank / unparseable."""
    text = edit.text().strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _set_freq(edit: QLineEdit, value: "float | None") -> None:
    """Show ``value`` in the field (blank when None)."""
    edit.setText("" if value is None else f"{value:g}")


class _SearchSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _SearchWorker(QRunnable):
    """Runs the PURE ``compute_search`` off the main thread (it releases the GIL)."""

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


class AnalyzePanelWidget(QWidget):
    """Filter / Search / Show tabs over the selected joint point cloud."""

    def __init__(self, ctrl: Controller, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ctrl = ctrl
        self._pool = QThreadPool.globalInstance() or QThreadPool(self)
        self._channel = GuiProgressBarChannel()
        self._channel.progress.connect(self._on_progress)
        self._filter_widget: Optional[QWidget] = None

        self._build_ui()
        self._load_from_state()

    # --- construction ----------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_filter_tab(), "Filter")
        self._tabs.addTab(self._build_search_tab(), "Search")
        self._tabs.addTab(self._build_show_tab(), "Show")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs)

    # --- Filter tab ------------------------------------------------------

    def _build_filter_tab(self) -> QWidget:
        holder = QWidget()
        self._filter_layout = QVBoxLayout(holder)
        self._filter_placeholder = QLabel(
            "Select points on a spectrum first, then open this tab."
        )
        self._filter_placeholder.setEnabled(False)
        self._filter_layout.addWidget(self._filter_placeholder)
        return holder

    def _refresh_filter_tab(self) -> None:
        """(Re)build the cross-spectrum selector for the current spectra."""
        from zcu_tools.fluxdep_gui.ui.interactive.selector import SelectorWidget
        from zcu_tools.notebook.persistance import SpectrumResult

        if self._filter_widget is not None:
            self._filter_layout.removeWidget(self._filter_widget)
            self._filter_widget.deleteLater()
            self._filter_widget = None

        spectrums: dict[str, SpectrumResult] = {
            n: SpectrumResult(
                type=e.spec_type,
                flux_half=e.flux_half,
                flux_int=e.flux_int,
                flux_period=e.flux_period,
                spectrum=e.raw,
                points=e.points,
            )
            for n, e in self._ctrl.state.spectrums.items()
            if e.points_selected
        }
        if not spectrums:
            self._filter_placeholder.setVisible(True)
            return
        self._filter_placeholder.setVisible(False)
        selector = SelectorWidget(
            spectrums, min_distance=self._ctrl.state.selection.min_distance
        )

        def _on_finish() -> None:
            _fluxs, _freqs, selected = selector.get_result()
            self._ctrl.set_selection(selected, selector.min_distance())

        selector.finished.connect(_on_finish)
        self._filter_widget = selector
        self._filter_layout.addWidget(selector)

    # --- Search tab ------------------------------------------------------

    def _build_search_tab(self) -> QWidget:
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]
        from qtpy.QtWidgets import QSplitter  # type: ignore[attr-defined]

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

        self._preset = QComboBox()
        self._preset.addItems(sorted(_BOUND_PRESETS))
        self._preset.activated.connect(self._on_preset_selected)
        form.addRow("Bounds preset", self._preset)

        self._ej_lo, self._ej_hi = _bound_spinbox(2.0), _bound_spinbox(15.0)
        self._ec_lo, self._ec_hi = _bound_spinbox(0.2), _bound_spinbox(2.0)
        self._el_lo, self._el_hi = _bound_spinbox(0.1), _bound_spinbox(2.0)
        form.addRow("EJ bounds", self._bound_row(self._ej_lo, self._ej_hi))
        form.addRow("EC bounds", self._bound_row(self._ec_lo, self._ec_hi))
        form.addRow("EL bounds", self._bound_row(self._el_lo, self._el_hi))

        # r_f / sample_f are optional (blank = unset); a transition category that
        # needs one is validated before search.
        self._r_f = _freq_edit()
        self._sample_f = _freq_edit()
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

        # Right: the search's native diagnostic figure (routed via the backend).
        self._diag_stack = QStackedWidget()
        diag_placeholder = QLabel("Search to see the diagnostic plot.")
        diag_placeholder.setEnabled(False)
        self._diag_stack.addWidget(diag_placeholder)
        self._diag_container = FigureContainer(self._diag_stack, diag_placeholder)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(form_box)
        splitter.addWidget(self._diag_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 640])
        holder = QWidget()
        QVBoxLayout(holder).addWidget(splitter)
        return holder

    # --- Show tab --------------------------------------------------------

    def _build_show_tab(self) -> QWidget:
        from qtpy.QtCore import Qt  # type: ignore[attr-defined]
        from qtpy.QtWidgets import QSplitter  # type: ignore[attr-defined]

        tools_box = QGroupBox("Display")
        tools = QFormLayout(tools_box)

        self._x_lo, self._x_hi = _bound_spinbox(0.0), _bound_spinbox(1.0)
        self._y_lo, self._y_hi = _bound_spinbox(0.0), _bound_spinbox(10.0)
        tools.addRow("x limits", self._bound_row(self._x_lo, self._x_hi))
        tools.addRow("y limits", self._bound_row(self._y_lo, self._y_hi))
        auto_btn = QPushButton("Auto limits")
        auto_btn.clicked.connect(self._apply_auto_limits)
        tools.addRow(auto_btn)

        self._show_const_freq = QCheckBox("r_f / sample_f reference lines")
        self._show_const_freq.setChecked(True)
        tools.addRow(self._show_const_freq)

        self._transitions_show = TransitionsForm()
        tools.addRow(QLabel("Transitions to show"))
        tools.addRow(self._transitions_show)

        redraw = QPushButton("Apply")
        redraw.clicked.connect(self._redraw_show)
        tools.addRow(redraw)

        self._fit_figure = Figure(figsize=(6, 4))
        self._fit_canvas = FigureCanvasQTAgg(self._fit_figure)

        tools_box.setMaximumWidth(380)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tools_box)
        splitter.addWidget(self._fit_canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 640])
        holder = QWidget()
        QVBoxLayout(holder).addWidget(splitter)
        return holder

    # --- helpers ---------------------------------------------------------

    def _bound_row(self, lo: QDoubleSpinBox, hi: QDoubleSpinBox) -> QWidget:
        row = QHBoxLayout()
        row.addWidget(lo)
        row.addWidget(QLabel("–"))
        row.addWidget(hi)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    def _load_from_state(self) -> None:
        from zcu_tools.fluxdep_gui.ui.paths import default_database_file

        fit = self._ctrl.state.fit
        # Seed the database path: the prior fit's path, else a bundled default so
        # the user usually doesn't have to browse at all.
        self._db_edit.setText(
            fit.database_path or default_database_file(self._ctrl.state.project)
        )
        self._ej_lo.setValue(fit.EJb[0])
        self._ej_hi.setValue(fit.EJb[1])
        self._ec_lo.setValue(fit.ECb[0])
        self._ec_hi.setValue(fit.ECb[1])
        self._el_lo.setValue(fit.ELb[0])
        self._el_hi.setValue(fit.ELb[1])
        _set_freq(self._r_f, fit.r_f)
        _set_freq(self._sample_f, fit.sample_f)
        self._transitions_form.set_transitions(fit.transitions)
        self._transitions_show.set_transitions(fit.transitions)
        self._export_btn.setEnabled(fit.has_result)

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.tabText(index) == "Filter":
            self._refresh_filter_tab()

    # --- Search actions --------------------------------------------------

    def _missing_freq_message(self, transitions, r_f, sample_f):
        """Return a message if a chosen transition needs an unset r_f/sample_f."""
        from zcu_tools.fluxdep_gui.state import (
            transitions_need_r_f,
            transitions_need_sample_f,
        )

        if transitions_need_r_f(transitions) and r_f is None:
            return (
                "The transitions include a blue/red-side or mirror category, which "
                "needs r_f. Fill r_f (GHz), or remove those transitions."
            )
        if transitions_need_sample_f(transitions) and sample_f is None:
            return (
                "The transitions include a mirror category, which needs sample_f. "
                "Fill sample_f (GHz), or remove the mirror transitions."
            )
        return None

    def _commit_params(self) -> None:
        transitions = self._transitions_form.get_transitions()
        r_f = _parse_freq(self._r_f)
        sample_f = _parse_freq(self._sample_f)
        self._ctrl.set_fit_params(
            database_path=self._db_edit.text().strip(),
            EJb=(self._ej_lo.value(), self._ej_hi.value()),
            ECb=(self._ec_lo.value(), self._ec_hi.value()),
            ELb=(self._el_lo.value(), self._el_hi.value()),
            transitions=transitions,
            r_f=r_f,
            sample_f=sample_f,
        )

    def _on_preset_selected(self, _index: int) -> None:
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
        from zcu_tools.fluxdep_gui.ui.paths import database_dir

        # Start in the current path's folder if set, else the project / bundled
        # simulation database directory.
        current = self._db_edit.text().strip()
        start = os.path.dirname(current) if current else ""
        if not start or not os.path.isdir(start):
            start = database_dir(self._ctrl.state.project)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select database", start, filter="HDF5 (*.h5 *.hdf5);;All files (*)"
        )
        if path:
            self._db_edit.setText(path)

    def _on_search(self) -> None:
        db_path = self._db_edit.text().strip()
        if not db_path:
            self._status.setText("Select a database first.")
            return
        if not os.path.isfile(db_path):
            self._status.setText(f"Database not found: {db_path}")
            return
        # Pre-check that needed r_f / sample_f are filled for the chosen
        # transitions (else search_in_database would fail with a cryptic error).
        missing = self._missing_freq_message(
            self._transitions_form.get_transitions(),
            _parse_freq(self._r_f),
            _parse_freq(self._sample_f),
        )
        if missing:
            self._show_message("Missing frequency", missing)
            return
        try:
            self._commit_params()
        except ValueError as exc:
            self._status.setText(f"Invalid parameters: {exc}")
            return
        self._search_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._status.setText("Searching…")

        # Clear pyplot's global figure stack so a previous search's figure can't
        # resurface (its embedded canvas already lives in the container).
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
        set_current_container(None)
        self._search_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._ctrl.record_search_result(result)
        EJ, EC, EL = result.params
        self._status.setText(f"EJ={EJ:.3f}  EC={EC:.3f}  EL={EL:.3f}")
        self._export_btn.setEnabled(True)
        self._apply_auto_limits()
        self._redraw_show()
        # surface the fit visualisation
        self._tabs.setCurrentIndex(self._tabs.count() - 1)  # Show tab

    def _on_search_failed(self, message: str) -> None:
        set_current_container(None)
        self._search_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._status.setText("Search failed.")
        self._show_message("Search failed", friendly_fit_message("Search", message))

    def _on_export(self) -> None:
        try:
            path = self._ctrl.export_params()
        except Exception as exc:  # noqa: BLE001 — surface to the panel
            self._status.setText("Export failed.")
            self._show_message("Export failed", friendly_fit_message("Export", exc))
            return
        self._status.setText(f"Exported → {path}")

    def _show_message(self, title: str, message: str) -> None:
        from qtpy.QtWidgets import QMessageBox  # type: ignore[attr-defined]

        QMessageBox.warning(self, title, message)

    # --- Show actions ----------------------------------------------------

    def _apply_auto_limits(self) -> None:
        """Set the x/y limit boxes to the auto-derived (notebook) limits."""
        fit = self._ctrl.state.fit
        s_fluxs, s_freqs = self._ctrl.selected_pointcloud()
        (x_lo, x_hi), (y_lo, y_hi) = derive_auto_limits(
            self._ctrl.state.spectrums, s_fluxs, s_freqs, fit.r_f, fit.sample_f
        )
        self._x_lo.setValue(x_lo)
        self._x_hi.setValue(x_hi)
        self._y_lo.setValue(y_lo)
        self._y_hi.setValue(y_hi)

    def _redraw_show(self) -> None:
        """Render the fit visualisation with the current display tools."""
        fit = self._ctrl.state.fit
        if fit.params is None:
            return
        spectrums = self._ctrl.state.spectrums
        s_fluxs, s_freqs = self._ctrl.selected_pointcloud()
        x_lo, x_hi = self._x_lo.value(), self._x_hi.value()
        flux_lo, flux_hi = (x_lo, x_hi) if x_hi > x_lo else (0.0, 1.0)
        t_fluxs = np.linspace(flux_lo, flux_hi, _N_SIM_FLUX)
        _, energies = calculate_energy_vs_flux(
            fit.params, t_fluxs, cutoff=40, evals_count=15
        )

        # The show-transitions subset may need an r_f/sample_f that's unset;
        # check before drawing so it surfaces a friendly message, not a crash.
        show_transitions = self._transitions_show.get_transitions()
        missing = self._missing_freq_message(show_transitions, fit.r_f, fit.sample_f)
        if missing:
            self._status.setText("Show: " + missing.split(".")[0] + ".")
            show_transitions = self._transitions_form.get_transitions()  # fit set

        aligned = next((e for e in spectrums.values() if e.aligned), None)
        try:
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
                title=(
                    f"EJ/EC/EL = ({fit.params[0]:.2f}, {fit.params[1]:.2f}, "
                    f"{fit.params[2]:.2f})"
                ),
                xlim=(self._x_lo.value(), self._x_hi.value()),
                ylim=(self._y_lo.value(), self._y_hi.value()),
                show_const_freq=self._show_const_freq.isChecked(),
                plot_transitions=show_transitions,
            )
        except Exception as exc:  # noqa: BLE001 — a show-config issue, not fatal
            logger.exception("render_fit_figure failed")
            self._status.setText(f"Could not draw: {exc}")
            return
        self._fit_canvas.draw_idle()

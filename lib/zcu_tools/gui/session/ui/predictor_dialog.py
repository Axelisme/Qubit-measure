"""Independent fluxonium / SQUID transmon simulator dialog; edits and calibration remain local.

The optional device control supplies values only. params.json loads are read-only
prefill. No operation installs a model or writes a component or source file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT
from qtpy.QtCore import QTimer, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QCloseEvent, QShowEvent  # type: ignore[attr-defined]
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.expected_error import ExpectedError, FailedPreconditionError
from zcu_tools.gui.session.persistence import PersistenceError, SingleFileCaretaker
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.session.ui.predictor_cache import (
    PredictorSnapshot,
    PredictorSnapshotCodec,
)
from zcu_tools.gui.session.ui.predictor_canvas import PredictorCurveCanvas
from zcu_tools.gui.widgets.remote_edit_gate import (
    bind_control_enabled,
    dispose_control_watches,
    local_write_allowed,
)
from zcu_tools.gui.widgets.spinbox import FiniteFloatSpinBox, TrimDoubleSpinBox
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.parameters import ModelParams
from zcu_tools.simulate.predictor import (
    CalibrationResult,
    MatrixPredictor,
    PredictorInputError,
)
from zcu_tools.simulate.transmon import TransmonParams

if TYPE_CHECKING:
    from zcu_tools.device.base import BaseDeviceInfo
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.events import DeviceChangedPayload
    from zcu_tools.gui.session.parameters import ParameterContext
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

logger = logging.getLogger(__name__)

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

# Initial standalone trial values, not defaults for missing component inputs.
_DEFAULT_EJ: float = 4.0
_DEFAULT_EC: float = 1.0
_DEFAULT_EL: float = 1.0
_DEFAULT_EJMAX: float = 20.0
_DEFAULT_TRANSMON_EC: float = 0.25
_DEFAULT_ASYMMETRY: float = 0.0
_DEFAULT_FLUX_HALF: float = 0.0
_DEFAULT_FLUX_PERIOD: float = 0.005
_DEFAULT_FLUX_BIAS: float = 0.0
_DEFAULT_CALIBRATION_EXPR: str = "qubit.freq"

# Column indices for the 4-column tracked-transitions table.
# Both matrix operators (n and phi) are shown side by side — there is no selector.
_COL_TRANSITION = 0
_COL_FREQ = 1
_COL_MAG_N = 2
_COL_MAG_PHI = 3


@dataclass(frozen=True)
class PredictorDialogState:
    """App-neutral persistent UI state for PredictorDialog."""

    tracked_transitions: tuple[tuple[int, int], ...]
    tab_index: int
    params_path_text: str


def _normalize_tracked_transitions(
    transitions: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for transition in transitions:
        if len(transition) != 2:
            continue
        frm_raw, to_raw = transition
        if isinstance(frm_raw, bool) or isinstance(to_raw, bool):
            continue
        frm = int(frm_raw)
        to = int(to_raw)
        candidate = (frm, to)
        if frm < 0 or to < 0 or frm >= to or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return tuple(normalized)


class PredictorDialog(QDialog):
    """Own detached fluxonium or SQUID transmon inputs for prediction and calibration.

    device optionally supplies cached/live device values. parameter_provider
    supplies fixed ParameterContext captures for component choices/alias default
    and frequency expressions; omitted or None-returning providers allow only
    explicit trials. params optionally prefills complete or partial local inputs.
    Unresolved frequency paths disable calibration without substituting a value.
    persistent_on_close hides instead of finishing, retaining the plotted model.
    Optional control_identity gates the controls but leaves Close and plots
    usable. Browse rechecks that owner after the file chooser before trial edits.
    Omitted authority preserves the behavior of apps without control identity.
    """

    state_changed = Signal()

    def __init__(
        self,
        predictor: PredictorControlPort,
        parent: QWidget | None = None,
        *,
        device: DeviceControlPort | None = None,
        parameter_provider: Callable[[], ParameterContext | None] | None = None,
        persistent_on_close: bool = False,
        control_identity: ControlIdentity | None = None,
        params: ModelParams | None = None,
    ) -> None:
        super().__init__(parent)
        self._pred = predictor
        self._control_identity = control_identity
        self._dev = device
        self._parameter_provider = parameter_provider or (lambda: None)
        self._persistent_on_close = persistent_on_close
        self._live_mode = False
        self._suppress_state_changed = False
        self._updating_parameters = False
        self._known_fields: dict[FiniteFloatSpinBox, QCheckBox] = {}
        self._caretaker: SingleFileCaretaker[PredictorSnapshot, None] | None = None
        self.setWindowTitle("Predictor")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(480)
        # Open tall enough that the transition table shows several rows.
        self.resize(1000, 700)

        # Mutable list of currently tracked (from, to) transition pairs.
        # Modified only by _add_transition / _delete_transition / _on_remove_selected.
        self._tracked: list[tuple[int, int]] = list(_DEFAULT_TRANSITIONS)

        # Last computed results; None when predictor is not loaded or
        # tracked list is empty. Both matrix operators are always tracked.
        self._last_freq_result = None
        self._last_mat_n_result = None
        self._last_mat_phi_result = None
        self._synced_xlim: tuple[float, float] | None = None
        self._syncing_xlim = False
        # Matrix curves are an optional model capability, learned from the built model.
        self._matrix_supported = True

        # Root layout: left controls | right tab widget.
        root_layout = QHBoxLayout(self)

        left_widget = self._build_controls()
        tabs = self._build_canvases()
        root_layout.addWidget(left_widget, stretch=0)
        root_layout.addWidget(tabs, stretch=1)

        self._initialize_trial(params)

        self._unsubscribe_device_changed: Callable[[], None] | None = (
            device.on_device_changed(self._on_device_changed)
            if device is not None
            else None
        )
        self.finished.connect(self._cleanup_subscriptions)
        self.destroyed.connect(self._cleanup_subscriptions)

    def _build_controls(self) -> QWidget:
        """Compose editing groups, status and actions, then bind their authority gates."""
        # ── Left: controls column ─────────────────────────────────────────
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        model_group = self._build_model_group()
        left_layout.addWidget(model_group)
        transitions_group = self._build_transitions_group()
        left_layout.addWidget(transitions_group, stretch=1)
        calibration_group = self._build_calibration_group()
        left_layout.addWidget(calibration_group)

        # ── status label ──────────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setObjectName("predictor_status")
        left_layout.addWidget(self._status_label)

        # ── Remove + Close on one horizontal row (saves vertical space) ────
        # Remove removes ALL currently selected transition rows; Close closes
        # the dialog. A spacer pushes Close to the right edge.
        action_row = QHBoxLayout()
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_selected)
        remove_region = QWidget()
        remove_layout = QHBoxLayout(remove_region)
        remove_layout.setContentsMargins(0, 0, 0, 0)
        remove_layout.addWidget(self._remove_btn)
        action_row.addWidget(remove_region)
        action_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self._on_close_requested)
        action_row.addWidget(close_btn)
        left_layout.addLayout(action_row)
        for content in (
            model_group,
            transitions_group,
            calibration_group,
            remove_region,
        ):
            bind_control_enabled(
                content,
                self._control_identity,
                retirement_dialog=None if self._persistent_on_close else self,
            )

        return left_widget

    def _build_model_group(self) -> QGroupBox:
        """Build model/component selection and file loading beside trial energy fields."""
        # ── Model params (editable EJ/EC/EL/flux anchors/flux_bias) ───────
        # File loading only prefills independent trial parameters.
        model_group = QGroupBox("Model params")
        model_form = QFormLayout(model_group)

        self._model_combo = QComboBox()
        self._model_combo.addItem("Fluxonium", "fluxonium")
        self._model_combo.addItem("SQUID transmon", "transmon")
        model_form.addRow("Model:", self._model_combo)

        self._component_combo = QComboBox()
        self._component_combo.setMinimumContentsLength(12)
        component_row = QHBoxLayout()
        component_row.addWidget(self._component_combo, stretch=1)
        self._component_refresh_btn = QPushButton("↻")
        self._component_refresh_btn.setToolTip(
            "Refresh component choices without loading parameters"
        )
        self._component_refresh_btn.clicked.connect(self._refresh_components)
        component_row.addWidget(self._component_refresh_btn)
        self._component_load_btn = QPushButton("Load component")
        self._component_load_btn.clicked.connect(self._on_load_component_clicked)
        component_row.addWidget(self._component_load_btn)
        # Same as the params.json row: a holder widget's margins squeeze the row.
        model_form.addRow("Component:", component_row)

        self._params_path_edit = QLineEdit()
        self._params_path_edit.setPlaceholderText("params.json containing fluxdep_fit")
        params_path_row = QHBoxLayout()
        params_path_row.addWidget(self._params_path_edit, stretch=1)
        self._load_btn = QPushButton("Load")
        self._load_btn.clicked.connect(self._on_load_path_clicked)
        params_path_row.addWidget(self._load_btn)
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._on_browse_file)
        params_path_row.addWidget(self._browse_btn)
        # Add the row layout directly: a holder widget's default layout margins
        # squeezed the edit and buttons to a few pixels inside the form row.
        model_form.addRow("params.json:", params_path_row)

        self._add_energy_fields(model_form)
        return model_group

    def _add_energy_fields(self, model_form: QFormLayout) -> None:
        """Add nullable energy and flux rows, preserving model-dependent visibility."""
        # Energies in GHz; flux anchors and bias in device-value units. Wide
        # ranges + many decimals allow exploration. The params classes own physics
        # validation; unchecked fields remain unknown rather than zero-filled.
        self._ej_spin = self._make_param_spin(_DEFAULT_EJ)
        self._ec_spin = self._make_param_spin(_DEFAULT_EC)
        self._el_spin = self._make_param_spin(_DEFAULT_EL)
        self._fluxonium_energies_label = QLabel("Energies:")
        self._fluxonium_energies_row = self._make_param_row(
            ("EJ", self._ej_spin),
            ("EC", self._ec_spin),
            ("EL", self._el_spin),
        )
        model_form.addRow(self._fluxonium_energies_label, self._fluxonium_energies_row)
        # SQUID transmon: EJmax/EC in GHz, d is the dimensionless asymmetry.
        self._ejmax_spin = self._make_param_spin(_DEFAULT_EJMAX)
        self._transmon_ec_spin = self._make_param_spin(_DEFAULT_TRANSMON_EC)
        self._asymmetry_spin = self._make_param_spin(_DEFAULT_ASYMMETRY)
        self._transmon_energies_label = QLabel("Energies:")
        self._transmon_energies_row = self._make_param_row(
            ("EJmax", self._ejmax_spin),
            ("EC", self._transmon_ec_spin),
            ("d", self._asymmetry_spin),
        )
        model_form.addRow(self._transmon_energies_label, self._transmon_energies_row)
        self._model_combo.currentIndexChanged.connect(self._sync_model_rows)
        self._sync_model_rows()
        self._flux_half_spin = self._make_param_spin(_DEFAULT_FLUX_HALF)
        self._flux_period_spin = self._make_param_spin(_DEFAULT_FLUX_PERIOD)
        # flux_bias lives here (was in the old "Load predictor" group).
        self._flux_bias_spin = self._make_param_spin(_DEFAULT_FLUX_BIAS)
        model_form.addRow(
            "Flux:",
            self._make_param_row(
                ("half", self._flux_half_spin),
                ("period", self._flux_period_spin),
                ("bias", self._flux_bias_spin),
            ),
        )

    def _build_transitions_group(self) -> QGroupBox:
        """Build device/transition control rows and the shared multi-selection result table."""
        # ── Tracked transitions group (predict controls + table + remove) ──
        transitions_group = QGroupBox("Tracked transitions")
        transitions_layout = QVBoxLayout(transitions_group)

        # Controls above the table.  Keep device value/device selector and
        # transition-add controls on separate rows so long device names do not
        # force the whole left column wider than the plot area needs.
        transitions_layout.addLayout(self._build_value_controls())
        transitions_layout.addLayout(self._build_transition_controls())

        # 4-column table: Transition | f (MHz) | |n| | |phi|
        # Multi-row selection enabled; individual delete buttons are gone.
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Transition", "f (MHz)", "|n|", "|phi|"])
        header = self._table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        transitions_layout.addWidget(self._table)

        return transitions_group

    def _build_value_controls(self) -> QHBoxLayout:
        """Build the marker value and optional device selector/read controls."""
        self._value_controls_row = QHBoxLayout()

        self._value_controls_row.addWidget(QLabel("Device value:"))
        self._predict_value_spin = TrimDoubleSpinBox()
        self._predict_value_spin.setRange(-1e6, 1e6)
        self._predict_value_spin.setDecimals(6)
        self._predict_value_spin.setValue(0.0)
        self._predict_value_spin.valueChanged.connect(self._on_spinbox_changed)
        self._value_controls_row.addWidget(self._predict_value_spin)

        self._device_combo: QComboBox | None = None
        self._device_refresh_btn: QPushButton | None = None
        self._device_read_btn: QPushButton | None = None
        if self._dev is not None:
            device_combo = QComboBox()
            device_combo.setMinimumWidth(160)
            device_combo.setMinimumContentsLength(16)
            device_combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            device_combo.currentIndexChanged.connect(self._on_device_selection_changed)
            self._value_controls_row.addWidget(device_combo)
            self._device_combo = device_combo

            device_refresh_btn = QPushButton("↻")
            device_refresh_btn.setFixedWidth(28)
            device_refresh_btn.setToolTip("Refresh device list and cached value")
            device_refresh_btn.clicked.connect(self._on_refresh_devices_clicked)
            self._value_controls_row.addWidget(device_refresh_btn)
            self._device_refresh_btn = device_refresh_btn

            device_read_btn = QPushButton("Use")
            device_read_btn.setToolTip("Read selected device value")
            device_read_btn.clicked.connect(self._on_read_device_clicked)
            self._value_controls_row.addWidget(device_read_btn)
            self._device_read_btn = device_read_btn

        self._value_controls_row.addStretch()
        return self._value_controls_row

    def _build_transition_controls(self) -> QHBoxLayout:
        """Build the authored from/to pair and its explicit Add action."""
        self._transition_controls_row = QHBoxLayout()
        self._transition_controls_row.addWidget(QLabel("Transition:"))
        self._transition_controls_row.addWidget(QLabel("from"))
        self._add_from_spin = QSpinBox()
        self._add_from_spin.setRange(0, 20)
        self._add_from_spin.setValue(0)
        self._transition_controls_row.addWidget(self._add_from_spin)
        self._transition_controls_row.addWidget(QLabel("to"))
        self._add_to_spin = QSpinBox()
        self._add_to_spin.setRange(0, 20)
        self._add_to_spin.setValue(1)
        self._transition_controls_row.addWidget(self._add_to_spin)
        self._add_btn = QPushButton("Add")
        self._add_btn.clicked.connect(self._on_add_clicked)
        self._transition_controls_row.addWidget(self._add_btn)

        self._transition_controls_row.addStretch()
        return self._transition_controls_row

    def _build_calibration_group(self) -> QGroupBox:
        """Build expression-bearing calibration controls for the selected transition."""
        # ── Calibration: measured frequency -> predictor flux_bias ────────
        calibration_group = QGroupBox("Calibration")
        calibration_form = QFormLayout(calibration_group)
        self._calibration_freq_field = EvalNumericField(
            minimum=-1e9,
            maximum=1e9,
            decimals=6,
            parameter_provider=self._parameter_provider,
        )
        self._calibration_freq_field.load_expression(
            _DEFAULT_CALIBRATION_EXPR, direct_fallback=0.0
        )
        calibration_form.addRow("Frequency (MHz):", self._calibration_freq_field)

        self._calibrate_btn = QPushButton("Calibrate")
        self._calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        self._calibration_freq_field.raw_changed.connect(self._sync_calibration_button)
        calibration_form.addRow("Selected transition:", self._calibrate_btn)
        return calibration_group

    def _build_canvases(self) -> QTabWidget:
        """Build independent plot tabs with shared marker callbacks and synchronized x limits."""
        # ── Right: QTabWidget with three canvases ─────────────────────────
        # Frequency, |n|, |phi|. All three share one marker value and one set of
        # follow/lock callbacks, so a click-follow on any tab drives the others.
        # Each tab owns a native Matplotlib toolbar; x-ranges stay synced across
        # tabs while y-ranges remain per-tab because their units differ.
        self._tab_widget = QTabWidget()

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
                on_follow=self._on_canvas_follow,
                on_lock=self._on_canvas_lock,
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

        self._tab_widget.addTab(self._freq_tab, "Frequency")
        self._tab_widget.addTab(self._mat_n_tab, "|n|")
        self._tab_widget.addTab(self._mat_phi_tab, "|phi|")

        return self._tab_widget

    def _initialize_trial(self, params: ModelParams | None) -> None:
        """Populate initial data and connect debounced trial updates after widgets exist."""
        # ── Debounce timer for per-row column updates ─────────────────────
        # Used both by the spinbox and by the marker follow-loop so cursor motion
        # updates the spinbox/table/markers at the debounce cadence, not per pixel.
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._update_value_columns)

        # ── Predictor info cache (flux_half, flux_period for conversions) ──
        self._flux_half: float | None = None
        self._flux_period: float | None = None

        # Populate table from default transitions before loading predictor.
        self._rebuild_table()
        self._refresh_device_selector()

        self._parameter_timer = QTimer(self)
        self._parameter_timer.setSingleShot(True)
        self._parameter_timer.setInterval(_DEBOUNCE_MS)
        self._parameter_timer.timeout.connect(self._refresh_curves)
        self._refresh_components()
        self._model_combo.currentIndexChanged.connect(self._on_parameters_changed)
        for spin, known in self._known_fields.items():
            spin.valueChanged.connect(self._on_parameters_changed)
            known.toggled.connect(self._on_parameters_changed)

        if params is not None:
            self.set_parameters(params)
        else:
            self._refresh_curves()
        self._sync_live_mode_controls()
        self._params_path_edit.textChanged.connect(self._emit_state_changed)
        self._tab_widget.currentChanged.connect(self._emit_state_changed)

    def dispose_control_watches(self) -> None:
        """Retire owned authority projections before final view destruction.

        Hiding a persistent dialog is not retirement. Call on the owner thread
        before its parent shuts down or schedules the view's Qt deletion.
        """
        dispose_control_watches(self)

    def deleteLater(self) -> None:
        """Release authority watches before direct owner-thread Qt deletion."""
        self.dispose_control_watches()
        super().deleteLater()

    def parameters(self) -> ModelParams:
        """Return the selected model's local trial: GHz energies and native affine inputs.

        Unchecked fields return None. No source/session read or write occurs.
        Invalid physical values are rejected only when queried.
        """
        affine = (
            self._parameter_value(self._flux_half_spin),
            self._parameter_value(self._flux_period_spin),
            self._parameter_value(self._flux_bias_spin),
        )
        if self._model_combo.currentData() == "transmon":
            return TransmonParams(
                self._parameter_value(self._ejmax_spin),
                self._parameter_value(self._transmon_ec_spin),
                self._parameter_value(self._asymmetry_spin),
                *affine,
            )
        return FluxoniumParams(
            self._parameter_value(self._ej_spin),
            self._parameter_value(self._ec_spin),
            self._parameter_value(self._el_spin),
            *affine,
        )

    def set_parameters(self, params: ModelParams) -> None:
        """Replace the detached trial and select its model, retaining invalid physics.

        Every finite float is retained exactly, including subnormals. None unchecks
        the field; users check and edit it to complete the model.
        Computation validates physics later. If enabled, only user cache is saved.
        """
        self._updating_parameters = True
        try:
            self._model_combo.setCurrentIndex(self._model_combo.findData(params.model))
            if isinstance(params, TransmonParams):
                energies = (
                    (self._ejmax_spin, params.EJmax),
                    (self._transmon_ec_spin, params.EC),
                    (self._asymmetry_spin, params.d),
                )
            else:
                energies = (
                    (self._ej_spin, params.EJ),
                    (self._ec_spin, params.EC),
                    (self._el_spin, params.EL),
                )
            for spin, value in (
                *energies,
                (self._flux_half_spin, params.flux_half),
                (self._flux_period_spin, params.flux_period),
                (self._flux_bias_spin, params.flux_bias),
            ):
                self._known_fields[spin].setChecked(value is not None)
                spin.setEnabled(value is not None and not self._live_mode)
                if value is not None:
                    spin.setValue(value)
                else:
                    spin.clear()
        finally:
            self._updating_parameters = False
        self._parameter_timer.stop()
        self._refresh_curves()
        self._sync_calibration_button()
        self._save_parameters()

    def _sync_model_rows(self, *_args: object) -> None:
        """Show only the selected model's energy inputs; keep hidden values untouched."""
        transmon = self._model_combo.currentData() == "transmon"
        self._fluxonium_energies_label.setVisible(not transmon)
        self._fluxonium_energies_row.setVisible(not transmon)
        self._transmon_energies_label.setVisible(transmon)
        self._transmon_energies_row.setVisible(transmon)

    def selected_component(self) -> str | None:
        """Return the concrete prefill selection, or None with no choices.

        Selection alone never loads parameters or binds the trial to a component.
        """
        return self._component_combo.currentText() or None

    def load_component(self, qubit: str | None = None) -> None:
        """Prefill current-point params via injected params_for; never write back.

        Omit qubit to use the selector (initially alias qubit, when bound).
        Unknown/unsupported component and unavailable context errors propagate.
        Missing fields remain None. Loading is explicit, not live binding.
        """
        name = qubit if qubit is not None else self.selected_component()
        if name is None:
            raise FailedPreconditionError("No qubit component selected")
        params = self._pred.params_for_qubit(name)
        self.set_parameters(params)
        index = self._component_combo.findText(name)
        if index >= 0:
            self._component_combo.setCurrentIndex(index)

    def enable_parameter_cache(self, *, cache_dir: Path | None = None) -> None:
        """Restore and enable versioned user-cache persistence for this trial.

        None uses SingleFileCaretaker's existing zcu_tools/gui user-cache root
        (platformdirs, with a temporary-directory fallback). An explicit directory
        isolates a caller's cache. The file is predictor.json, never a component
        DocumentStore. Enabling restores immediately, replacing current inputs;
        subsequent edits and close save only local model/params via atomic replace.
        Read, version or data errors restore an empty trial and display an error.
        Save I/O errors leave the trial editable and display an error. Calling
        twice raises ValueError rather than silently changing the cache owner.
        """
        if self._caretaker is not None:
            raise ValueError("Parameter cache is already enabled")
        caretaker = SingleFileCaretaker(
            self,
            codec=PredictorSnapshotCodec(),
            filename="predictor.json",
            cache_dir=cache_dir,
        )
        outcome = caretaker.restore_all()
        self._caretaker = caretaker
        if outcome.load_error is not None:
            self._set_status(str(outcome.load_error), error=True)

    def capture_persisted_state(self) -> PredictorSnapshot:
        """Capture detached model/params for user cache, not component storage."""
        params = self.parameters()
        return PredictorSnapshot(model=params.model, params=params)

    def restore_persisted_state(self, state: PredictorSnapshot) -> None:
        """Restore a validated cache snapshot into editable local fields only."""
        self.set_parameters(state.params)

    def _parameter_value(self, spin: FiniteFloatSpinBox) -> float | None:
        return spin.value() if self._known_fields[spin].isChecked() else None

    def _save_parameters(self) -> None:
        if self._caretaker is not None:
            try:
                self._caretaker.flush()
            except PersistenceError as exc:
                self._set_status(str(exc), error=True)

    def _on_parameters_changed(self, *_args: object) -> None:
        if self._updating_parameters:
            return
        for spin, known in self._known_fields.items():
            spin.setEnabled(known.isChecked() and not self._live_mode)
        self._clear_predictor_display()
        self._sync_calibration_button()
        missing = self.parameters().missing()
        if missing:
            self._set_status("Missing parameters: " + ", ".join(missing), error=True)
        self._save_parameters()
        self._parameter_timer.start()

    def _refresh_components(self) -> None:
        previous = self.selected_component()
        self._component_combo.clear()
        try:
            context = self._parameter_provider()
            if context is None:
                return
            names = [
                name
                for name, component in context.document().components.items()
                if component.kind.startswith("qubit/")
            ]
            self._component_combo.addItems(names)
            preferred = previous
            if preferred is None:
                preferred = next(
                    (
                        row.resolved
                        for row in context.resolved_aliases()
                        if row.alias == "qubit"
                    ),
                    None,
                )
            if preferred in names:
                self._component_combo.setCurrentIndex(names.index(preferred))
        except ExpectedError as exc:
            self._set_status(str(exc), error=True)
        finally:
            self._component_load_btn.setEnabled(self.selected_component() is not None)

    def _on_load_component_clicked(self) -> None:
        try:
            self.load_component()
        except (ExpectedError, PredictorInputError) as exc:
            self._set_status(str(exc), error=True)

    def predict_frequency(
        self, value: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """Return MHz from the current trial at a native device value; invalid inputs raise."""
        from zcu_tools.gui.session.services.predictor import PredictFreqRequest

        return self._pred.predict_freq(
            self.parameters(), PredictFreqRequest(value, transition)
        )

    def calibrate(
        self, value: float, frequency_mhz: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """Compute candidate native bias and update only this dialog's bias field.

        Inputs are the measured native value, MHz frequency and increasing level
        pair. Displays actual MHz and clamped status from the model result.
        Validation/numerical failures propagate without edits. No source is written.
        """
        from zcu_tools.gui.session.services.predictor import CalibrateFluxBiasRequest

        result = self._pred.calibrate_flux_bias(
            self.parameters(),
            CalibrateFluxBiasRequest(value, frequency_mhz, transition),
        )
        self._display_calibration(result)
        return result.flux_bias

    def load_parameters(self, path: str) -> None:
        """Read an existing params.json into this local trial, never changing the file.

        PredictorLoadError reports unreadable/missing fluxdep_fit; invalid inputs
        raise ValueError without changing fields. See read_fluxdep_fit_params.
        """
        from zcu_tools.gui.session.services.predictor import read_fluxdep_fit_params

        self.set_parameters(read_fluxdep_fit_params(path))
        self._params_path_edit.setText(path)

    def capture_state(self) -> PredictorDialogState:
        """Return the persistent UI state only; live marker/device/cache state is excluded."""
        return PredictorDialogState(
            tracked_transitions=tuple(self._tracked),
            tab_index=int(self._tab_widget.currentIndex()),
            params_path_text=self._params_path_edit.text(),
        )

    def restore_state(self, state: PredictorDialogState) -> None:
        """Restore display preferences without changing the local trial parameters."""
        transitions = _normalize_tracked_transitions(state.tracked_transitions)
        tab_count = self._tab_widget.count()
        tab_index = max(0, min(int(state.tab_index), max(0, tab_count - 1)))
        self._suppress_state_changed = True
        try:
            self._tracked = list(transitions)
            self._params_path_edit.setText(state.params_path_text)
            self._tab_widget.setCurrentIndex(tab_index)
            self._rebuild_table()
        finally:
            self._suppress_state_changed = False
        self._refresh_curves()
        self._update_value_columns()

    def _emit_state_changed(self, *_args: object) -> None:
        if self._suppress_state_changed:
            return
        self.state_changed.emit()

    def reject(self) -> None:
        self._save_parameters()
        self._parameter_timer.stop()
        self._debounce_timer.stop()
        if self._persistent_on_close:
            self.hide()
            return
        super().reject()

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        self._save_parameters()
        self._parameter_timer.stop()
        self._debounce_timer.stop()
        if self._persistent_on_close:
            if a0 is not None:
                a0.ignore()
            self.hide()
            return
        super().closeEvent(a0)

    def showEvent(self, a0: QShowEvent | None) -> None:
        if a0 is not None:
            super().showEvent(a0)
            if a0.spontaneous():
                return

        self._sync_calibration_button()
        if self._live_mode:
            self._set_predict_value(
                self._predict_value_spin.value(), update_status=False
            )
            self._set_status("Live predictor view", error=False)
            return
        self._refresh_device_selector()
        self._refresh_components()
        selected = self._selected_device_name()
        if selected is not None:
            self._apply_cached_device_value(selected)
            self._request_live_device_refresh(selected)
            return
        self._set_predict_value(self._predict_value_spin.value(), update_status=False)

    def _on_close_requested(self) -> None:
        self.reject()

    def set_live_mode(self, enabled: bool, *, status: str | None = None) -> None:
        """Toggle app-driven live display mode.

        Live mode locks local trial fields and marker interaction, driven by
        the host app, while the user can still inspect the plot/table selection.
        """
        self._live_mode = bool(enabled)
        self._sync_live_mode_controls()
        if self._live_mode:
            self._debounce_timer.stop()
            self._refresh_curves()
            self._set_status(status or "Live predictor view", error=False)
        else:
            self._set_status(status or "Manual predictor control", error=False)

    def set_live_device_value(self, value: float, *, status: str | None = None) -> None:
        """Set the live marker to a host-provided device value."""
        self._set_predict_value(float(value), update_status=False)
        if self._live_mode:
            self._set_status(
                status or f"Live device value: {float(value):.6g}",
                error=False,
            )

    def _sync_live_mode_controls(self) -> None:
        enabled = not self._live_mode
        for widget in (
            self._params_path_edit,
            self._load_btn,
            self._browse_btn,
            self._ej_spin,
            self._ec_spin,
            self._el_spin,
            self._ejmax_spin,
            self._transmon_ec_spin,
            self._asymmetry_spin,
            self._flux_half_spin,
            self._flux_period_spin,
            self._flux_bias_spin,
            self._predict_value_spin,
            self._add_from_spin,
            self._add_to_spin,
            self._add_btn,
            self._remove_btn,
            self._calibration_freq_field,
        ):
            widget.setEnabled(enabled)
        for spin, known in self._known_fields.items():
            known.setEnabled(enabled)
            spin.setEnabled(enabled and known.isChecked())
        self._model_combo.setEnabled(enabled)
        self._component_combo.setEnabled(enabled)
        self._component_refresh_btn.setEnabled(enabled)
        self._component_load_btn.setEnabled(
            enabled and self.selected_component() is not None
        )
        if self._device_combo is not None:
            self._device_combo.setEnabled(enabled)
        if self._device_refresh_btn is not None:
            self._device_refresh_btn.setEnabled(enabled)
        self._update_device_buttons()
        self._sync_calibration_button()
        for canvas in self._all_canvases:
            canvas.set_interaction_enabled(enabled)

    # ------------------------------------------------------------------
    # Model-param widgets / helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_param_spin(default: float = 0.0) -> FiniteFloatSpinBox:
        """Preserve every finite model float; physical validation belongs to the model."""
        spin = FiniteFloatSpinBox()
        spin.setValue(default)
        return spin

    def _make_param_row(self, *pairs: tuple[str, FiniteFloatSpinBox]) -> QWidget:
        holder = QWidget()
        row = QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        for label, spin in pairs:
            known = QCheckBox(label)
            known.setToolTip(
                "Checked: known input. Uncheck for unknown; check to enter a value."
            )
            known.setChecked(True)
            self._known_fields[spin] = known
            row.addWidget(known)
            row.addWidget(spin, stretch=1)
        return holder

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

    def _clear_predictor_display(self) -> None:
        self._synced_xlim = None
        for canvas in self._all_canvases:
            canvas.clear()
        for row_idx in range(self._table.rowCount()):
            for col in (_COL_FREQ, _COL_MAG_N, _COL_MAG_PHI):
                item = self._table.item(row_idx, col)
                if item is not None:
                    item.setText("—")

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
        try:
            model = self.parameters().build_predictor()
        except PredictorInputError:
            return None, None
        self._matrix_supported = isinstance(model, MatrixPredictor)
        return model.value_to_flux, model.flux_to_value

    # ------------------------------------------------------------------
    # Device-value selector helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_numeric_info_value(info: BaseDeviceInfo | None) -> float | None:
        if info is None:
            return None
        raw = getattr(info, "value", None)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        return float(raw)

    def _selected_device_name(self) -> str | None:
        combo = self._device_combo
        if combo is None:
            return None
        data = combo.currentData()
        return data if isinstance(data, str) and data else None

    def _update_device_buttons(self) -> None:
        selected = (not self._live_mode) and self._selected_device_name() is not None
        if self._device_read_btn is not None:
            self._device_read_btn.setEnabled(selected)

    def _refresh_device_selector(self) -> None:
        if self._dev is None or self._device_combo is None:
            return

        combo = self._device_combo
        current_name = self._selected_device_name()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(none)", userData=None)
        try:
            entries = self._dev.list_devices()
            for entry in entries:
                if entry.status != DeviceStatus.CONNECTED.value:
                    continue
                value = self._dev.get_cached_device_value(entry.name)
                if value is None:
                    continue
                unit = self._dev.get_device_unit(entry.name)
                unit_suffix = f" ({unit})" if unit and unit != "none" else ""
                combo.addItem(
                    f"{entry.name}  [{entry.type_name}]{unit_suffix}",
                    userData=entry.name,
                )
        except Exception as exc:
            self._set_status(f"Failed to refresh device list: {exc}", error=True)
        finally:
            idx = combo.findData(current_name) if current_name is not None else 0
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
            self._update_device_buttons()

    def _set_predict_value(self, value: float, *, update_status: bool = True) -> None:
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        self._sync_markers(self._predict_value_spin.value())
        self._debounce_timer.stop()
        self._update_value_columns()
        if update_status:
            self._set_status(f"Device value: {self._predict_value_spin.value():.6g}")

    def _apply_cached_device_value(self, name: str) -> bool:
        if self._dev is None:
            return False
        value = self._dev.get_cached_device_value(name)
        if value is None:
            self._set_status(f"No cached device value for {name!r}.", error=True)
            return False
        self._set_predict_value(value, update_status=False)
        self._set_status(f"Cached device value from {name}: {value:.6g}", error=False)
        return True

    def _request_live_device_refresh(self, name: str) -> None:
        if self._dev is None:
            return
        try:
            if not self._dev.try_poll_device_info(name):
                self._set_status(
                    f"Skipped refresh for {name!r}: a run is active.",
                    error=False,
                )
        except Exception as exc:
            self._set_status(f"Failed to refresh {name!r}: {exc}", error=True)

    def _on_refresh_devices_clicked(self) -> None:
        if self._live_mode:
            self._set_status("Live mode controls the device value.", error=False)
            return
        self._refresh_device_selector()
        selected = self._selected_device_name()
        if selected is None:
            self._set_predict_value(
                self._predict_value_spin.value(), update_status=False
            )
            return
        self._apply_cached_device_value(selected)
        self._request_live_device_refresh(selected)

    def _on_read_device_clicked(self) -> None:
        if self._live_mode:
            self._set_status("Live mode controls the device value.", error=False)
            return
        if self._dev is None:
            return
        selected = self._selected_device_name()
        if selected is None:
            self._set_status("Select a device first.", error=True)
            return
        try:
            info = self._dev.get_device_info(selected)
        except Exception as exc:
            self._set_status(f"Failed to read {selected!r}: {exc}", error=True)
            return
        value = self._extract_numeric_info_value(info)
        if value is None:
            self._set_status(f"Device {selected!r} has no numeric value.", error=True)
            return
        self._set_predict_value(value, update_status=False)
        self._set_status(f"Read device value from {selected}: {value:.6g}", error=False)

    def _on_device_selection_changed(self, _index: int) -> None:
        self._update_device_buttons()
        if self._live_mode:
            return
        selected = self._selected_device_name()
        if selected is None:
            self._set_predict_value(
                self._predict_value_spin.value(), update_status=False
            )
            return
        self._apply_cached_device_value(selected)
        self._request_live_device_refresh(selected)

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _rebuild_table(self) -> None:
        """Rebuild the QTableWidget rows from self._tracked.

        Called after any add/delete operation.  Preserves the previous row
        selection if the transition is still present.
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

            # Column 2: |n| placeholder — populated by _update_value_columns
            mag_n_item = QTableWidgetItem("—")
            self._table.setItem(row_idx, _COL_MAG_N, mag_n_item)

            # Column 3: |phi| placeholder — populated by _update_value_columns
            mag_phi_item = QTableWidgetItem("—")
            self._table.setItem(row_idx, _COL_MAG_PHI, mag_phi_item)

        self._table.resizeColumnsToContents()
        self._table.itemSelectionChanged.connect(self._on_selection_changed)

        # Restore selection for transitions still present.
        for transition in selected:
            if transition in self._tracked:
                row = self._tracked.index(transition)
                self._table.setRangeSelected(
                    QTableWidgetSelectionRange(
                        row, 0, row, self._table.columnCount() - 1
                    ),
                    True,
                )
        self._sync_calibration_button()

    def _selected_transitions(self) -> list[tuple[int, int]]:
        """Return the (from, to) tuples for all currently selected table rows."""
        selected_rows = {idx.row() for idx in self._table.selectedIndexes()}
        return [
            self._tracked[row]
            for row in sorted(selected_rows)
            if 0 <= row < len(self._tracked)
        ]

    def _selected_highlights(self) -> tuple[tuple[int, int], ...]:
        """Return all currently selected transitions for canvas highlighting."""
        return tuple(self._selected_transitions())

    def _resolve_calibration_frequency_mhz(
        self, *, update_status: bool = True
    ) -> float | None:
        from zcu_tools.gui.session.expression import (
            EvalRef,
            coerce_eval_result,
            evaluate_numeric_expr,
        )

        raw = self._calibration_freq_field.read_raw()
        if not isinstance(raw, EvalRef):
            return float(raw)
        try:
            parameters = self._parameter_provider()
            resolved = coerce_eval_result(
                evaluate_numeric_expr(raw.expr, parameters), raw.type_
            )
        except Exception as exc:
            if update_status:
                self._set_status(f"Calibration frequency: {exc}", error=True)
            return None
        if not (raw.minimum <= float(resolved) <= raw.maximum):
            if update_status:
                self._set_status(
                    f"Calibration frequency: value {resolved} out of range",
                    error=True,
                )
            return None
        return float(resolved)

    def _calibration_block_status(self) -> tuple[str, bool] | None:
        if self._live_mode:
            return ("Live mode keeps calibration read-only.", False)
        try:
            self.parameters().build_predictor()
        except PredictorInputError as exc:
            return (str(exc), True)
        selected = self._selected_transitions()
        if len(selected) != 1:
            return ("Select exactly one transition to calibrate.", True)
        if self._resolve_calibration_frequency_mhz(update_status=False) is None:
            return ("Calibration frequency is unresolved.", True)
        return None

    def _sync_calibration_button(self) -> None:
        status = self._calibration_block_status()
        reason = status[0] if status is not None else None
        self._calibrate_btn.setEnabled(reason is None)
        self._calibrate_btn.setToolTip("" if reason is None else reason)

    def _on_calibrate_clicked(self) -> None:
        block_status = self._calibration_block_status()
        if block_status is not None:
            reason, is_error = block_status
            self._set_status(reason, error=is_error)
            self._sync_calibration_button()
            return
        from zcu_tools.gui.session.services.predictor import (
            CalibrateFluxBiasRequest,
            PredictorLoadError,
        )

        frequency_mhz = self._resolve_calibration_frequency_mhz()
        if frequency_mhz is None:
            self._sync_calibration_button()
            return
        selected = self._selected_transitions()
        req = CalibrateFluxBiasRequest(
            value=self._predict_value_spin.value(),
            frequency_mhz=frequency_mhz,
            transition=selected[0],
        )
        try:
            result = self._pred.calibrate_flux_bias(self.parameters(), req)
        except (PredictorLoadError, PredictorInputError) as exc:
            self._set_status(str(exc), error=True)
            return
        except Exception as exc:
            logger.exception("predictor calibration failed")
            self._set_status(str(exc), error=True)
            return
        self._display_calibration(result)
        logger.info("PredictorDialog: calibrated flux bias %r", req)

    def _display_calibration(self, result: CalibrationResult) -> None:
        self._flux_bias_spin.setValue(result.flux_bias)
        self._set_status(
            f"Calibrated flux_bias={result.flux_bias:.6g}; "
            f"{result.frequency_mhz:.6g} MHz; clamped={str(result.clamped).lower()}",
            error=False,
        )
        self._sync_calibration_button()

    def _update_value_columns(self) -> None:
        """Fill f / |n| / |phi| columns for every tracked transition at the marker.

        Called by the debounce timer.  Uses point-predict for each row so each value
        is as accurate as possible.  Expected unavailable-model failures show "—";
        unexpected backend failures fail the whole update once to avoid log storms.
        """
        from zcu_tools.gui.session.services.predictor import (
            PredictFreqRequest,
        )

        marker_value = self._predict_value_spin.value()
        try:
            for row_idx, transition in enumerate(self._tracked):
                # Freq column: per-row scalar predict_freq (most accurate)
                try:
                    freq = self._pred.predict_freq(
                        self.parameters(),
                        PredictFreqRequest(value=marker_value, transition=transition),
                    )
                    freq_text = f"{freq:.4f}"
                except (FailedPreconditionError, PredictorInputError):
                    freq_text = "—"

                freq_item = self._table.item(row_idx, _COL_FREQ)
                if freq_item is not None:
                    freq_item.setText(freq_text)

                # |n| and |phi| columns: one single-point matrix curve call per
                # operator (avoids the level<=1 cap in predict_matrix_element).
                n_item = self._table.item(row_idx, _COL_MAG_N)
                if n_item is not None:
                    n_item.setText(
                        self._point_matrix_text(marker_value, transition, "n")
                    )
                phi_item = self._table.item(row_idx, _COL_MAG_PHI)
                if phi_item is not None:
                    phi_item.setText(
                        self._point_matrix_text(marker_value, transition, "phi")
                    )
        except Exception as exc:
            logger.exception("predictor value-column update failed")
            self._set_status(str(exc), error=True)
            self._set_all_value_columns("—")

    def _set_all_value_columns(self, text: str) -> None:
        for row_idx in range(self._table.rowCount()):
            for col_idx in (_COL_FREQ, _COL_MAG_N, _COL_MAG_PHI):
                item = self._table.item(row_idx, col_idx)
                if item is not None:
                    item.setText(text)

    def _point_matrix_text(
        self, value: float, transition: tuple[int, int], operator: Literal["n", "phi"]
    ) -> str:
        """|<i|operator|j>| at a single device value, formatted, or "—" on expected failure."""
        from zcu_tools.gui.session.services.predictor import (
            PredictMatrixCurveRequest,
        )

        try:
            mat_result = self._pred.predict_matrix_element_curve(
                self.parameters(),
                PredictMatrixCurveRequest(
                    values=np.array([value], dtype=np.float64),
                    transitions=(transition,),
                    operator=operator,
                ),
            )
            return f"{mat_result.mags[0, 0]:.5f}"
        except (FailedPreconditionError, PredictorInputError):
            return "—"

    # ------------------------------------------------------------------
    # Curve refresh
    # ------------------------------------------------------------------

    def _refresh_curves(self) -> None:
        """Recompute and redraw the freq curve and both matrix-element curves."""
        from zcu_tools.gui.session.services.predictor import (
            PredictCurveRequest,
        )

        if not self._tracked:
            for canvas in self._all_canvases:
                canvas.clear()
            return

        value_to_flux, flux_to_value = self._make_affine()
        if value_to_flux is None or flux_to_value is None:
            self._clear_predictor_display()
            try:
                self.parameters().build_predictor()
            except PredictorInputError as exc:
                self._set_status(str(exc), error=True)
            return

        # Build value grid from the default flux window.
        v_lo = flux_to_value(_DEFAULT_FLUX_WINDOW[0])
        v_hi = flux_to_value(_DEFAULT_FLUX_WINDOW[1])
        if v_lo > v_hi:
            v_lo, v_hi = v_hi, v_lo
        grid = np.linspace(v_lo, v_hi, _CURVE_GRID_N, dtype=np.float64)

        highlights = self._selected_highlights()
        marker_value = self._predict_value_spin.value()

        # ── Frequency curves ──────────────────────────────────────────────
        try:
            freq_result = self._pred.predict_freq_curve(
                self.parameters(),
                PredictCurveRequest(values=grid, transitions=tuple(self._tracked)),
            )
            self._last_freq_result = freq_result
            self._freq_canvas.render_curves(
                values=freq_result.values,
                labels=freq_result.labels,
                series=freq_result.freqs_mhz,
                ylabel="Frequency (MHz)",
                highlights=highlights,
                marker_value=marker_value,
                flux_window=_DEFAULT_FLUX_WINDOW,
                value_to_flux=value_to_flux,
                flux_to_value=flux_to_value,
            )
        except (FailedPreconditionError, PredictorInputError) as exc:
            self._set_status(str(exc), error=True)
            return
        except Exception as exc:
            logger.exception("predictor frequency curve refresh failed")
            self._set_status(str(exc), error=True)
            return

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
                operator,
                canvas,
                grid,
                highlights,
                marker_value,
                flux_to_value,
                value_to_flux,
            )
            for operator, canvas in matrix_canvases
        ):
            return

        self._apply_synced_xlim_to_all()

        # Refresh all value columns for the new marker position.
        self._update_value_columns()

    def _sync_matrix_capability(self) -> None:
        """Hide matrix tabs/columns for models without optional matrix elements."""
        supported = self._matrix_supported
        for index, canvas, column in (
            (1, self._mat_n_canvas, _COL_MAG_N),
            (2, self._mat_phi_canvas, _COL_MAG_PHI),
        ):
            self._tab_widget.setTabEnabled(index, supported)
            self._table.setColumnHidden(column, not supported)
            if not supported:
                canvas.clear()
        if not supported and self._tab_widget.currentIndex() != 0:
            self._tab_widget.setCurrentIndex(0)

    def _render_matrix_curve(
        self,
        operator: Literal["n", "phi"],
        canvas: PredictorCurveCanvas,
        grid: np.ndarray,
        highlights: tuple[tuple[int, int], ...],
        marker_value: float,
        flux_to_value: Callable[[float], float],
        value_to_flux: Callable[[float], float],
    ) -> bool:
        """Compute+render one operator's matrix-element curve onto ``canvas``.

        Returns True on success; on an expected failure it sets the status and
        returns False so the caller can stop the refresh.
        """
        from zcu_tools.gui.session.services.predictor import (
            PredictMatrixCurveRequest,
        )

        try:
            mat_result = self._pred.predict_matrix_element_curve(
                self.parameters(),
                PredictMatrixCurveRequest(
                    values=grid,
                    transitions=tuple(self._tracked),
                    operator=operator,
                ),
            )
        except (FailedPreconditionError, PredictorInputError) as exc:
            self._set_status(str(exc), error=True)
            return False
        except Exception as exc:
            logger.exception(
                "predictor matrix curve refresh failed: operator=%s", operator
            )
            self._set_status(str(exc), error=True)
            return False

        if operator == "n":
            self._last_mat_n_result = mat_result
        else:
            self._last_mat_phi_result = mat_result
        canvas.render_curves(
            values=mat_result.values,
            labels=mat_result.labels,
            series=mat_result.mags,
            ylabel=f"|<i|{operator}|j>|",
            highlights=highlights,
            marker_value=marker_value,
            flux_window=_DEFAULT_FLUX_WINDOW,
            value_to_flux=value_to_flux,
            flux_to_value=flux_to_value,
        )
        return True

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
        self._emit_state_changed()

    def _on_add_clicked(self) -> None:
        """Validate and add the transition from the add-row spinboxes."""
        if self._live_mode:
            self._set_status(
                "Live mode keeps tracked transitions read-only.", error=False
            )
            return
        frm = self._add_from_spin.value()
        to = self._add_to_spin.value()
        if self._add_transition(frm, to):
            self._rebuild_table()
            self._refresh_curves()
            self._emit_state_changed()

    def _on_remove_selected(self) -> None:
        """Remove ALL currently selected transitions and refresh."""
        if self._live_mode:
            self._set_status(
                "Live mode keeps tracked transitions read-only.", error=False
            )
            return
        to_remove = self._selected_transitions()
        if not to_remove:
            return
        for transition in to_remove:
            if transition in self._tracked:
                self._tracked.remove(transition)
        self._rebuild_table()
        self._refresh_curves()
        self._emit_state_changed()

    # ------------------------------------------------------------------
    # Spinbox / canvas bidirectional coupling
    # ------------------------------------------------------------------

    def _sync_markers(self, value: float) -> None:
        """Move the marker on every canvas to ``value`` (no recompute)."""
        for canvas in self._all_canvases:
            canvas.set_marker(value)

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

    def _on_spinbox_changed(self, value: float) -> None:
        """Spinbox changed → move all canvas markers; schedule debounced column update."""
        if self._live_mode:
            return
        self._sync_markers(value)
        self._debounce_timer.start()

    def _on_canvas_follow(self, value: float) -> None:
        """Marker following the cursor → update spinbox + all markers; debounce recompute.

        Fired on every motion event while the canvas is in follow mode. The
        spinbox value is updated immediately (with signals blocked to avoid a
        re-entrant column update), the markers track the cursor, but the heavy
        recompute (table f/|n|/|phi|) is debounced so motion does not recompute
        on every pixel.
        """
        if self._live_mode:
            return
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        self._sync_markers(value)
        self._debounce_timer.start()

    def _on_canvas_lock(self, value: float) -> None:
        """Marker locked (second click) → final immediate recompute of the columns."""
        if self._live_mode:
            return
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        self._sync_markers(value)
        # Lock triggers an immediate update (cancel any pending debounce first).
        self._debounce_timer.stop()
        self._update_value_columns()

    def _on_selection_changed(self) -> None:
        """Table row selection changed → restyle highlighted curve on all canvases."""
        highlights = self._selected_highlights()
        for canvas in self._all_canvases:
            canvas.set_highlights(highlights)
        self._sync_calibration_button()

    # ------------------------------------------------------------------
    # Browse / Apply handlers
    # ------------------------------------------------------------------

    def _on_load_path_clicked(self) -> None:
        """Read the typed params.json path into local fields only."""
        if self._live_mode:
            self._set_status("Live mode keeps model params read-only.", error=False)
            return
        path = self._params_path_edit.text().strip()
        if not path:
            self._set_status("Enter a params.json path.", error=True)
            return
        self._load_params_file(path)

    def _on_browse_file(self) -> None:
        """Pick a params.json, populate the typed path, then load it.

        Reading changes only the local fields, never the source or session.
        """
        if self._live_mode:
            self._set_status("Live mode keeps model params read-only.", error=False)
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select params.json", "", "JSON files (*.json);;All files (*)"
        )
        if not path or not local_write_allowed(self._control_identity):
            return
        self._params_path_edit.setText(path)
        self._load_params_file(path)

    def _load_params_file(self, path: str) -> bool:
        from zcu_tools.gui.session.services.predictor import (
            PredictorLoadError,
            read_fluxdep_fit_params,
        )

        try:
            params = read_fluxdep_fit_params(path)
        except PredictorLoadError as exc:
            self._set_status(str(exc), error=True)
            return False
        self.set_parameters(params)
        self._set_status("Loaded parameters into simulator", error=False)
        return True

    # ------------------------------------------------------------------
    # Bus subscription
    # ------------------------------------------------------------------

    def _cleanup_subscriptions(self, *_args: object) -> None:
        unsubscribe_device = self._unsubscribe_device_changed
        if unsubscribe_device is not None:
            self._unsubscribe_device_changed = None
            unsubscribe_device()

    def _on_device_changed(self, payload: DeviceChangedPayload) -> None:
        if not self.isVisible():
            return
        if self._live_mode:
            return
        before = self._selected_device_name()
        self._refresh_device_selector()
        selected = self._selected_device_name()
        if selected is None:
            return
        if payload.name is None or payload.name == selected or before != selected:
            self._apply_cached_device_value(selected)

    # ------------------------------------------------------------------
    # Status helper
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False) -> None:
        self._status_label.setText(msg)
        color = "red" if error else "green"
        self._status_label.setStyleSheet(f"color: {color};")

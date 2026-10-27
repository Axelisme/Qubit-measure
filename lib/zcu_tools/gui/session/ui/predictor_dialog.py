"""Independent fluxonium / SQUID transmon simulator dialog; edits and calibration remain local.

The optional device control supplies values only. Entry predictor files are read-only
prefill. No operation installs a model or writes a component or source file.
"""

# tolerated(file-size): Device polling/subscriptions, transition selection/table and
# calibration share marker/debounce and persistent-close ordering. Splitting their
# callbacks would spread that timing across more owners; parameter editing and
# curve-local state already have dedicated widget owners.

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

# tolerated: qtpy exposes Signal at runtime but its QtCore stub does not export it.
from qtpy.QtCore import QTimer, Signal  # pyright: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QCloseEvent, QShowEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTableWidgetSelectionRange,
    QVBoxLayout,
    QWidget,
)

from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.persistence import PersistenceError, SingleFileCaretaker
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.session.ui.predictor_cache import (
    PredictorSnapshot,
    PredictorSnapshotCodec,
)
from zcu_tools.gui.session.ui.predictor_curves import PredictorCurveTabs
from zcu_tools.gui.session.ui.predictor_parameters import PredictorParameterEditor
from zcu_tools.gui.widgets import error_colour, set_foreground, success_colour
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox
from zcu_tools.simulate.parameters import ModelParams
from zcu_tools.simulate.predictor import (
    CalibrationResult,
    PredictorInputError,
)

if TYPE_CHECKING:
    from zcu_tools.device.base import BaseDeviceInfo
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.events import DeviceChangedPayload
    from zcu_tools.gui.session.parameters import ParameterContext
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

logger = logging.getLogger(__name__)

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

_DEFAULT_CALIBRATION_EXPR: str = ""

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
    supplies fixed ParameterContext captures for optional frequency expressions.
    Predictor files come from the current entry through predictor control.
    params optionally prefills complete or partial local inputs. Blank fields are
    unknown (None); nonnumeric or nonfinite text reports a named field error and
    blocks computation. Loading null leaves the field blank and editable.
    Unresolved frequency paths disable calibration without substituting a value.
    persistent_on_close hides instead of finishing, retaining the plotted model.
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
        params: ModelParams | None = None,
    ) -> None:
        super().__init__(parent)
        self._pred = predictor
        self._dev = device
        self._parameter_provider = parameter_provider or (lambda: None)
        self._persistent_on_close = persistent_on_close
        self._live_mode = False
        self._suppress_state_changed = False
        self._caretaker: SingleFileCaretaker[PredictorSnapshot, None] | None = None
        self.setWindowTitle("Predictor")
        self.setMinimumWidth(1000)
        self.setMinimumHeight(480)
        # Open tall enough that the transition table shows several rows.
        self.resize(1000, 700)

        # Mutable list of currently tracked (from, to) transition pairs.
        # Modified only by _add_transition / _delete_transition / _on_remove_selected.
        self._tracked: list[tuple[int, int]] = list(_DEFAULT_TRANSITIONS)

        # Root layout: left curve tabs | right controls column, so Close sits at
        # the dialog's right edge like the other dialogs.
        root_layout = QHBoxLayout(self)

        # ── Right: controls column ────────────────────────────────────────
        controls_widget = QWidget()
        controls_widget.setMinimumWidth(480)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self._parameter_editor = PredictorParameterEditor(
            predictor,
            on_edited=self._on_parameters_changed,
            on_replaced=self._on_parameters_replaced,
            on_status=lambda message, error: self._set_status(message, error=error),
        )
        controls_layout.addWidget(self._parameter_editor)

        controls_layout.addWidget(self._build_transition_group(), stretch=1)

        controls_layout.addWidget(self._build_calibration_group())

        # ── status label ──────────────────────────────────────────────────
        self._status_label = QLabel("")
        make_label_selectable(self._status_label)
        self._status_label.setObjectName("predictor_status")
        controls_layout.addWidget(self._status_label)

        controls_layout.addLayout(self._build_transition_actions())

        self._curve_tabs = PredictorCurveTabs(
            predictor,
            on_follow=self._on_canvas_follow,
            on_lock=self._on_canvas_lock,
            on_status=lambda message, error: self._set_status(message, error=error),
            on_clear_values=lambda: self._set_all_value_columns("—"),
            on_matrix_capability=lambda supported: self._set_matrix_columns_visible(
                supported=supported
            ),
        )

        # Matching minima and stretch split available width evenly, rather than
        # letting the controls' larger sizeHint squeeze the curves on opening.
        self._curve_tabs.setMinimumWidth(480)
        root_layout.addWidget(self._curve_tabs, stretch=1)
        root_layout.addWidget(controls_widget, stretch=1)

        self._initialize_trial(params)

        self._unsubscribe_device_changed: Callable[[], None] | None = (
            device.on_device_changed(self._on_device_changed)
            if device is not None
            else None
        )
        self.finished.connect(self._cleanup_subscriptions)
        self.destroyed.connect(self._cleanup_subscriptions)

    def _build_transition_group(self) -> QGroupBox:
        # ── Tracked transitions group (predict controls + table + remove) ──
        transitions_group = QGroupBox("Tracked transitions")
        transitions_layout = QVBoxLayout(transitions_group)

        transitions_layout.addLayout(self._build_value_controls())

        transitions_layout.addLayout(self._build_transition_controls())

        # 4-column table: Transition | f (MHz) | |n| | |phi|
        # Multi-row selection enabled; individual delete buttons are gone.
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Transition", "f (MHz)", "|n|", "|phi|"])
        header = self._table.horizontalHeader()
        if header is not None:
            # Value columns follow their text so frequencies are never cut to
            # "134...."; |phi| takes the remaining width.
            for column in range(3):
                header.setSectionResizeMode(
                    column, QHeaderView.ResizeMode.ResizeToContents
                )
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        transitions_layout.addWidget(self._table)

        return transitions_group

    def _build_value_controls(self) -> QHBoxLayout:
        # Controls above the table.  Keep device value/device selector and
        # transition-add controls on separate rows so long device names do not
        # force the whole left column wider than the plot area needs.
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

    def _build_transition_actions(self) -> QHBoxLayout:
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
        return action_row

    def _initialize_trial(self, params: ModelParams | None) -> None:
        """Populate table/device choices, then apply the trial after timers exist."""
        # ── Debounce timer for per-row column updates ─────────────────────
        # Used both by the spinbox and by the marker follow-loop so cursor motion
        # updates the spinbox/table/markers at the debounce cadence, not per pixel.
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(_DEBOUNCE_MS)
        self._debounce_timer.timeout.connect(self._update_value_columns)

        # Populate table from default transitions before loading predictor.
        self._rebuild_table()
        self._refresh_device_selector()

        self._parameter_timer = QTimer(self)
        self._parameter_timer.setSingleShot(True)
        self._parameter_timer.setInterval(_DEBOUNCE_MS)
        self._parameter_timer.timeout.connect(self._refresh_curves)
        self._parameter_editor.refresh_files()

        if params is not None:
            self.set_parameters(params)
        else:
            self._refresh_curves()
        self._sync_live_mode_controls()
        self._curve_tabs.currentChanged.connect(self._emit_state_changed)

    def parameters(self) -> ModelParams:
        """Return the selected model's local trial: GHz energies and native affine inputs.

        Blank fields return None. Nonnumeric or nonfinite text raises a named
        PredictorInputError. No source/session read or write occurs.
        Invalid physical values are rejected only when queried.
        """
        return self._parameter_editor.parameters()

    def set_parameters(self, params: ModelParams) -> None:
        """Replace the detached trial and select its model, retaining invalid physics.

        Every finite float is retained exactly, including subnormals. None clears
        the field; users enter a value to complete the model.
        Computation validates physics later. If enabled, only user cache is saved.
        """
        self._parameter_editor.set_parameters(params)

    def selected_file(self) -> str | None:
        """Return the selected entry filename, or None; selection alone never loads."""
        return self._parameter_editor.selected_file()

    def load_file(self, filename: str | None = None) -> None:
        """Prefill this local trial from a current-entry predictor filename.

        Omit filename to load the current selection. Failures from predictor
        control propagate without changing the trial. The source is never written.
        """
        self._parameter_editor.load_file(filename)

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

    def _save_parameters(self) -> None:
        if self._caretaker is not None:
            try:
                self.parameters()
            except PredictorInputError:
                # Invalid drafts stay editable; keep the last serializable cache.
                return
            try:
                self._caretaker.flush()
            except PersistenceError as exc:
                self._set_status(str(exc), error=True)

    def _on_parameters_replaced(self) -> None:
        """Settle one programmatic trial replacement synchronously; failures propagate."""
        self._parameter_timer.stop()
        self._refresh_curves()
        self._sync_calibration_button()
        self._save_parameters()

    def _on_parameters_changed(self) -> None:
        """Clear stale results and save a valid user edit before scheduling refresh."""
        self._parameter_timer.stop()
        self._clear_predictor_display()
        self._sync_calibration_button()
        try:
            missing = self.parameters().missing()
        except PredictorInputError as exc:
            self._set_status(str(exc), error=True)
            return
        if missing:
            self._set_status("Missing parameters: " + ", ".join(missing), error=True)
        else:
            self._set_status("")
        self._save_parameters()
        self._parameter_timer.start()

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

    def capture_state(self) -> PredictorDialogState:
        """Return the persistent UI state only; live marker/device/cache state is excluded."""
        return PredictorDialogState(
            tracked_transitions=tuple(self._tracked),
            tab_index=int(self._curve_tabs.currentIndex()),
        )

    def restore_state(self, state: PredictorDialogState) -> None:
        """Restore display preferences without changing the local trial parameters."""
        transitions = _normalize_tracked_transitions(state.tracked_transitions)
        tab_count = self._curve_tabs.count()
        tab_index = max(0, min(int(state.tab_index), max(0, tab_count - 1)))
        self._suppress_state_changed = True
        try:
            self._tracked = list(transitions)
            self._curve_tabs.setCurrentIndex(tab_index)
            self._rebuild_table()
        finally:
            self._suppress_state_changed = False
        self._refresh_curves()
        self._update_value_columns()

    def _emit_state_changed(self, *_args: object) -> None:
        if self._suppress_state_changed:
            return
        self.state_changed.emit()

    def done(self, a0: int) -> None:
        """Retire calibration callbacks at terminal teardown, not hide/reopen.

        Repeated cleanup is safe. Failures propagate for retry before finishing
        or permitting Qt child deletion.
        """
        self._calibration_freq_field.dispose()
        super().done(a0)

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
        self._parameter_editor.refresh_files()
        selected = self._selected_device_name()
        if selected is not None:
            self._apply_cached_device_value(selected)
            self._request_live_device_refresh(selected)
            return
        self._set_predict_value(self._predict_value_spin.value(), update_status=False)

    def _on_close_requested(self) -> None:
        self.reject()

    def set_live_mode(self, *, enabled: bool, status: str | None = None) -> None:
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
        self._parameter_editor.set_live_mode(enabled=self._live_mode)
        for widget in (
            self._predict_value_spin,
            self._add_from_spin,
            self._add_to_spin,
            self._add_btn,
            self._remove_btn,
            self._calibration_freq_field,
        ):
            widget.setEnabled(enabled)
        if self._device_combo is not None:
            self._device_combo.setEnabled(enabled)
        if self._device_refresh_btn is not None:
            self._device_refresh_btn.setEnabled(enabled)
        self._update_device_buttons()
        self._sync_calibration_button()
        self._curve_tabs.set_interaction_enabled(enabled=enabled)

    def _clear_predictor_display(self) -> None:
        self._curve_tabs.clear_curves()
        self._set_all_value_columns("—")

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
        # tolerated: Device query ports have open failure sets; show status and restore combo signal delivery.
        except Exception as exc:  # noqa: BLE001
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
        self._curve_tabs.set_marker(self._predict_value_spin.value())
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
        # tolerated: Refresh admission/provider failures must become status text, not escape the Qt callback.
        except Exception as exc:  # noqa: BLE001
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
        # tolerated: Driver read failures have no closed exception set; report status without replacing the value.
        except Exception as exc:  # noqa: BLE001
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
        # tolerated: Arbitrary capture/source failures must block calibration and retain the diagnostic message.
        except Exception as exc:  # noqa: BLE001
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
        from zcu_tools.gui.session.services.predictor import CalibrateFluxBiasRequest

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
        except PredictorInputError as exc:
            self._set_status(str(exc), error=True)
            return
        except Exception as exc:
            logger.exception("predictor calibration failed")
            self._set_status(str(exc), error=True)
            return
        self._display_calibration(result)
        logger.info("PredictorDialog: calibrated flux bias %r", req)

    def _display_calibration(self, result: CalibrationResult) -> None:
        self._parameter_editor.set_flux_bias(result.flux_bias)
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
        """Refresh the curve view, then settle table values only after successful rendering."""
        if self._curve_tabs.refresh_curves(
            self.parameters(),
            tuple(self._tracked),
            self._selected_highlights(),
            self._predict_value_spin.value(),
        ):
            self._update_value_columns()

    def _set_matrix_columns_visible(self, *, supported: bool) -> None:
        for column in (_COL_MAG_N, _COL_MAG_PHI):
            self._table.setColumnHidden(column, not supported)

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

    def _on_spinbox_changed(self, value: float) -> None:
        """Spinbox changed → move all canvas markers; schedule debounced column update."""
        if self._live_mode:
            return
        self._curve_tabs.set_marker(value)
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
        self._curve_tabs.set_marker(value)
        self._debounce_timer.start()

    def _on_canvas_lock(self, value: float) -> None:
        """Marker locked (second click) → final immediate recompute of the columns."""
        if self._live_mode:
            return
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        self._curve_tabs.set_marker(value)
        # Lock triggers an immediate update (cancel any pending debounce first).
        self._debounce_timer.stop()
        self._update_value_columns()

    def _on_selection_changed(self) -> None:
        """Table row selection changed → restyle highlighted curve on all canvases."""
        highlights = self._selected_highlights()
        self._curve_tabs.set_highlights(highlights)
        self._sync_calibration_button()

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

    def _set_status(self, msg: str, *, error: bool = False) -> None:
        self._status_label.setText(msg)
        set_foreground(
            self._status_label, error_colour(self) if error else success_colour(self)
        )

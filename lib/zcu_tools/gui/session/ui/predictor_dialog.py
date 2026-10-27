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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

# tolerated: qtpy exposes Signal at runtime but its QtCore stub does not export it.
from qtpy.QtCore import QTimer, Signal  # pyright: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QCloseEvent, QShowEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
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
from zcu_tools.gui.lifetime import LifetimeScope
from zcu_tools.gui.session.persistence import PersistenceError, SingleFileCaretaker
from zcu_tools.gui.session.ui.eval_field import EvalNumericField
from zcu_tools.gui.session.ui.predictor_cache import (
    PredictorSnapshot,
    PredictorSnapshotCodec,
)
from zcu_tools.gui.session.ui.predictor_curves import (
    CurveCoordinates,
    PredictorCurveTabs,
)
from zcu_tools.gui.session.ui.predictor_flux import PredictorFluxEditor
from zcu_tools.gui.session.ui.predictor_parameters import PredictorParameterEditor
from zcu_tools.gui.widgets import error_colour, set_foreground, success_colour
from zcu_tools.gui.widgets.selectable_text import make_label_selectable
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorFlux,
    PredictorSource,
    save_predictor_file,
)
from zcu_tools.simulate.parameters import ModelParams
from zcu_tools.simulate.predictor import (
    CalibrationResult,
    PredictorInputError,
)

if TYPE_CHECKING:
    from zcu_tools.gui.session.device_control import DeviceControlPort
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
        self._scope = LifetimeScope(self)
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
            on_flux_loaded=self._load_flux,
        )
        controls_layout.addWidget(self._parameter_editor)
        self._flux_editor = PredictorFluxEditor(
            self._scope.callback(self._on_parameters_changed)
        )
        self._scope.add_cleanup(self._flux_editor.retire_callbacks)
        controls_layout.addWidget(self._flux_editor)

        controls_layout.addWidget(self._build_transition_group(), stretch=1)

        controls_layout.addWidget(self._build_calibration_group())

        # ── status label ──────────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
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

    def _build_transition_group(self) -> QGroupBox:
        # ── Tracked transitions group (predict controls + table + remove) ──
        transitions_group = QGroupBox("Tracked transitions")
        transitions_layout = QVBoxLayout(transitions_group)

        transitions_layout.addLayout(self._build_value_controls())

        transitions_layout.addLayout(self._build_transition_controls())

        # 4-column table: Transition | f (MHz) | |n| | |phi|
        # Multi-row selection enabled; individual delete buttons are gone.
        self._table = QTableWidget(0, 4)
        self._table.setObjectName("predictor_transitions")
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
        row = QHBoxLayout()
        row.addWidget(QLabel("Axis value:"))
        self._predict_value_spin = TrimDoubleSpinBox()
        self._predict_value_spin.setObjectName("predictor_axis_value")
        self._predict_value_spin.setRange(-1e6, 1e6)
        self._predict_value_spin.setDecimals(6)
        self._predict_value_spin.valueChanged.connect(self._on_spinbox_changed)
        row.addWidget(self._predict_value_spin)
        row.addStretch()
        return row

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
        # ── Calibration: measured frequency -> local offset correction ────────
        calibration_group = QGroupBox("Calibration")
        calibration_form = QFormLayout(calibration_group)
        self._calibration_freq_field = EvalNumericField(
            minimum=-1e9,
            maximum=1e9,
            decimals=6,
            parameter_provider=self._parameter_provider,
        )
        self._calibration_freq_field.set_input_name("predictor_calibration_frequency")
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
        self._save_as_btn = QPushButton("Save as…")
        self._scope.connect(self._save_as_btn.clicked, self._on_save_as_clicked)
        action_row.addWidget(self._save_as_btn)
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
        """Return the selected model's local GHz energy/asymmetry trial.

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

    def flux(self) -> PredictorFlux:
        """Return the detached trial frame; missing/invalid frame raises PredictorInputError."""
        return self._flux_editor.flux()

    def device_values(self) -> dict[str, float]:
        """Return all native trial values; missing/invalid values raise a named error."""
        return self._flux_editor.device_values()

    def set_flux(
        self, flux: PredictorFlux, device_values: Mapping[str, float] | None = None
    ) -> None:
        """Replace the local frame and native values, defaulting to device cache.

        Missing cached devices stay blank and block computation. The first file
        device becomes the axis. No hardware read or source write occurs.
        """
        if device_values is None:
            self._load_flux(flux)
        else:
            self._flux_editor.set_flux(flux, device_values)
        self._on_parameters_replaced()

    def _load_flux(self, flux: PredictorFlux) -> None:
        values: dict[str, float] = {}
        if self._dev is not None:
            for name in flux.period:
                value = self._dev.get_cached_device_value(name)
                if value is not None:
                    values[name] = value
        self._flux_editor.set_flux(flux, values)

    def selected_file(self) -> str | None:
        """Return the selected entry filename, or None; selection alone never loads."""
        return self._parameter_editor.selected_file()

    def load_file(self, filename: str | None = None) -> None:
        """Prefill this local trial from a current-entry predictor filename.

        Omit filename to load the current selection. Failures from predictor
        control propagate without changing the trial. The source is never written.
        """
        self._parameter_editor.load_file(filename)

    def save_as(self, destination: Path) -> None:
        """Write the current energy/frame trial as a v2 predictor at destination.

        Device values and axis selection are display inputs, not file fields.
        Provenance is unknown for this local trial. Invalid trial text or I/O
        failures propagate; no component or source file is implicitly updated.
        """
        params = self.parameters()
        save_predictor_file(
            destination,
            PredictorFile(
                params.model, params, self.flux(), PredictorSource(None, None)
            ),
        )

    def _on_save_as_clicked(self) -> None:
        filename, _filter = QFileDialog.getSaveFileName(
            self, "Save predictor as", "", "Predictor JSON (*.json)"
        )
        if not filename:
            return
        try:
            self.save_as(Path(filename))
        except (OSError, ValueError) as exc:
            self._set_status(str(exc), error=True)
            return
        self._set_status(f"Saved predictor: {filename}")

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
        flux = self.flux() if self._flux_editor.has_frame() else None
        return PredictorSnapshot(
            model=params.model,
            params=params,
            flux=flux.to_dict() if flux is not None else None,
            device_values=self._flux_editor.known_device_values(),
            axis_device=self._flux_editor.axis_device()
            if flux is not None and flux.period
            else None,
        )

    def restore_persisted_state(self, state: PredictorSnapshot) -> None:
        """Restore a validated cache snapshot into editable local fields only."""
        self._flux_editor.set_flux(
            PredictorFlux.from_mapping(state.flux) if state.flux is not None else None,
            state.device_values,
        )
        if state.axis_device is not None:
            self._flux_editor.set_axis_device(state.axis_device)
        self.set_parameters(state.params)

    def _save_parameters(self) -> None:
        if self._caretaker is not None:
            try:
                self.capture_persisted_state()
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
        self._set_status("")
        self._sync_axis_marker()
        self._refresh_curves()
        self._sync_calibration_button()
        self._save_parameters()

    def _on_parameters_changed(self) -> None:
        """Clear stale results and save a valid user edit before scheduling refresh."""
        self._parameter_timer.stop()
        self._sync_axis_marker()
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

    def predict_frequency(self, transition: tuple[int, int] = (0, 1)) -> float:
        """Return MHz at all current native trial values; invalid inputs raise."""
        from zcu_tools.gui.session.services.predictor import PredictFreqRequest

        return self._pred.predict_freq(
            self.parameters(),
            PredictFreqRequest(
                self.flux().to_frame(), self.device_values(), transition
            ),
        )

    def calibrate(
        self, frequency_mhz: float, transition: tuple[int, int] = (0, 1)
    ) -> CalibrationResult:
        """Add a measured transition's Phi0 correction to the current local offset.

        frequency_mhz is measured MHz; transition is an increasing level pair.
        All current native device values and the current frame enter the query.
        Return correction, actual MHz and clamped. Failures leave inputs unchanged.
        Neither the component nor the source predictor file is written.
        """
        from zcu_tools.gui.session.services.predictor import CalibrateOffsetRequest

        result = self._pred.calibrate_offset(
            self.parameters(),
            CalibrateOffsetRequest(
                self.flux().to_frame(), self.device_values(), frequency_mhz, transition
            ),
        )
        self._display_calibration(result)
        return result

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

    def reject(self) -> None:
        self._save_parameters()
        self._parameter_timer.stop()
        self._debounce_timer.stop()
        if self._persistent_on_close:
            self.hide()
            return
        self._scope.close()
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
        self._scope.close()
        super().closeEvent(a0)

    def showEvent(self, a0: QShowEvent | None) -> None:
        if a0 is not None:
            super().showEvent(a0)
            if a0.spontaneous():
                return

        self._sync_calibration_button()
        self._sync_axis_marker()
        self._curve_tabs.set_marker(self._predict_value_spin.value())
        self._update_value_columns()
        if self._live_mode:
            self._set_status("Live predictor view", error=False)
            return
        self._parameter_editor.refresh_files()

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
        self._flux_editor.setEnabled(enabled)
        self._save_as_btn.setEnabled(enabled)
        self._sync_calibration_button()
        self._curve_tabs.set_interaction_enabled(enabled=enabled)

    def _clear_predictor_display(self) -> None:
        self._curve_tabs.clear_curves()
        self._set_all_value_columns("—")

    # ------------------------------------------------------------------
    # Device-value selector helpers
    # ------------------------------------------------------------------

    def _set_predict_value(self, value: float, *, update_status: bool = True) -> None:
        self._set_axis_input(value)
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        self._curve_tabs.set_marker(self._predict_value_spin.value())
        self._debounce_timer.stop()
        self._update_value_columns()
        if update_status:
            self._set_status(f"Device value: {self._predict_value_spin.value():.6g}")

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
            self.flux()
            self.device_values()
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
        frequency_mhz = self._resolve_calibration_frequency_mhz()
        if frequency_mhz is None:
            return
        try:
            self.calibrate(frequency_mhz, self._selected_transitions()[0])
        except PredictorInputError as exc:
            self._set_status(str(exc), error=True)
        except Exception as exc:
            logger.exception("predictor calibration failed")
            self._set_status(str(exc), error=True)

    def _display_calibration(self, result: CalibrationResult) -> None:
        self._flux_editor.set_offset(self.flux().offset + result.offset_correction)
        self._set_status(
            f"Calibrated offset={self.flux().offset:.6g} Φ0 "
            f"(correction {result.offset_correction:+.6g} Φ0); "
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
                        PredictFreqRequest(
                            self.flux().to_frame(), self.device_values(), transition
                        ),
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
                    frame=self.flux().to_frame(),
                    axis_device=self._flux_editor.axis_device(),
                    device_values=self.device_values(),
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
        try:
            coordinates = CurveCoordinates(
                self.flux().to_frame(),
                self._flux_editor.axis_device(),
                self.device_values(),
            )
            params = self.parameters()
        except PredictorInputError as exc:
            self._clear_predictor_display()
            self._set_status(str(exc), error=True)
            return
        if self._curve_tabs.refresh_curves(
            params,
            tuple(self._tracked),
            self._selected_highlights(),
            self._predict_value_spin.value(),
            coordinates=coordinates,
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

    def _set_axis_input(self, value: float) -> None:
        try:
            name = self._flux_editor.axis_device()
        except PredictorInputError:
            return
        self._flux_editor.set_device_value(name, value, notify=False)

    def _sync_axis_marker(self) -> None:
        try:
            name = self._flux_editor.axis_device()
            value = self.device_values()[name]
        except PredictorInputError:
            return
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)

    def _on_spinbox_changed(self, value: float) -> None:
        """Spinbox changed → move all canvas markers; schedule debounced column update."""
        if self._live_mode:
            return
        self._set_axis_input(value)
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
        self._set_axis_input(value)
        self._predict_value_spin.blockSignals(True)
        self._predict_value_spin.setValue(value)
        self._predict_value_spin.blockSignals(False)
        self._curve_tabs.set_marker(value)
        self._debounce_timer.start()

    def _on_canvas_lock(self, value: float) -> None:
        """Marker locked (second click) → final immediate recompute of the columns."""
        if self._live_mode:
            return
        self._set_axis_input(value)
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
    # Status helper
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, *, error: bool = False) -> None:
        self._status_label.setText(msg)
        set_foreground(
            self._status_label, error_colour(self) if error else success_colour(self)
        )

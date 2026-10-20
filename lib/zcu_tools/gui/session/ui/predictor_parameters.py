"""Detached predictor parameter controls and explicit entry-file prefill."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from zcu_tools.gui.expected_error import ExpectedError, FailedPreconditionError
from zcu_tools.gui.widgets.spinbox import FiniteFloatSpinBox
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.parameters import ModelParams
from zcu_tools.simulate.transmon import TransmonParams

if TYPE_CHECKING:
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

logger = logging.getLogger(__name__)

# Initial standalone trial values, not defaults for missing file inputs.
_DEFAULT_EJ = 4.0
_DEFAULT_EC = 1.0
_DEFAULT_EL = 1.0
_DEFAULT_EJMAX = 20.0
_DEFAULT_TRANSMON_EC = 0.25
_DEFAULT_ASYMMETRY = 0.0
_DEFAULT_FLUX_HALF = 0.0
_DEFAULT_FLUX_PERIOD = 0.005
_DEFAULT_FLUX_BIAS = 0.0


class PredictorParameterEditor(QGroupBox):
    """Own nullable model controls, local edit locks and explicit prefill.

    predictor supplies current-entry filenames and read-only predictor inputs.
    No trial writes back to the source file or a component.
    All callbacks run synchronously on the Qt owner thread, with no signal relay:
    on_edited runs after a user edit's local enabled-state update; on_replaced
    runs once after a whole set_parameters/prefill, with field reentry suppressed.
    The caller owns refresh/debounce/cache ordering in those callbacks. Their
    failures propagate for programmatic calls, including file prefill.
    on_status receives (message, is_error).
    """

    def __init__(
        self,
        predictor: PredictorControlPort,
        *,
        on_edited: Callable[[], None],
        on_replaced: Callable[[], None],
        on_status: Callable[[str, bool], None],
    ) -> None:
        super().__init__("Model params")
        self._pred = predictor
        self._on_edited = on_edited
        self._on_replaced = on_replaced
        self._on_status = on_status
        self._live_mode = False
        self._updating_parameters = False
        self._known_fields: dict[FiniteFloatSpinBox, QCheckBox] = {}
        self._build_controls()
        self._model_combo.currentIndexChanged.connect(self._on_parameters_changed)
        for spin, known in self._known_fields.items():
            spin.valueChanged.connect(self._on_parameters_changed)
            known.toggled.connect(self._on_parameters_changed)

    def _build_controls(self) -> None:
        # ── Model params (editable EJ/EC/EL/flux anchors/flux_bias) ───────
        # File loading only prefills independent trial parameters.
        model_form = QFormLayout(self)

        self._add_model_sources(model_form)

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

    def _add_model_sources(self, model_form: QFormLayout) -> None:
        self._model_combo = QComboBox()
        self._model_combo.addItem("Fluxonium", "fluxonium")
        self._model_combo.addItem("SQUID transmon", "transmon")
        model_form.addRow("Model:", self._model_combo)
        self._file_combo = QComboBox()
        self._file_combo.setObjectName("predictor_file")
        self._file_combo.setMinimumContentsLength(12)
        file_row = QHBoxLayout()
        file_row.addWidget(self._file_combo, stretch=1)
        self._file_refresh_btn = QPushButton("Refresh files")
        self._file_refresh_btn.setToolTip(
            "Refresh entry predictor filenames without loading"
        )
        self._file_refresh_btn.clicked.connect(self.refresh_files)
        file_row.addWidget(self._file_refresh_btn)
        self._file_load_btn = QPushButton("Load")
        self._file_load_btn.clicked.connect(self._on_load_file_clicked)
        file_row.addWidget(self._file_load_btn)
        model_form.addRow("Predictor file:", file_row)
        self._file_hint = QLabel()
        self._file_hint.setWordWrap(True)
        model_form.addRow(self._file_hint)

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
        Computation validates physics later. Calls on_replaced exactly once after
        all edits, synchronously; callback failures propagate to the caller.
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
        self._on_replaced()

    def _sync_model_rows(self, *_args: object) -> None:
        """Show only the selected model's energy inputs; keep hidden values untouched."""
        transmon = self._model_combo.currentData() == "transmon"
        self._fluxonium_energies_label.setVisible(not transmon)
        self._fluxonium_energies_row.setVisible(not transmon)
        self._transmon_energies_label.setVisible(transmon)
        self._transmon_energies_row.setVisible(transmon)

    def selected_file(self) -> str | None:
        """Return the selected entry filename, or None; selection alone never loads."""
        return self._file_combo.currentText() or None

    def load_file(self, filename: str | None = None) -> None:
        """Prefill this local trial from a current-entry predictor filename.

        Omit filename to load the current selection. Failures from predictor
        control propagate without changing the trial. The source is never written.
        """
        name = filename if filename is not None else self.selected_file()
        if name is None:
            raise FailedPreconditionError("No predictor file selected")
        predictor = self._pred.load_file(name)
        self.set_parameters(predictor.params)
        index = self._file_combo.findText(name)
        if index >= 0:
            self._file_combo.setCurrentIndex(index)

    def _parameter_value(self, spin: FiniteFloatSpinBox) -> float | None:
        return spin.value() if self._known_fields[spin].isChecked() else None

    def _on_parameters_changed(self, *_args: object) -> None:
        if self._updating_parameters:
            return
        for spin, known in self._known_fields.items():
            spin.setEnabled(known.isChecked() and not self._live_mode)
        self._on_edited()

    def refresh_files(self) -> None:
        """Refresh filenames without loading; retain any still-existing selection."""
        previous = self.selected_file()
        self._file_combo.clear()
        try:
            names = self._pred.list_files()
            self._file_combo.addItems(names)
            if previous in names:
                self._file_combo.setCurrentIndex(names.index(previous))
            self._file_hint.setText(
                ""
                if names
                else "Open an entry with predictors/*.json, or add predictor files."
            )
        except (ExpectedError, OSError) as exc:
            self._file_hint.setText(str(exc))
            logger.exception("predictor file listing failed")
        self._file_combo.setEnabled(
            not self._live_mode and self.selected_file() is not None
        )
        self._file_load_btn.setEnabled(
            not self._live_mode and self.selected_file() is not None
        )

    def _on_load_file_clicked(self) -> None:
        try:
            self.load_file()
        except (ExpectedError, OSError, ValueError) as exc:
            self._set_status(str(exc), error=True)
        else:
            self._set_status("Loaded predictor file into local trial", error=False)

    def set_flux_bias(self, value: float) -> None:
        """Edit native bias like a spin edit, notifying on_edited if its value changes.

        Known/unknown state is retained. FiniteFloatSpinBox rejects nonfinite input.
        """
        self._flux_bias_spin.setValue(value)

    def set_live_mode(self, *, enabled: bool) -> None:
        """Lock/unlock all model and file controls, retaining nullable inputs."""
        self._live_mode = enabled
        editable = not enabled
        for spin, known in self._known_fields.items():
            known.setEnabled(editable)
            spin.setEnabled(editable and known.isChecked())
        self._model_combo.setEnabled(editable)
        self._file_combo.setEnabled(editable and self.selected_file() is not None)
        self._file_refresh_btn.setEnabled(editable)
        self._file_load_btn.setEnabled(editable and self.selected_file() is not None)

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
            editor = spin.lineEdit()
            if editor is not None:
                editor.setObjectName(f"predictor_input_{label}")
            known = QCheckBox(label)
            known.setToolTip(
                "Checked: known input. Uncheck for unknown; check to enter a value."
            )
            known.setChecked(True)
            self._known_fields[spin] = known
            row.addWidget(known)
            row.addWidget(spin, stretch=1)
        return holder

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._on_status(message, error)

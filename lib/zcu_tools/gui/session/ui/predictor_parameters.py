"""Detached predictor parameter controls and explicit component/file prefill."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from zcu_tools.gui.expected_error import ExpectedError, FailedPreconditionError
from zcu_tools.gui.widgets.spinbox import FiniteFloatSpinBox
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.parameters import ModelParams
from zcu_tools.simulate.predictor import PredictorInputError
from zcu_tools.simulate.transmon import TransmonParams

if TYPE_CHECKING:
    from zcu_tools.gui.session.parameters import ParameterContext
    from zcu_tools.gui.session.predictor_control import PredictorControlPort

# Initial standalone trial values, not defaults for missing component inputs.
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

    predictor supplies read-only component queries; parameter_provider supplies
    captures for the component selector. No trial writes back to either source.
    All callbacks run synchronously on the Qt owner thread, with no signal relay:
    on_edited runs after a user edit's local enabled-state update; on_replaced
    runs once after a whole set_parameters/prefill, with field reentry suppressed.
    The caller owns refresh/debounce/cache ordering in those callbacks. Their
    failures propagate for programmatic calls, including file/component prefill.
    on_status receives (message, is_error); on_path_changed has no payload and
    reports only path text changes, not trial edits.
    """

    def __init__(
        self,
        predictor: PredictorControlPort,
        parameter_provider: Callable[[], ParameterContext | None],
        *,
        on_edited: Callable[[], None],
        on_replaced: Callable[[], None],
        on_status: Callable[[str, bool], None],
        on_path_changed: Callable[[], None],
    ) -> None:
        super().__init__("Model params")
        self._pred = predictor
        self._parameter_provider = parameter_provider
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
        self._params_path_edit.textChanged.connect(on_path_changed)

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

        self._component_combo = QComboBox()
        self._component_combo.setMinimumContentsLength(12)
        component_row = QHBoxLayout()
        component_row.addWidget(self._component_combo, stretch=1)
        self._component_refresh_btn = QPushButton("↻")
        self._component_refresh_btn.setToolTip(
            "Refresh component choices without loading parameters"
        )
        self._component_refresh_btn.clicked.connect(self.refresh_components)
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

    def _parameter_value(self, spin: FiniteFloatSpinBox) -> float | None:
        return spin.value() if self._known_fields[spin].isChecked() else None

    def _on_parameters_changed(self, *_args: object) -> None:
        if self._updating_parameters:
            return
        for spin, known in self._known_fields.items():
            spin.setEnabled(known.isChecked() and not self._live_mode)
        self._on_edited()

    def refresh_components(self) -> None:
        """Refresh choices and alias default, retaining selection; never load a trial."""
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

    def load_parameters(self, path: str) -> None:
        """Read an existing params.json into this local trial, never changing the file.

        PredictorLoadError reports unreadable/missing fluxdep_fit; invalid inputs
        raise ValueError without changing fields. See read_fluxdep_fit_params.
        """
        from zcu_tools.gui.session.services.predictor import read_fluxdep_fit_params

        self.set_parameters(read_fluxdep_fit_params(path))
        self._params_path_edit.setText(path)

    def path_text(self) -> str:
        """Return the unmodified params.json preference text, without reading a file."""
        return self._params_path_edit.text()

    def set_path_text(self, text: str) -> None:
        """Replace path preference text and notify on_path_changed if it changed."""
        self._params_path_edit.setText(text)

    def set_flux_bias(self, value: float) -> None:
        """Edit native bias like a spin edit, notifying on_edited if its value changes.

        Known/unknown state is retained. FiniteFloatSpinBox rejects nonfinite input.
        """
        self._flux_bias_spin.setValue(value)

    def set_live_mode(self, *, enabled: bool) -> None:
        """Lock/unlock all model and prefill controls, retaining nullable inputs."""
        self._live_mode = enabled
        editable = not enabled
        for widget in (
            self._params_path_edit,
            self._load_btn,
            self._browse_btn,
            self._model_combo,
            self._component_combo,
            self._component_refresh_btn,
        ):
            widget.setEnabled(editable)
        for spin, known in self._known_fields.items():
            known.setEnabled(editable)
            spin.setEnabled(editable and known.isChecked())
        self._component_load_btn.setEnabled(
            editable and self.selected_component() is not None
        )

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
            self,
            "Select params.json",
            "",
            "JSON files (*.json);;All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
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

    def _set_status(self, message: str, *, error: bool = False) -> None:
        self._on_status(message, error)

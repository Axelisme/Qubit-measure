"""Qt-free detached predictor drafts and explicit request assembly.

Setters never notify callers or write to a session, component or source file.
Parsing raises named PredictorInputError failures; control failures propagate.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Literal

import numpy as np

from zcu_tools.analysis.flux import FluxFrame
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.expression import EvalRef, resolve_eval_ref
from zcu_tools.gui.session.services.predictor import (
    CalibrateOffsetRequest,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
)
from zcu_tools.resources.predictor_file import PredictorFlux
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulate.parameters import ModelParams
from zcu_tools.simulate.predictor import CalibrationResult, PredictorInputError
from zcu_tools.simulate.transmon import TransmonParams

if TYPE_CHECKING:
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.types import ParameterContext


@dataclass(frozen=True)
class CalibrationFrequency:
    """Calibration expression outcome, without status updates or logging.

    frequency_mhz is the resolved measured MHz, or None on failure.
    error is the original detailed failure message, or None on success.
    Exactly one field is non-None.
    """

    frequency_mhz: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class CalibrationReadiness:
    """A calibration blocker: reason is the exact UI message; is_error selects red status."""

    reason: str
    is_error: bool


@dataclass(frozen=True)
class CurveCoordinates:
    """Native scan coordinates shared by frequency and matrix plots.

    frame holds Phi0 offset and signed native periods/units. axis_device names
    one frame device. device_values fixes every other device in its native unit.
    """

    frame: FluxFrame
    axis_device: str
    device_values: Mapping[str, float]


@dataclass(frozen=True)
class RowValues:
    """One table row: MHz frequency and dimensionless |n|/|phi| magnitudes.

    freq_mhz, mag_n and mag_phi are None when their expected query failure
    makes that value unavailable. Unexpected failures propagate instead.
    """

    freq_mhz: float | None = None
    mag_n: float | None = None
    mag_phi: float | None = None


@dataclass(frozen=True)
class TrialInputs:
    """One parsed trial snapshot for queries.

    params holds nullable GHz energies and dimensionless asymmetry.
    frame holds Phi0 offset and signed native periods/units.
    device_values supplies every frame device in its native unit.
    axis_device is the selected instrument, or None for a constant-flux frame.
    """

    params: ModelParams
    frame: FluxFrame
    device_values: Mapping[str, float]
    axis_device: str | None

    def curve_coordinates(self) -> CurveCoordinates:
        """Return native scan coordinates; no axis raises PredictorInputError."""
        if self.axis_device is None:
            raise PredictorInputError("Select a horizontal axis instrument")
        return CurveCoordinates(self.frame, self.axis_device, self.device_values)


class PredictorTrial:
    """Own independent model inputs; predictor executes explicit read-only queries."""

    def __init__(self, predictor: PredictorControlPort) -> None:
        self._predictor = predictor
        self._model = "fluxonium"
        self._energy_texts = {
            "fluxonium": {"EJ": "4.0", "EC": "1.0", "EL": "1.0"},
            "transmon": {"EJmax": "20.0", "EC": "0.25", "d": "0.0"},
        }
        self._recorded: PredictorFlux | None = None
        self._offset_text = "0.0"
        self._device_texts: dict[str, str] = {}
        self._axis: str | None = None

    def set_parameters(self, params: ModelParams) -> None:
        """Replace nullable energy/asymmetry inputs, retaining invalid physics."""
        self.select_model(params.model)
        values = (
            {"EJmax": params.EJmax, "EC": params.EC, "d": params.d}
            if isinstance(params, TransmonParams)
            else {"EJ": params.EJ, "EC": params.EC, "EL": params.EL}
        )
        for name, value in values.items():
            self.set_energy_text(
                params.model, name, "" if value is None else repr(value)
            )

    @property
    def model(self) -> str:
        """Selected model name: fluxonium or transmon."""
        return self._model

    def select_model(self, model: str) -> None:
        """Select fluxonium/transmon without clearing either draft; unknown names raise KeyError."""
        if model not in self._energy_texts:
            raise KeyError(model)
        self._model = model

    def energy_text(self, model: str, name: str) -> str:
        """Return raw energy text; model is fluxonium/transmon, name is EJ/EC/EL or EJmax/EC/d.

        Unknown model or field names raise KeyError.
        """
        return self._energy_texts[model][name]

    def set_energy_text(self, model: str, name: str, text: str) -> None:
        """Keep raw text, including invalid drafts; names follow energy_text, else KeyError."""
        if name not in self._energy_texts[model]:
            raise KeyError(name)
        self._energy_texts[model][name] = text

    def energy_error(self, model: str, name: str) -> str | None:
        """Return a named nonnumeric/nonfinite error or None, without changing drafts.

        Blank means unknown and is valid. Names follow energy_text, else KeyError.
        """
        text = self.energy_text(model, name).strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError:
            return f"{name}: enter a finite number or leave blank."
        if not isfinite(value):
            return f"{name}: enter a finite number or leave blank."
        return None

    def parameters(self) -> ModelParams:
        """Parse selected GHz energies/asymmetry; blanks are None, invalid text raises.

        PredictorInputError names the first invalid field. Physics is validated
        only when queried, not when editing or reading nullable parameters.
        """
        values: list[float | None] = []
        for name, text in self._energy_texts[self.model].items():
            if error := self.energy_error(self.model, name):
                raise PredictorInputError(error)
            values.append(float(text) if text.strip() else None)
        if self.model == "transmon":
            return TransmonParams(*values)
        return FluxoniumParams(*values)

    def set_flux(self, flux: PredictorFlux | None, values: Mapping[str, float]) -> None:
        """Replace frame and known native values; ignore extras and leave missing blank.

        None clears the frame and values. Numeric text uses .15g, matching the UI.
        """
        self._recorded = flux
        self._offset_text = format(flux.offset, ".15g") if flux is not None else "0.0"
        self._device_texts = (
            {
                name: format(values[name], ".15g") if name in values else ""
                for name in flux.period
            }
            if flux is not None
            else {}
        )

        self._axis = next(iter(self._device_texts), None)

    @property
    def offset_text(self) -> str:
        """Raw Phi0 offset draft, including blank or invalid text."""
        return self._offset_text

    def set_offset_text(self, text: str) -> None:
        """Keep raw offset text without parsing or notifying."""
        self._offset_text = text

    def set_offset(self, offset: float) -> None:
        """Replace the Phi0 offset draft using .15g text, without notifying."""
        self.set_offset_text(format(offset, ".15g"))

    def device_text(self, name: str) -> str:
        """Return raw native draft for a loaded device; unknown names raise KeyError."""
        return self._device_texts[name]

    def set_device_text(self, name: str, text: str) -> None:
        """Keep a loaded device's raw native draft; unknown names raise KeyError."""
        if name not in self._device_texts:
            raise KeyError(name)
        self._device_texts[name] = text

    def set_device_value(self, name: str, value: float) -> None:
        """Replace a native draft using .15g text; unknown names raise KeyError."""
        self.set_device_text(name, format(value, ".15g"))

    def has_frame(self) -> bool:
        """Return whether a caller has supplied a detached flux frame."""
        return self._recorded is not None

    def known_device_values(self) -> dict[str, float]:
        """Return nonblank native inputs; invalid drafts raise a named input error."""
        return {
            name: _number(text, name)
            for name, text in self._device_texts.items()
            if text.strip()
        }

    def axis_device(self) -> str:
        """Return the selected instrument; no axis raises PredictorInputError."""
        if self._axis is None:
            raise PredictorInputError("Select a horizontal axis instrument")
        return self._axis

    def set_axis_device(self, name: str) -> None:
        """Select a loaded instrument, without notification; unknown names raise ValueError."""
        if name not in self._device_texts:
            raise ValueError(f"Unknown axis device {name!r}")
        self._axis = name

    def flux(self) -> PredictorFlux:
        """Return the frame; missing frame or invalid offset raises PredictorInputError."""
        if self._recorded is None:
            raise PredictorInputError("Load a predictor file to set the flux frame")
        return PredictorFlux(
            _number(self._offset_text, "offset"),
            self._recorded.period,
            self._recorded.unit,
        )

    def device_values(self) -> dict[str, float]:
        """Return all native values; blank/invalid drafts raise a named error."""
        return {name: _number(text, name) for name, text in self._device_texts.items()}

    def inputs(self, *, for_curves: bool = False) -> TrialInputs:
        """Parse a detached query snapshot; invalid drafts raise named errors.

        Scalar queries validate parameters, frame, then native values. Curve
        queries (for_curves=True) validate frame, axis, native values, then
        parameters, preserving the curve status precedence when drafts conflict.
        """
        if for_curves:
            frame = self.flux().to_frame()
            axis = self.axis_device()
            values = self.device_values()
            params = self.parameters()
        else:
            params = self.parameters()
            frame = self.flux().to_frame()
            values = self.device_values()
            axis = self._axis
        return TrialInputs(params, frame, values, axis)

    def calibration_readiness(
        self,
        *,
        live_mode: bool,
        selected: Sequence[tuple[int, int]],
        frequency: float | EvalRef,
        parameter_provider: Callable[[], ParameterContext | None] | None = None,
    ) -> CalibrationReadiness | None:
        """Return the first calibration blocker, or None when ready, without changes.

        live_mode blocks before parsing. selected contains the selected level pairs.
        frequency is measured MHz or a bounded EvalRef in MHz.
        parameter_provider captures expression sources
        only after valid physics, frame, native values and exactly one selection.
        Expected input errors become blockers; unexpected model/control failures
        propagate. Callers own button/status presentation and all notifications.
        """
        if live_mode:
            return CalibrationReadiness("Live mode keeps calibration read-only.", False)
        try:
            self.parameters().build_predictor()
            self.flux()
            self.device_values()
        except PredictorInputError as exc:
            return CalibrationReadiness(str(exc), True)
        if len(selected) != 1:
            return CalibrationReadiness(
                "Select exactly one transition to calibrate.", True
            )
        resolved = self.resolve_calibration_frequency_mhz(frequency, parameter_provider)
        if resolved.frequency_mhz is None:
            return CalibrationReadiness("Calibration frequency is unresolved.", True)
        return None

    def resolve_calibration_frequency_mhz(
        self,
        frequency: float | EvalRef,
        parameter_provider: Callable[[], ParameterContext | None] | None = None,
    ) -> CalibrationFrequency:
        """Resolve measured MHz or a bounded MHz EvalRef without changing drafts.

        Capture parameter_provider only for EvalRef; None permits arithmetic only.
        Expression/coercion/range and arbitrary capture/source failures return the
        original detailed error. No query runs, event emits or logging occurs.
        Callers choose generic readiness or detailed status presentation.
        """
        if not isinstance(frequency, EvalRef):
            return CalibrationFrequency(frequency_mhz=float(frequency))
        try:
            parameters = (
                parameter_provider() if parameter_provider is not None else None
            )
            value = resolve_eval_ref(frequency, parameters)
        # tolerated: Arbitrary capture/source failures must block calibration and retain the diagnostic message.
        except Exception as exc:  # noqa: BLE001
            return CalibrationFrequency(error=str(exc))
        return CalibrationFrequency(frequency_mhz=float(value))

    def calibrate(
        self, frequency_mhz: float, transition: tuple[int, int] = (0, 1)
    ) -> CalibrationResult:
        """Add a measured transition's correction to the local Phi0 offset draft.

        frequency_mhz is measured MHz; transition is an increasing level pair.
        Assemble current parameters, frame and all native values once. Return
        correction, actual MHz and clamped. Input and control failures propagate
        without changing drafts. No component/file writes or events occur.
        Callers must present the new offset before publishing success status.
        """
        inputs = self.inputs()
        result = self._predictor.calibrate_offset(
            inputs.params,
            CalibrateOffsetRequest(
                inputs.frame, inputs.device_values, frequency_mhz, transition
            ),
        )
        self.set_offset(inputs.frame.offset + result.offset_correction)
        return result

    def predict_frequency(self, transition: tuple[int, int] = (0, 1)) -> float:
        """Return MHz for a level pair at current native values; query failures propagate."""
        inputs = self.inputs()
        return self._predictor.predict_freq(
            inputs.params,
            PredictFreqRequest(inputs.frame, inputs.device_values, transition),
        )

    def curve_coordinates(self) -> CurveCoordinates:
        """Return current scan coordinates; invalid drafts or a missing axis raise."""
        return self.inputs(for_curves=True).curve_coordinates()

    def row_values(
        self, transitions: tuple[tuple[int, int], ...], marker_value: float
    ) -> tuple[RowValues, ...]:
        """Query table rows at current native inputs with matrices at the native marker.

        Parse inputs once per batch. Transition order is retained; marker_value
        is the selected axis device's native value. PredictorInputError and
        FailedPreconditionError become None per value; an input parsing failure
        makes every row unavailable. Other exceptions propagate.
        """
        try:
            inputs = self.inputs()
        except (FailedPreconditionError, PredictorInputError):
            return tuple(RowValues() for _ in transitions)
        rows: list[RowValues] = []
        for transition in transitions:
            try:
                freq = self._predictor.predict_freq(
                    inputs.params,
                    PredictFreqRequest(inputs.frame, inputs.device_values, transition),
                )
            except (FailedPreconditionError, PredictorInputError):
                freq = None
            rows.append(
                RowValues(
                    freq,
                    self._matrix_value(inputs, marker_value, transition, "n"),
                    self._matrix_value(inputs, marker_value, transition, "phi"),
                )
            )
        return tuple(rows)

    def _matrix_value(
        self,
        inputs: TrialInputs,
        value: float,
        transition: tuple[int, int],
        operator: Literal["n", "phi"],
    ) -> float | None:
        try:
            coordinates = inputs.curve_coordinates()
            result = self._predictor.predict_matrix_element_curve(
                inputs.params,
                PredictMatrixCurveRequest(
                    coordinates.frame,
                    coordinates.axis_device,
                    np.array([value], dtype=np.float64),
                    coordinates.device_values,
                    (transition,),
                    operator,
                ),
            )
        except (FailedPreconditionError, PredictorInputError):
            return None
        return float(result.mags[0, 0])


def _number(text: str, name: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise PredictorInputError(f"{name}: enter a finite number") from exc
    if not isfinite(value):
        raise PredictorInputError(f"{name}: enter a finite number")
    return value

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from math import isfinite
from typing import ClassVar, Literal, TypedDict, overload

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate._calibration import calibrate_spectrum
from zcu_tools.simulate.fluxonium.prediction import FluxoniumPrediction, MatrixOperator
from zcu_tools.simulate.predictor import (
    CalibrationResult,
    PredictorInputError,
    parameter_number,
)


class FluxoniumParameterValues(TypedDict):
    """Named parameter projection; None denotes an unknown value.

    EJ/EC/EL are energies in GHz. flux_half is the zero-bias half-flux value,
    flux_period is the signed device-value change per quantum, and flux_bias
    is the additive device-value correction. All affine values use native units.
    """

    EJ: float | None
    EC: float | None
    EL: float | None
    flux_half: float | None
    flux_period: float | None
    flux_bias: float | None


@dataclass(frozen=True)
class FluxoniumParams:
    """Independent fluxonium model inputs; None means unknown, never a default.

    EJ, EC and EL are positive finite Josephson, charging and inductive energies
    in GHz. flux_half is the zero-bias half-flux device value; flux_period is
    the finite nonzero device-value change per flux quantum (either sign).
    flux_bias is the finite additive correction to the device value. All three
    affine inputs use the source's native device units. Instances are immutable;
    creating or completing one never changes its component or file source.
    """

    model: ClassVar[Literal["fluxonium"]] = "fluxonium"

    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None
    flux_half: float | None = None
    flux_period: float | None = None
    flux_bias: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> FluxoniumParams:
        """Decode named parameters in the class's GHz/native units, without defaults.

        Omitted/null fields remain unknown. PredictorInputError names unknown keys,
        nonnumeric/bool or nonfinite values. Physical constraints are checked
        by build_predictor, so partial inputs can still be inspected/edited.
        The source mapping is never changed.
        """
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise PredictorInputError(
                "Unknown parameters: " + ", ".join(sorted(unknown))
            )
        return cls(
            EJ=parameter_number(values, "EJ"),
            EC=parameter_number(values, "EC"),
            EL=parameter_number(values, "EL"),
            flux_half=parameter_number(values, "flux_half"),
            flux_period=parameter_number(values, "flux_period"),
            flux_bias=parameter_number(values, "flux_bias"),
        )

    def to_dict(self) -> FluxoniumParameterValues:
        """Return a detached named GHz/native projection, retaining unknown fields."""
        return FluxoniumParameterValues(
            EJ=self.EJ,
            EC=self.EC,
            EL=self.EL,
            flux_half=self.flux_half,
            flux_period=self.flux_period,
            flux_bias=self.flux_bias,
        )

    def missing(self) -> tuple[str, ...]:
        """Return unknown field names in declaration order, without side effects."""
        return tuple(
            field.name for field in fields(self) if getattr(self, field.name) is None
        )

    def build_predictor(self) -> FluxoniumPredictor:
        """Build a fresh MHz-query model, without modifying these parameters.

        PredictorInputError names missing fields, nonfinite inputs, nonpositive energies
        or a zero period. No physical or affine value is filled implicitly.
        """
        missing = self.missing()
        if missing:
            raise PredictorInputError("Missing parameters: " + ", ".join(missing))
        ej, ec, el = self.EJ, self.EC, self.EL
        half, period, bias = self.flux_half, self.flux_period, self.flux_bias
        # Explicit narrowing also keeps the runtime contract valid under python -O.
        if (
            ej is None
            or ec is None
            or el is None
            or half is None
            or period is None
            or bias is None
        ):
            raise PredictorInputError("Missing parameters")
        for name, value in (
            ("EJ", ej),
            ("EC", ec),
            ("EL", el),
            ("flux_half", half),
            ("flux_period", period),
            ("flux_bias", bias),
        ):
            if not isfinite(value):
                raise PredictorInputError(f"{name} must be finite")
            if name in ("EJ", "EC", "EL") and value <= 0:
                raise PredictorInputError(f"{name} must be positive")
        if period == 0:
            raise PredictorInputError("flux_period must be nonzero")
        return FluxoniumPredictor((ej, ec, el), half, period, bias)


class FluxoniumPredictor:
    """Fluxonium queries with GHz energies and native device-value affine inputs.

    params is (EJ, EC, EL); flux_half is the zero-bias half-flux value,
    flux_period is the nonzero signed device-value period and flux_bias is an
    additive device-value correction. Queries do not mutate the model. Legacy
    setters/update_bias change only this instance; clone builds an independent one.
    Frequencies are MHz and matrix magnitudes are dimensionless.
    """

    def __init__(
        self,
        params: tuple[float, float, float],
        flux_half: float,
        flux_period: float,
        flux_bias: float,
    ) -> None:
        self._engine = FluxoniumPrediction(
            params,
            flux_half=flux_half,
            flux_period=flux_period,
            flux_bias=flux_bias,
        )

    @property
    def params(self) -> tuple[float, float, float]:
        """Josephson, charging and inductive energies (EJ, EC, EL), in GHz."""
        return self._engine.params

    @params.setter
    def params(self, new_params: tuple[float, float, float]) -> None:
        self._engine.params = new_params

    @property
    def flux_half(self) -> float:
        """Zero-bias native device value at half a flux quantum."""
        return self._engine.flux_half

    @flux_half.setter
    def flux_half(self, new_flux_half: float) -> None:
        self._engine.flux_half = new_flux_half

    @property
    def flux_period(self) -> float:
        """Signed nonzero native device-value change for one flux quantum."""
        return self._engine.flux_period

    @flux_period.setter
    def flux_period(self, new_flux_period: float) -> None:
        self._engine.flux_period = new_flux_period

    @property
    def flux_bias(self) -> float:
        """Additive native device-value correction; queries never change it."""
        return self._engine.flux_bias

    @flux_bias.setter
    def flux_bias(self, new_flux_bias: float) -> None:
        self._engine.flux_bias = new_flux_bias

    def clone(self) -> FluxoniumPredictor:
        """Return an independent instance with identical energies and affine inputs."""
        return FluxoniumPredictor(
            self.params, self.flux_half, self.flux_period, self.flux_bias
        )

    def value_to_flux(self, cur_value: float) -> float:
        """Map a finite native value to flux quanta, without folding or mutation."""
        return self._engine.value_to_flux(cur_value)

    def _value_to_flux_array(
        self, cur_values: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self._engine.values_to_flux(cur_values)

    def flux_to_value(self, cur_flux: float) -> float:
        """Invert the affine for finite flux quanta; return native device units."""
        return self._engine.flux_to_value(cur_flux)

    def calculate_bias(
        self, cur_value: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> CalibrationResult:
        """Invert the exact spectrum; see Predictor.calculate_bias for result policy.

        Invalid finite inputs or non-increasing/integer levels raise PredictorInputError.
        Searches every branch, choosing smallest |bias| independent of current
        bias. Unreachable targets return closest actual MHz with clamped=true.
        No parameters change, and numerical failures propagate without fallback.
        """
        frm, to = transition
        if type(frm) is not int or type(to) is not int or frm < 0 or to <= frm:
            raise PredictorInputError(
                "Calibration needs nonnegative integer from-level < to-level"
            )
        if not (isfinite(cur_value) and isfinite(cur_freq)):
            raise PredictorInputError("cur_value and cur_freq must be finite")

        def frequencies(fluxs: NDArray[np.float64]) -> NDArray[np.float64]:
            # Convert using this engine's affine; requested flux, not its current
            # bias, determines the spectrum used for inversion.
            values = np.array(
                [self.flux_to_value(float(flux)) for flux in fluxs], dtype=np.float64
            )
            return self._engine.predict_frequencies_mhz(values, (transition,))[1][0]

        zero_bias_flux = (cur_value - self.flux_half) / self.flux_period + 0.5
        return calibrate_spectrum(
            frequencies, zero_bias_flux, self.flux_period, cur_freq
        )

    def update_bias(self, flux_bias: float) -> None:
        """Replace this instance's additive native bias; no source or clone is changed."""
        self._engine = FluxoniumPrediction(
            self.params,
            flux_half=self.flux_half,
            flux_period=self.flux_period,
            flux_bias=flux_bias,
        )

    def _predict_freq(self, cur_value: float, transition: tuple[int, int]) -> float:
        values = np.array([cur_value], dtype=np.float64)
        freq = self._predict_freq_array(values, transition)
        return float(freq[0])

    @overload
    def predict_freq(
        self, cur_value: float, transition: tuple[int, int] = (0, 1)
    ) -> float: ...

    @overload
    def predict_freq(
        self, cur_value: NDArray[np.float64], transition: tuple[int, int] = (0, 1)
    ) -> NDArray[np.float64]: ...

    def predict_freq(
        self,
        cur_value: float | NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
    ) -> float | NDArray[np.float64]:
        """
        Predict the transition frequency of a fluxonium qubit.
        Args:
            cur_value (float): Current value.
            transition (tuple[from, to]): transition between which level
        Returns:
            float: transition frequency in MHz.
        """
        if isinstance(cur_value, Iterable):
            return self._predict_freq_array(np.asarray(cur_value), transition)
        return self._predict_freq(cur_value, transition)

    def _predict_freq_array(
        self,
        cur_values: NDArray[np.float64],
        transition: tuple[int, int],
    ) -> NDArray[np.float64]:
        """Predict one transition over a value array through the shared engine."""

        engine_transition, sign = _frequency_engine_transition(transition)
        _, freqs_mhz = self._engine.predict_frequencies_mhz(
            np.asarray(cur_values, dtype=np.float64),
            (engine_transition,),
            cutoff=40,
        )
        return np.asarray(sign * freqs_mhz[0], dtype=np.float64)

    def predict_frequencies_mhz(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, MHz frequencies), shaped (N,) and (T, N).

        values is a finite 1-D native device-value grid. transitions is a nonempty
        ordered tuple of nonnegative integer (from, to) levels with to >= from.
        Rows/columns preserve request order and duplicates. Invalid transitions
        raise PredictorInputError; numerical errors propagate. Inputs/model are unchanged.
        """
        return self._engine.predict_frequencies_mhz(values, transitions, cutoff=40)

    def predict_matrix_elements(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
        operator: MatrixOperator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, magnitudes), shaped (N,) and (T, N), without mutation.

        values is a finite 1-D native grid; fluxs are flux quanta. transitions is
        nonempty with nonnegative integer levels and to >= from. n is charge,
        phi is phase; dimensionless absolute rows include levels above one.
        Invalid transitions/operators raise PredictorInputError; numerical errors propagate.
        """
        return self._engine.predict_matrix_elements(values, transitions, operator)

    def _predict_matrix_element(
        self,
        cur_value: float,
        transition: tuple[int, int],
        operator: MatrixOperator,
    ) -> float:
        values = np.array([cur_value], dtype=np.float64)
        elems = self._predict_matrix_element_array(values, transition, operator)
        return float(elems[0])

    @overload
    def predict_matrix_element(
        self,
        cur_value: float,
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> float: ...

    @overload
    def predict_matrix_element(
        self,
        cur_value: NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> NDArray[np.float64]: ...

    def predict_matrix_element(
        self,
        cur_value: float | NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> float | NDArray[np.float64]:
        """
        Predict the matrix element of operator between two levels of a fluxonium qubit.
        Args:
            cur_value (float): Current value.
            transition (tuple[from, to]): transition between which level
            operator (str): 'phi' or 'n'
        Returns:
            float: matrix element of operator between two levels.
        """
        if isinstance(cur_value, Iterable):
            return self._predict_matrix_element_array(
                np.asarray(cur_value), transition, operator
            )
        return self._predict_matrix_element(cur_value, transition, operator)

    def _predict_matrix_element_array(
        self,
        cur_values: NDArray[np.float64],
        transition: tuple[int, int],
        operator: MatrixOperator,
    ) -> NDArray[np.float64]:
        """Predict one 0/1 matrix-element transition through the shared engine."""

        engine_transition = _matrix_engine_transition(transition)
        _, elems = self._engine.predict_matrix_elements(
            np.asarray(cur_values, dtype=np.float64),
            (engine_transition,),
            operator,
        )
        return np.asarray(elems[0], dtype=np.float64)


def _frequency_engine_transition(
    transition: tuple[int, int],
) -> tuple[tuple[int, int], float]:
    frm, to = transition
    if frm < 0 or to < 0:
        raise PredictorInputError(f"Transition levels must be >= 0, got {transition}")
    if to >= frm:
        return transition, 1.0
    return (to, frm), -1.0


def _matrix_engine_transition(transition: tuple[int, int]) -> tuple[int, int]:
    frm, to = transition
    if frm < 0 or to < 0:
        raise PredictorInputError(f"Transition levels must be >= 0, got {transition}")
    if frm > 1 or to > 1:
        raise PredictorInputError(
            "FluxoniumPredictor matrix elements support only levels 0 and 1; "
            f"got {transition}"
        )
    if to >= frm:
        return transition
    return (to, frm)

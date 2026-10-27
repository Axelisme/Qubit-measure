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

    EJ/EC/EL are energies in GHz.
    """

    EJ: float | None
    EC: float | None
    EL: float | None


@dataclass(frozen=True)
class FluxoniumParams:
    """Independent fluxonium model inputs; None means unknown, never a default.

    EJ, EC and EL are positive finite Josephson, charging and inductive energies
    in GHz. Instances are immutable; no component or file source changes.
    """

    model: ClassVar[Literal["fluxonium"]] = "fluxonium"

    EJ: float | None = None
    EC: float | None = None
    EL: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> FluxoniumParams:
        """Decode named parameters in the class's GHz units, without defaults.

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
        )

    def to_dict(self) -> FluxoniumParameterValues:
        """Return a detached named GHz projection, retaining unknown fields."""
        return FluxoniumParameterValues(
            EJ=self.EJ,
            EC=self.EC,
            EL=self.EL,
        )

    def missing(self) -> tuple[str, ...]:
        """Return unknown field names in declaration order, without side effects."""
        return tuple(
            field.name for field in fields(self) if getattr(self, field.name) is None
        )

    def build_predictor(self) -> FluxoniumPredictor:
        """Build a fresh MHz-query model, without modifying these parameters.

        PredictorInputError names missing fields, nonfinite inputs or nonpositive energies.
        No value is filled implicitly.
        """
        missing = self.missing()
        if missing:
            raise PredictorInputError("Missing parameters: " + ", ".join(missing))
        ej, ec, el = self.EJ, self.EC, self.EL
        if ej is None or ec is None or el is None:
            raise PredictorInputError("Missing parameters")
        for name, value in (
            ("EJ", ej),
            ("EC", ec),
            ("EL", el),
        ):
            if not isfinite(value):
                raise PredictorInputError(f"{name} must be finite")
            if value <= 0:
                raise PredictorInputError(f"{name} must be positive")
        return FluxoniumPredictor((ej, ec, el))


class FluxoniumPredictor:
    """Fluxonium spectrum queries at reduced flux in Phi0.

    params is (EJ, EC, EL), with energies in GHz.
    Direct callers supply finite positive energies. Params.build_predictor checks
    completeness and physical ranges. Queries return MHz or dimensionless matrix
    magnitudes, never mutate inputs, and propagate numerical solver failures.
    """

    def __init__(self, params: tuple[float, float, float]) -> None:
        """Store validated energy inputs; defer spectrum evaluation to queries."""
        self._engine = FluxoniumPrediction(params)

    @property
    def params(self) -> tuple[float, float, float]:
        """Josephson, charging and inductive energies (EJ, EC, EL), in GHz."""
        return self._engine.params

    @params.setter
    def params(self, new_params: tuple[float, float, float]) -> None:
        self._engine.params = new_params

    def clone(self) -> FluxoniumPredictor:
        """Return an independent model with identical energies."""
        return FluxoniumPredictor(self.params)

    def calculate_offset_correction(
        self, cur_flux: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> CalibrationResult:
        """Return the minimum-|correction| Phi0 spectrum inversion result.

        cur_flux includes the caller's current offset; cur_freq is measured MHz.
        transition requires integer levels 0 <= from < to. Unreachable targets
        return globally nearest MHz with clamped=true. No inputs change.
        Invalid inputs raise PredictorInputError; numerical failures propagate.
        See Predictor.calculate_offset_correction for branch resolution/tolerances.
        """
        frm, to = transition
        if type(frm) is not int or type(to) is not int or frm < 0 or to <= frm:
            raise PredictorInputError("Calibration needs integer levels 0 <= from < to")
        if not (isfinite(cur_flux) and isfinite(cur_freq)):
            raise PredictorInputError("cur_flux and cur_freq must be finite")

        def frequencies(fluxs: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.predict_frequencies_mhz(fluxs, (transition,))[1][0]

        return calibrate_spectrum(frequencies, cur_flux, cur_freq)

    def _predict_freq(self, cur_flux: float, transition: tuple[int, int]) -> float:
        fluxs = np.array([cur_flux], dtype=np.float64)
        freq = self._predict_freq_array(fluxs, transition)
        return float(freq[0])

    @overload
    def predict_freq(
        self, cur_flux: float, transition: tuple[int, int] = (0, 1)
    ) -> float: ...

    @overload
    def predict_freq(
        self, cur_flux: NDArray[np.float64], transition: tuple[int, int] = (0, 1)
    ) -> NDArray[np.float64]: ...

    def predict_freq(
        self,
        cur_flux: float | NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
    ) -> float | NDArray[np.float64]:
        """
        Predict the transition frequency of a fluxonium qubit.
        Args:
            cur_flux (float): Reduced flux in Phi0.
            transition (tuple[from, to]): transition between which level
        Returns:
            float: transition frequency in MHz.
        """
        if isinstance(cur_flux, Iterable):
            return self._predict_freq_array(np.asarray(cur_flux), transition)
        return self._predict_freq(cur_flux, transition)

    def _predict_freq_array(
        self,
        fluxs: NDArray[np.float64],
        transition: tuple[int, int],
    ) -> NDArray[np.float64]:
        """Predict one transition over a reduced-flux array through the shared engine."""

        engine_transition, sign = _frequency_engine_transition(transition)
        _, freqs_mhz = self._engine.predict_frequencies_mhz(
            np.asarray(fluxs, dtype=np.float64),
            (engine_transition,),
            cutoff=40,
        )
        return np.asarray(sign * freqs_mhz[0], dtype=np.float64)

    def predict_frequencies_mhz(
        self,
        fluxs: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, MHz frequencies), shaped (N,) and (T, N).

        fluxs is a finite 1-D reduced-flux grid in Phi0. transitions is a nonempty
        ordered tuple of nonnegative integer (from, to) levels with to >= from.
        Rows/columns preserve request order and duplicates. Invalid transitions
        raise PredictorInputError; numerical errors propagate. Inputs/model are unchanged.
        """
        return self._engine.predict_frequencies_mhz(fluxs, transitions, cutoff=40)

    def predict_matrix_elements(
        self,
        fluxs: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
        operator: MatrixOperator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, magnitudes), shaped (N,) and (T, N), without mutation.

        fluxs is a finite 1-D reduced-flux grid in Phi0; fluxs are flux quanta. transitions is
        nonempty with nonnegative integer levels and to >= from. n is charge,
        phi is phase; dimensionless absolute rows include levels above one.
        Invalid transitions/operators raise PredictorInputError; numerical errors propagate.
        """
        return self._engine.predict_matrix_elements(fluxs, transitions, operator)

    def _predict_matrix_element(
        self,
        cur_flux: float,
        transition: tuple[int, int],
        operator: MatrixOperator,
    ) -> float:
        fluxs = np.array([cur_flux], dtype=np.float64)
        elems = self._predict_matrix_element_array(fluxs, transition, operator)
        return float(elems[0])

    @overload
    def predict_matrix_element(
        self,
        cur_flux: float,
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> float: ...

    @overload
    def predict_matrix_element(
        self,
        cur_flux: NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> NDArray[np.float64]: ...

    def predict_matrix_element(
        self,
        cur_flux: float | NDArray[np.float64],
        transition: tuple[int, int] = (0, 1),
        operator: MatrixOperator = "n",
    ) -> float | NDArray[np.float64]:
        """
        Predict the matrix element of operator between two levels of a fluxonium qubit.
        Args:
            cur_flux (float): Reduced flux in Phi0.
            transition (tuple[from, to]): transition between which level
            operator (str): 'phi' or 'n'
        Returns:
            float: matrix element of operator between two levels.
        """
        if isinstance(cur_flux, Iterable):
            return self._predict_matrix_element_array(
                np.asarray(cur_flux), transition, operator
            )
        return self._predict_matrix_element(cur_flux, transition, operator)

    def _predict_matrix_element_array(
        self,
        fluxs: NDArray[np.float64],
        transition: tuple[int, int],
        operator: MatrixOperator,
    ) -> NDArray[np.float64]:
        """Predict one 0/1 matrix-element transition through the shared engine."""

        engine_transition = _matrix_engine_transition(transition)
        _, elems = self._engine.predict_matrix_elements(
            np.asarray(fluxs, dtype=np.float64),
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

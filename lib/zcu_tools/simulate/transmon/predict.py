"""Neutral SQUID transmon parameters and spectrum queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from typing import ClassVar, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate._calibration import calibrate_spectrum
from zcu_tools.simulate.predictor import (
    CalibrationResult,
    PredictorInputError,
    parameter_number,
)

# Charge states n = -N..N at zero offset charge; converged far beyond the levels
# a front end can request.
_CHARGE_CUTOFF = 30
MAX_LEVEL = 20


class TransmonParameterValues(TypedDict):
    """Named parameter projection; None denotes an unknown value.

    EJmax/EC are energies in GHz; d is dimensionless asymmetry in [0, 1].
    """

    EJmax: float | None
    EC: float | None
    d: float | None


@dataclass(frozen=True)
class TransmonParams:
    """Independent SQUID transmon model inputs; None means unknown, never a default.

    EJmax and EC are positive finite Josephson and charging energies in GHz.
    d is finite absolute junction asymmetry in [0, 1]; zero is symmetric.
    Instances are immutable; no component or file source changes.
    """

    model: ClassVar[Literal["transmon"]] = "transmon"

    EJmax: float | None = None
    EC: float | None = None
    d: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TransmonParams:
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
            EJmax=parameter_number(values, "EJmax"),
            EC=parameter_number(values, "EC"),
            d=parameter_number(values, "d"),
        )

    def to_dict(self) -> TransmonParameterValues:
        """Return a detached named GHz projection, retaining unknown fields."""
        return TransmonParameterValues(
            EJmax=self.EJmax,
            EC=self.EC,
            d=self.d,
        )

    def missing(self) -> tuple[str, ...]:
        """Return unknown field names in declaration order, without side effects."""
        return tuple(
            field.name for field in fields(self) if getattr(self, field.name) is None
        )

    def build_predictor(self) -> TransmonPredictor:
        """Build a fresh MHz-query model, without modifying these parameters.

        PredictorInputError names missing fields, nonfinite inputs, nonpositive
        energies, or d outside [0, 1]. No value is filled implicitly.
        """
        missing = self.missing()
        if missing:
            raise PredictorInputError("Missing parameters: " + ", ".join(missing))
        ej, ec, asymmetry = self.EJmax, self.EC, self.d
        if ej is None or ec is None or asymmetry is None:
            raise PredictorInputError("Missing parameters")
        for name, value in (
            ("EJmax", ej),
            ("EC", ec),
            ("d", asymmetry),
        ):
            if not isfinite(value):
                raise PredictorInputError(f"{name} must be finite")
            if name != "d" and value <= 0:
                raise PredictorInputError(f"{name} must be positive")
        if not 0 <= asymmetry <= 1:
            raise PredictorInputError("d must be between 0 and 1")
        return TransmonPredictor((ej, ec, asymmetry))


class TransmonPredictor:
    """Transmon spectrum queries at reduced flux in Phi0.

    params is (EJmax, EC, d), with energies in GHz and asymmetry d in [0, 1].
    Direct callers supply finite positive energies. Params.build_predictor checks
    completeness and physical ranges. Frequency queries return MHz, never mutate
    inputs, and propagate numerical solver failures. Matrix curves are not provided.
    """

    def __init__(self, params: tuple[float, float, float]) -> None:
        """Store validated energy inputs; defer spectrum evaluation to queries."""
        self.params = params

    def predict_freq(
        self, cur_flux: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """Return E_to - E_from in MHz at a finite reduced flux in Phi0."""
        _, freqs = self.predict_frequencies_mhz(
            np.array([cur_flux], dtype=np.float64), (transition,)
        )
        return float(freqs[0, 0])

    def predict_frequencies_mhz(
        self,
        fluxs: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, MHz frequencies), shaped (N,) and (T, N).

        fluxs is a finite 1-D reduced-flux grid in Phi0; transitions is a nonempty
        ordered tuple of nonnegative integer (from, to) levels with from <= to and
        to <= MAX_LEVEL. Invalid inputs raise PredictorInputError; nothing is mutated.
        """
        _validate_transitions(transitions)
        fluxs = np.asarray(fluxs, dtype=np.float64)
        if fluxs.ndim != 1 or not np.all(np.isfinite(fluxs)):
            raise PredictorInputError("fluxs must be a finite 1-D array")
        levels = self._levels_ghz(fluxs, max(to for _, to in transitions) + 1)
        rows = [(levels[:, to] - levels[:, frm]) * 1e3 for frm, to in transitions]
        return fluxs, np.stack(rows, axis=0)

    def calculate_offset_correction(
        self, cur_flux: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> CalibrationResult:
        """Invert the exact spectrum; see Predictor.calculate_offset_correction for result policy.

        cur_flux is finite Phi0 including the caller offset; cur_freq is measured MHz.
        transition requires integers 0 <= from < to <= MAX_LEVEL.
        All branches participate in smallest-|correction| selection. An unreachable
        target returns nearest actual MHz and clamped=true. Invalid inputs raise
        PredictorInputError; numerical failures propagate and no parameters change.
        """
        _validate_transitions((transition,))
        if transition[0] >= transition[1]:
            raise PredictorInputError("Calibration needs from-level < to-level")
        if not (isfinite(cur_flux) and isfinite(cur_freq)):
            raise PredictorInputError("cur_flux and cur_freq must be finite")

        def frequencies(fluxs: NDArray[np.float64]) -> NDArray[np.float64]:
            levels = self._levels_ghz(fluxs, transition[1] + 1)
            return (levels[:, transition[1]] - levels[:, transition[0]]) * 1e3

        return calibrate_spectrum(frequencies, cur_flux, cur_freq)

    def _levels_ghz(
        self, fluxs: NDArray[np.float64], count: int
    ) -> NDArray[np.float64]:
        """Return the lowest ``count`` energies in GHz, shaped (N, count)."""
        import scqubits as scq

        ej_max, ec, asymmetry = self.params
        qubit = scq.TunableTransmon(
            EJmax=ej_max, EC=ec, d=asymmetry, ng=0.0, flux=0.0, ncut=_CHARGE_CUTOFF
        )
        out = np.empty((fluxs.size, count), dtype=np.float64)
        for index, flux in enumerate(fluxs):
            qubit.flux = float(flux)
            out[index] = qubit.eigenvals(evals_count=count)
        return out


def _validate_transitions(transitions: tuple[tuple[int, int], ...]) -> None:
    if not transitions:
        raise PredictorInputError("transitions must not be empty")
    for frm, to in transitions:
        if type(frm) is not int or type(to) is not int:
            raise PredictorInputError(
                f"Transition levels must be integers, got {(frm, to)}"
            )
        if frm < 0 or to < frm or to > MAX_LEVEL:
            raise PredictorInputError(
                f"Transition must satisfy 0 <= from <= to <= {MAX_LEVEL}, got {(frm, to)}"
            )

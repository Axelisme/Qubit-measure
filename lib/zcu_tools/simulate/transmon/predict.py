"""Neutral SQUID transmon parameters and spectrum queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from typing import ClassVar, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import root_scalar

from zcu_tools.simulate.fluxonium.prediction import FluxAffineMap
from zcu_tools.simulate.predictor import parameter_number

# Charge states n = -N..N at zero offset charge; converged far beyond the levels
# a front end can request.
_CHARGE_CUTOFF = 30
MAX_LEVEL = 20
_CALIBRATION_SCAN_POINTS = 401


class TransmonParameterValues(TypedDict):
    """Named parameter projection; None denotes an unknown value.

    EJmax/EC are energies in GHz; d is dimensionless asymmetry in [0, 1]. flux_half is the zero-bias half-flux value,
    flux_period is the signed device-value change per quantum, and flux_bias
    is the additive device-value correction. All affine values use native units.
    """

    EJmax: float | None
    EC: float | None
    d: float | None
    flux_half: float | None
    flux_period: float | None
    flux_bias: float | None


@dataclass(frozen=True)
class TransmonParams:
    """Independent SQUID transmon model inputs; None means unknown, never a default.

    EJmax and EC are positive finite Josephson and charging energies in GHz.
    d is the finite absolute junction asymmetry in [0, 1]; zero is symmetric. flux_half is the zero-bias half-flux device value; flux_period is
    the finite nonzero device-value change per flux quantum (either sign).
    flux_bias is the finite additive correction to the device value. All three
    affine inputs use the source's native device units. Instances are immutable;
    creating or completing one never changes its component or file source.
    """

    model: ClassVar[Literal["transmon"]] = "transmon"

    EJmax: float | None = None
    EC: float | None = None
    d: float | None = None
    flux_half: float | None = None
    flux_period: float | None = None
    flux_bias: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> TransmonParams:
        """Decode named parameters in the class's GHz/native units, without defaults.

        Omitted/null fields remain unknown. ValueError names unknown keys,
        nonnumeric/bool or nonfinite values. Physical constraints are checked
        by build_predictor, so partial inputs can still be inspected/edited.
        The source mapping is never changed.
        """
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError("Unknown parameters: " + ", ".join(sorted(unknown)))
        return cls(
            EJmax=parameter_number(values, "EJmax"),
            EC=parameter_number(values, "EC"),
            d=parameter_number(values, "d"),
            flux_half=parameter_number(values, "flux_half"),
            flux_period=parameter_number(values, "flux_period"),
            flux_bias=parameter_number(values, "flux_bias"),
        )

    def to_dict(self) -> TransmonParameterValues:
        """Return a detached named GHz/native projection, retaining unknown fields."""
        return TransmonParameterValues(
            EJmax=self.EJmax,
            EC=self.EC,
            d=self.d,
            flux_half=self.flux_half,
            flux_period=self.flux_period,
            flux_bias=self.flux_bias,
        )

    def missing(self) -> tuple[str, ...]:
        """Return unknown field names in declaration order, without side effects."""
        return tuple(
            field.name for field in fields(self) if getattr(self, field.name) is None
        )

    def build_predictor(self) -> TransmonPredictor:
        """Build a fresh MHz-query model, without modifying these parameters.

        ValueError names missing fields, nonfinite inputs, nonpositive energies
        or a zero period. No physical or affine value is filled implicitly.
        """
        missing = self.missing()
        if missing:
            raise ValueError("Missing parameters: " + ", ".join(missing))
        ej, ec, asymmetry = self.EJmax, self.EC, self.d
        half, period, bias = self.flux_half, self.flux_period, self.flux_bias
        # Explicit narrowing also keeps the runtime contract valid under python -O.
        if (
            ej is None
            or ec is None
            or asymmetry is None
            or half is None
            or period is None
            or bias is None
        ):
            raise ValueError("Missing parameters")
        for name, value in (
            ("EJmax", ej),
            ("EC", ec),
            ("d", asymmetry),
            ("flux_half", half),
            ("flux_period", period),
            ("flux_bias", bias),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in ("EJmax", "EC") and value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= asymmetry <= 1:
            raise ValueError("d must be between 0 and 1")
        if period == 0:
            raise ValueError("flux_period must be nonzero")
        return TransmonPredictor((ej, ec, asymmetry), half, period, bias)


class TransmonPredictor:
    """SQUID transmon queries with GHz energies and native device-value affine inputs.

    The Hamiltonian is H = 4 EC n^2 - EJ(flux) cos(phi) at zero offset charge,
    diagonalized exactly in the charge basis. flux is in flux quanta and
    EJ(flux) = EJmax * sqrt(cos^2(pi flux) + d^2 sin^2(pi flux)), so flux 0.5
    (``flux_half`` in device units) is the minimum-EJ point; d = 0 is symmetric
    and EJ vanishes there. Frequencies are MHz. Queries never mutate the model.
    Matrix elements are not provided by this model.
    """

    def __init__(
        self,
        params: tuple[float, float, float],
        flux_half: float,
        flux_period: float,
        flux_bias: float,
    ) -> None:
        self.params = params
        self._affine = FluxAffineMap(flux_half, flux_period, flux_bias)

    @property
    def flux_half(self) -> float:
        """Zero-bias native device value at half a flux quantum."""
        return self._affine.flux_half

    @property
    def flux_period(self) -> float:
        """Signed nonzero native device-value change for one flux quantum."""
        return self._affine.flux_period

    @property
    def flux_bias(self) -> float:
        """Additive native device-value correction; queries never change it."""
        return self._affine.flux_bias

    def value_to_flux(self, cur_value: float) -> float:
        """Map a finite native value to flux quanta, without folding."""
        return self._affine.value_to_flux(cur_value)

    def flux_to_value(self, cur_flux: float) -> float:
        """Invert the affine for finite flux quanta; return native device units."""
        return self._affine.flux_to_value(cur_flux)

    def predict_freq(
        self, cur_value: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """Return E_to - E_from in MHz at a finite native device value."""
        _, freqs = self.predict_frequencies_mhz(
            np.array([cur_value], dtype=np.float64), (transition,)
        )
        return float(freqs[0, 0])

    def predict_frequencies_mhz(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (fluxs, MHz frequencies), shaped (N,) and (T, N).

        values is a finite 1-D native device-value grid; transitions is a nonempty
        ordered tuple of nonnegative integer (from, to) levels with from <= to and
        to <= MAX_LEVEL. Invalid inputs raise ValueError; nothing is mutated.
        """
        _validate_transitions(transitions)
        value_arr = np.asarray(values, dtype=np.float64)
        if value_arr.ndim != 1 or not np.all(np.isfinite(value_arr)):
            raise ValueError("values must be a finite 1-D array")
        fluxs = self._affine.values_to_flux(value_arr)
        levels = self._levels_ghz(fluxs, max(to for _, to in transitions) + 1)
        rows = [(levels[:, to] - levels[:, frm]) * 1e3 for frm, to in transitions]
        return fluxs, np.stack(rows, axis=0)

    def calculate_bias(
        self, cur_value: float, cur_freq: float, transition: tuple[int, int] = (0, 1)
    ) -> float:
        """Return the candidate additive native bias nearest zero bias; never install it.

        Roots of the transition spectrum are found on one flux half period (the
        spectrum is even and one-periodic in flux), then mirrored and shifted by
        whole periods; the candidate with the smallest absolute bias wins.
        ValueError reports a frequency the transition never reaches.
        """
        _validate_transitions((transition,))
        if transition[0] >= transition[1]:
            raise ValueError("Calibration needs from-level < to-level")
        if not (isfinite(cur_value) and isfinite(cur_freq)):
            raise ValueError("cur_value and cur_freq must be finite")

        scan = np.linspace(0.0, 0.5, _CALIBRATION_SCAN_POINTS)

        def gap(flux: NDArray[np.float64]) -> NDArray[np.float64]:
            levels = self._levels_ghz(flux, transition[1] + 1)
            return (
                levels[:, transition[1]] - levels[:, transition[0]]
            ) * 1e3 - cur_freq

        samples = gap(scan)
        roots: list[float] = []
        for index in np.flatnonzero(samples == 0.0):
            roots.append(float(scan[index]))
        for index in np.flatnonzero(samples[:-1] * samples[1:] < 0.0):
            bracket = (float(scan[index]), float(scan[index + 1]))
            solution = root_scalar(
                lambda flux: float(gap(np.array([flux]))[0]),
                bracket=bracket,
                method="brentq",
                xtol=1e-12,
            )
            roots.append(float(solution.root))
        if not roots:
            raise ValueError(
                f"Transition {transition} never reaches {cur_freq} MHz with these parameters"
            )

        zero_bias_flux = FluxAffineMap(
            self.flux_half, self.flux_period, 0.0
        ).value_to_flux(cur_value)
        candidates = [
            sign * root + round(zero_bias_flux - sign * root)
            for root in roots
            for sign in (1.0, -1.0)
        ]
        flux = min(candidates, key=lambda candidate: abs(candidate - zero_bias_flux))
        return (flux - zero_bias_flux) * self.flux_period

    def _levels_ghz(
        self, fluxs: NDArray[np.float64], count: int
    ) -> NDArray[np.float64]:
        """Return the lowest ``count`` energies in GHz, shaped (N, count)."""
        ej_max, ec, asymmetry = self.params
        phase = np.pi * fluxs
        ej = ej_max * np.sqrt(np.cos(phase) ** 2 + (asymmetry * np.sin(phase)) ** 2)
        charges = np.arange(-_CHARGE_CUTOFF, _CHARGE_CUTOFF + 1, dtype=np.float64)
        diagonal = 4.0 * ec * charges**2
        out = np.empty((fluxs.size, count), dtype=np.float64)
        for index, energy in enumerate(ej):
            off = np.full(charges.size - 1, -0.5 * energy)
            out[index] = eigh_tridiagonal(
                diagonal,
                off,
                eigvals_only=True,
                select="i",
                select_range=(0, count - 1),
            )
        return out


def _validate_transitions(transitions: tuple[tuple[int, int], ...]) -> None:
    if not transitions:
        raise ValueError("transitions must not be empty")
    for frm, to in transitions:
        if type(frm) is not int or type(to) is not int:
            raise ValueError(f"Transition levels must be integers, got {(frm, to)}")
        if frm < 0 or to < frm or to > MAX_LEVEL:
            raise ValueError(
                f"Transition must satisfy 0 <= from <= to <= {MAX_LEVEL}, got {(frm, to)}"
            )

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium.dispersive import (
    DressedLabelingError,
    calculate_dispersive_vs_flux,
    calculate_dispersive_vs_flux_fast,
)

PredictionBackend = Literal["fast", "scqubits"]
MatrixOperator = Literal["n", "phi"]


@dataclass(frozen=True, slots=True)
class PredictionResolution:
    """Hilbert-space resolution for Fluxonium prediction."""

    qub_dim: int = 15
    qub_cutoff: int = 30
    res_dim: int = 4

    def __post_init__(self) -> None:
        for name, value in (
            ("qub_dim", self.qub_dim),
            ("qub_cutoff", self.qub_cutoff),
            ("res_dim", self.res_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")


DEFAULT_PREDICTION_RESOLUTION = PredictionResolution()


@dataclass(frozen=True, slots=True)
class FluxAffineMap:
    """Affine conversion between device values and normalized flux."""

    flux_half: float
    flux_period: float
    flux_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.flux_period == 0.0:
            raise ValueError("flux_period must be non-zero (value<->flux affine)")

    def value_to_flux(self, value: float) -> float:
        return (value + self.flux_bias - self.flux_half) / self.flux_period + 0.5

    def values_to_flux(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        arr = np.asarray(values, dtype=np.float64)
        fluxs = (arr + self.flux_bias - self.flux_half) / self.flux_period + 0.5
        return np.asarray(fluxs, dtype=np.float64)

    def flux_to_value(self, flux: float) -> float:
        return (flux - 0.5) * self.flux_period + self.flux_half - self.flux_bias


@dataclass(frozen=True, slots=True)
class DispersivePredictionResult:
    """Dispersive prediction lines plus lightweight backend provenance."""

    lines: tuple[NDArray[np.float64], ...]
    backend: PredictionBackend

    @property
    def used_fallback(self) -> bool:
        return self.backend == "scqubits"


class FluxoniumPrediction:
    """Production prediction engine for Fluxonium simulation policy.

    The engine owns the value-to-flux affine, typed resolution, dispersive
    fast/scqubits fallback, fallback provenance, and construction of axis-bound
    cache sessions. GUI/session callers should adapt this class instead of
    reimplementing prediction policy.
    """

    def __init__(
        self,
        params: tuple[float, float, float],
        *,
        flux_half: float = 0.0,
        flux_period: float = 1.0,
        flux_bias: float = 0.0,
        resolution: PredictionResolution = DEFAULT_PREDICTION_RESOLUTION,
    ) -> None:
        if len(params) != 3:
            raise ValueError(f"params must be (EJ, EC, EL), got {params!r}")
        self.params: tuple[float, float, float] = (
            float(params[0]),
            float(params[1]),
            float(params[2]),
        )
        self.affine = FluxAffineMap(flux_half, flux_period, flux_bias)
        self.resolution = resolution

    def value_to_flux(self, value: float) -> float:
        return self.affine.value_to_flux(value)

    def values_to_flux(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.affine.values_to_flux(values)

    def flux_to_value(self, flux: float) -> float:
        return self.affine.flux_to_value(flux)

    def bind_flux_axis(self, fluxs: NDArray[np.float64]) -> FluxoniumPredictionSession:
        return FluxoniumPredictionSession(self, fluxs)

    def predict_dispersive(
        self,
        fluxs: NDArray[np.float64],
        g: float,
        bare_rf: float,
        *,
        return_dim: int = 2,
    ) -> DispersivePredictionResult:
        if return_dim <= 0:
            raise ValueError(f"return_dim must be positive, got {return_dim}")

        flux_arr = np.asarray(fluxs, dtype=np.float64)
        res = self.resolution
        try:
            lines = calculate_dispersive_vs_flux_fast(
                self.params,
                flux_arr,
                bare_rf,
                g,
                res_dim=res.res_dim,
                qub_cutoff=res.qub_cutoff,
                qub_dim=res.qub_dim,
                return_dim=return_dim,
            )
        except DressedLabelingError:
            lines = calculate_dispersive_vs_flux(
                self.params,
                flux_arr,
                bare_rf,
                g,
                progress=False,
                res_dim=res.res_dim,
                qub_cutoff=res.qub_cutoff,
                qub_dim=res.qub_dim,
                return_dim=return_dim,
            )
            return DispersivePredictionResult(
                lines=_as_float_lines(lines), backend="scqubits"
            )
        return DispersivePredictionResult(lines=_as_float_lines(lines), backend="fast")

    def predict_frequencies_mhz(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
        *,
        cutoff: int = 40,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Frequency curves for one value grid.

        Returns ``(fluxs, freqs_mhz)`` where ``freqs_mhz`` has shape
        ``(n_transitions, n_values)``.
        """
        _validate_transitions(transitions)
        if cutoff <= 0:
            raise ValueError(f"cutoff must be positive, got {cutoff}")

        from zcu_tools.simulate.fluxonium import energies as energies_mod

        value_arr = np.asarray(values, dtype=np.float64)
        fluxs = self.values_to_flux(value_arr)
        evals_count = max(to for _, to in transitions) + 5
        _, energies = energies_mod.calculate_energy_vs_flux(
            self.params, fluxs, cutoff=cutoff, evals_count=evals_count
        )

        rows: list[NDArray[np.float64]] = []
        for frm, to in transitions:
            diff = (energies[:, to] - energies[:, frm]) * 1e3
            rows.append(np.asarray(diff, dtype=np.float64))
        return fluxs, np.stack(rows, axis=0)

    def predict_matrix_elements(
        self,
        values: NDArray[np.float64],
        transitions: tuple[tuple[int, int], ...],
        operator: MatrixOperator,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Matrix-element curves for one value grid.

        Returns ``(fluxs, magnitudes)`` where ``magnitudes`` has shape
        ``(n_transitions, n_values)``.
        """
        _validate_transitions(transitions)

        from zcu_tools.simulate.fluxonium import matrix_element as mat_mod

        value_arr = np.asarray(values, dtype=np.float64)
        fluxs = self.values_to_flux(value_arr)
        return_dim = max(max(frm, to) for frm, to in transitions) + 1
        if operator == "n":
            _, opers = mat_mod.calculate_n_oper_vs_flux(
                self.params, fluxs, return_dim=return_dim
            )
        elif operator == "phi":
            _, opers = mat_mod.calculate_phi_oper_vs_flux(
                self.params, fluxs, return_dim=return_dim
            )
        else:
            raise ValueError(f"unsupported matrix operator: {operator!r}")

        rows: list[NDArray[np.float64]] = []
        for frm, to in transitions:
            rows.append(np.abs(opers[:, frm, to]).astype(np.float64))
        return fluxs, np.stack(rows, axis=0)


class FluxoniumPredictionSession:
    """Axis-bound dispersive cache for one engine and one flux axis."""

    def __init__(self, engine: FluxoniumPrediction, fluxs: NDArray[np.float64]) -> None:
        self._engine = engine
        self._fluxs = np.array(fluxs, dtype=np.float64, copy=True)
        self._fluxs.setflags(write=False)

        @lru_cache(maxsize=None)
        def _cached(
            g: float,
            bare_rf: float,
            return_dim: int,
        ) -> DispersivePredictionResult:
            return self._engine.predict_dispersive(
                self._fluxs, g, bare_rf, return_dim=return_dim
            )

        self._cached = _cached

    def predict_dispersive(
        self,
        g: float,
        bare_rf: float,
        *,
        return_dim: int = 2,
    ) -> DispersivePredictionResult:
        return self._cached(g, bare_rf, return_dim)

    def flux_axis(self) -> NDArray[np.float64]:
        return self._fluxs


def _as_float_lines(
    lines: tuple[NDArray[np.float64], ...],
) -> tuple[NDArray[np.float64], ...]:
    return tuple(np.asarray(line, dtype=np.float64) for line in lines)


def _validate_transitions(transitions: tuple[tuple[int, int], ...]) -> None:
    if len(transitions) == 0:
        raise ValueError("transitions must not be empty")
    for frm, to in transitions:
        if frm < 0 or to < 0:
            raise ValueError(f"Transition levels must be >= 0, got ({frm}, {to})")
        if to < frm:
            raise ValueError(
                f"Transition to-level must be >= from-level, got ({frm}, {to})"
            )

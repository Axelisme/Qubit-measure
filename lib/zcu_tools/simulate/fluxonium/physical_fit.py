"""Local physical-model fitting helpers for Fluxonium predictors."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor


@dataclass(frozen=True)
class FluxoniumModelSnapshot:
    """Immutable parameters needed to reconstruct a FluxoniumPredictor."""

    params: tuple[float, float, float]
    flux_half: float
    flux_period: float
    flux_bias: float

    def __post_init__(self) -> None:
        params = tuple(float(value) for value in self.params)
        if len(params) != 3:
            raise ValueError("Fluxonium model params must be a 3-tuple")
        if any(not math.isfinite(value) or value <= 0.0 for value in params):
            raise ValueError("Fluxonium model params must be positive and finite")
        flux_half = _finite("flux_half", self.flux_half)
        flux_period = _finite("flux_period", self.flux_period)
        if flux_period == 0.0:
            raise ValueError("Fluxonium model flux_period must be non-zero")
        flux_bias = _finite("flux_bias", self.flux_bias)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "flux_half", flux_half)
        object.__setattr__(self, "flux_period", flux_period)
        object.__setattr__(self, "flux_bias", flux_bias)

    def make_predictor(self) -> FluxoniumPredictor:
        return FluxoniumPredictor(
            self.params,
            self.flux_half,
            self.flux_period,
            self.flux_bias,
        )


@dataclass(frozen=True)
class FluxoniumLocalFitResult:
    """Result of a bounded local physical-model fit."""

    accepted: bool
    reason: str
    base: FluxoniumModelSnapshot
    fitted: FluxoniumModelSnapshot | None
    predictor: FluxoniumPredictor | None
    n_points: int
    base_rms_mhz: float
    fitted_rms_mhz: float


def fit_local_fluxonium_model(
    base: FluxoniumModelSnapshot,
    measured_points: Iterable[tuple[float, float]],
    *,
    weights: Iterable[float] | None = None,
    transition: tuple[int, int] = (0, 1),
    param_relative_span: float = 0.25,
    flux_bias_period_fraction: float = 0.10,
) -> FluxoniumLocalFitResult:
    """Fit ``(EJ, EC, EL, flux_bias)`` near ``base`` without mutating it.

    Optimization failures are returned as rejected results. Invalid caller inputs
    fail fast because a rejected fit would hide a broken run configuration.
    """

    fluxes, freqs, weight_array = _validated_points(measured_points, weights)
    if fluxes.size < 4:
        return _rejected(base, "not enough points", fluxes.size, math.nan, math.nan)

    base_pred = _predict(base, fluxes, transition)
    base_rms = _weighted_rms(base_pred - freqs, weight_array)
    if not math.isfinite(base_rms):
        return _rejected(
            base, "base residual is not finite", fluxes.size, base_rms, math.nan
        )

    lower, upper = _bounds(
        base,
        param_relative_span=param_relative_span,
        flux_bias_period_fraction=flux_bias_period_fraction,
    )
    initial = np.asarray((*base.params, base.flux_bias), dtype=np.float64)

    def residual(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        snapshot = FluxoniumModelSnapshot(
            params=(float(vector[0]), float(vector[1]), float(vector[2])),
            flux_half=base.flux_half,
            flux_period=base.flux_period,
            flux_bias=float(vector[3]),
        )
        return (
            np.sqrt(weight_array) * (_predict(snapshot, fluxes, transition) - freqs)
        ).astype(np.float64)

    try:
        result = least_squares(
            residual,
            initial,
            bounds=(lower, upper),
            method="trf",
            max_nfev=200,
            x_scale="jac",
        )
    except Exception as exc:
        return _rejected(
            base, f"least_squares failed: {exc}", fluxes.size, base_rms, math.nan
        )

    if not result.success:
        return _rejected(base, str(result.message), fluxes.size, base_rms, math.nan)
    if not np.all(np.isfinite(result.x)):
        return _rejected(
            base, "fit result is not finite", fluxes.size, base_rms, math.nan
        )

    fitted = FluxoniumModelSnapshot(
        params=(float(result.x[0]), float(result.x[1]), float(result.x[2])),
        flux_half=base.flux_half,
        flux_period=base.flux_period,
        flux_bias=float(result.x[3]),
    )
    fitted_pred = _predict(fitted, fluxes, transition)
    fitted_rms = _weighted_rms(fitted_pred - freqs, weight_array)
    if not math.isfinite(fitted_rms):
        return _rejected(
            base, "fit residual is not finite", fluxes.size, base_rms, fitted_rms
        )

    predictor = fitted.make_predictor()
    return FluxoniumLocalFitResult(
        accepted=True,
        reason="accepted",
        base=base,
        fitted=fitted,
        predictor=predictor,
        n_points=int(fluxes.size),
        base_rms_mhz=float(base_rms),
        fitted_rms_mhz=float(fitted_rms),
    )


def _finite(name: str, value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Fluxonium model {name} must be finite")
    return out


def _validated_points(
    measured_points: Iterable[tuple[float, float]],
    weights: Iterable[float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    points = tuple(measured_points)
    fluxes = np.asarray([_finite("measured flux", point[0]) for point in points])
    freqs = np.asarray([_finite("measured frequency", point[1]) for point in points])
    if weights is None:
        weight_array = np.ones(fluxes.shape, dtype=np.float64)
    else:
        raw_weights = tuple(weights)
        if len(raw_weights) != len(points):
            raise ValueError("Fluxonium fit weights must match measured point count")
        weight_array = np.asarray(
            [_positive_finite("fit weight", value) for value in raw_weights],
            dtype=np.float64,
        )
    return fluxes.astype(np.float64), freqs.astype(np.float64), weight_array


def _positive_finite(name: str, value: Any) -> float:
    out = _finite(name, value)
    if out <= 0.0:
        raise ValueError(f"Fluxonium model {name} must be positive")
    return out


def _predict(
    snapshot: FluxoniumModelSnapshot,
    fluxes: Sequence[float] | NDArray[np.float64],
    transition: tuple[int, int],
) -> NDArray[np.float64]:
    predictor = snapshot.make_predictor()
    return np.asarray(
        predictor.predict_freq(np.asarray(fluxes, dtype=np.float64), transition)
    )


def _weighted_rms(
    residuals: NDArray[np.float64], weights: NDArray[np.float64]
) -> float:
    if residuals.size == 0:
        return math.nan
    return float(np.sqrt(np.average(np.square(residuals), weights=weights)))


def _bounds(
    base: FluxoniumModelSnapshot,
    *,
    param_relative_span: float,
    flux_bias_period_fraction: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    span = _positive_finite("param_relative_span", param_relative_span)
    bias_fraction = _positive_finite(
        "flux_bias_period_fraction", flux_bias_period_fraction
    )
    if span >= 1.0:
        raise ValueError("Fluxonium fit param_relative_span must be < 1")
    params = np.asarray(base.params, dtype=np.float64)
    bias_window = abs(base.flux_period) * bias_fraction
    lower = np.asarray(
        (
            *(params * (1.0 - span)),
            base.flux_bias - bias_window,
        ),
        dtype=np.float64,
    )
    upper = np.asarray(
        (
            *(params * (1.0 + span)),
            base.flux_bias + bias_window,
        ),
        dtype=np.float64,
    )
    return lower, upper


def _rejected(
    base: FluxoniumModelSnapshot,
    reason: str,
    n_points: int,
    base_rms_mhz: float,
    fitted_rms_mhz: float,
) -> FluxoniumLocalFitResult:
    return FluxoniumLocalFitResult(
        accepted=False,
        reason=reason,
        base=base,
        fitted=None,
        predictor=None,
        n_points=int(n_points),
        base_rms_mhz=float(base_rms_mhz),
        fitted_rms_mhz=float(fitted_rms_mhz),
    )

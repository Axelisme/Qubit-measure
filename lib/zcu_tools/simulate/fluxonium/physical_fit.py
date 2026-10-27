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
    """Immutable energies in GHz and additive reduced-flux correction in Phi0.

    params contains positive finite EJ/EC/EL. offset_correction is finite and
    added to input reduced phi before querying the reconstructed numeric model.
    Invalid energies or correction raise ValueError.
    """

    params: tuple[float, float, float]
    offset_correction: float = 0.0

    def __post_init__(self) -> None:
        params = tuple(float(value) for value in self.params)
        if len(params) != 3:
            raise ValueError("Fluxonium model params must be a 3-tuple")
        if any(not math.isfinite(value) or value <= 0.0 for value in params):
            raise ValueError("Fluxonium model params must be positive and finite")
        correction = _finite("offset_correction", self.offset_correction)
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "offset_correction", correction)

    def make_predictor(self) -> FluxoniumPredictor:
        """Restore an independent numeric model; queries take phi + correction."""
        return FluxoniumPredictor(self.params)


@dataclass(frozen=True)
class FluxoniumLocalFitResult:
    """Bounded local fit in reduced flux, with residual RMS in MHz.

    accepted/reason describe solver success or rejection. base is unchanged;
    fitted and predictor are None on rejection. On success predictor contains
    fitted energies only; add fitted.offset_correction (Phi0) to query phi.
    n_points counts measured pairs. RMS may be NaN when unavailable.
    """

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
    offset_correction_span: float = 0.10,
) -> FluxoniumLocalFitResult:
    """Fit energies (GHz) and additive phi correction (Phi0) without mutation.

    measured_points contains (reduced phi, frequency MHz), before base's
    correction. weights are positive finite per-point weights. Energies vary
    by param_relative_span; correction varies by offset_correction_span Phi0
    around base. Successful snapshots restore the same corrected curve.
    Fewer than four points or solver failures return rejected results. Invalid
    points, weights or bounds raise ValueError; no native device map is used.
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
        offset_correction_span=offset_correction_span,
    )
    initial = np.asarray((*base.params, base.offset_correction), dtype=np.float64)

    def residual(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        snapshot = FluxoniumModelSnapshot(
            params=(float(vector[0]), float(vector[1]), float(vector[2])),
            offset_correction=float(vector[3]),
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
    # tolerated: Optimizer/model callback failures return a rejected fit with the original base.
    except Exception as exc:  # noqa: BLE001
        return _rejected(
            base, f"least_squares failed: {exc}", fluxes.size, base_rms, math.nan
        )

    if not result.success or not np.all(np.isfinite(result.x)):
        reason = (
            str(result.message) if not result.success else "fit result is not finite"
        )
        return _rejected(base, reason, fluxes.size, base_rms, math.nan)

    fitted = FluxoniumModelSnapshot(
        params=(float(result.x[0]), float(result.x[1]), float(result.x[2])),
        offset_correction=float(result.x[3]),
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
        predictor.predict_freq(
            np.asarray(fluxes, dtype=np.float64) + snapshot.offset_correction,
            transition,
        )
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
    offset_correction_span: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    span = _positive_finite("param_relative_span", param_relative_span)
    correction_window = _positive_finite(
        "offset_correction_span", offset_correction_span
    )
    if span >= 1.0:
        raise ValueError("Fluxonium fit param_relative_span must be < 1")
    params = np.asarray(base.params, dtype=np.float64)
    lower = np.asarray(
        (
            *(params * (1.0 - span)),
            base.offset_correction - correction_window,
        ),
        dtype=np.float64,
    )
    upper = np.asarray(
        (
            *(params * (1.0 + span)),
            base.offset_correction + correction_window,
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

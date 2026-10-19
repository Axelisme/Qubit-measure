"""Purcell equilibrium limits, cache ownership and relaxation-rate arithmetic."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

from zcu_tools.progress_bar import make_pbar
from zcu_tools.resources.qubit_params import (
    QubitParams,
)
from zcu_tools.simulate.fluxonium import (
    calculate_purcell_t1_vs_flux,
)

from .base import (
    DEFAULT_TEMP_LOWER_BOUND,
    DEFAULT_TEMP_UPPER_BOUND,
)
from .models import (
    PurcellCorrectionResult,
    PurcellEffectParams,
    T1CurveContext,
    T1PreparedData,
)

_PURCELL_CACHE_SIZE = 64


_CACHE_FLOAT_DECIMALS = 12


def load_t1_purcell_params(
    context: T1CurveContext,
    *,
    kappa_ghz: float,
) -> PurcellEffectParams:
    """Read bare_rf and g in GHz from context.result_dir/params.json.

    kappa_ghz is the positive finite linewidth kappa_rate/(2*pi) in GHz.
    Return validated resonator inputs. Missing dispersive data raises
    RuntimeError; invalid values raise ValueError and file errors propagate.
    """
    params_file = QubitParams(
        os.path.join(context.result_dir, "params.json"), readonly=True
    )
    dispersive_fit = params_file.get_dispersive_fit()
    if dispersive_fit is None:
        raise RuntimeError("params.json needs a dispersive section with bare_rf and g")
    return PurcellEffectParams(
        kappa_ghz=kappa_ghz,
        bare_rf=dispersive_fit.bare_rf,
        g=dispersive_fit.g,
    )


def calculate_purcell_t1_limit(
    context: T1CurveContext,
    fluxs: NDArray[np.float64],
    purcell: PurcellEffectParams,
    *,
    Temp: float,
) -> NDArray[np.float64]:
    """Return cached 0↔1 Purcell equilibrium T1 in ns on normalized fluxs.

    context supplies fluxonium energies in GHz; purcell supplies GHz resonator
    inputs. Temp must be positive and finite in K, otherwise raise ValueError.
    The result has the same shape and order as fluxs. Its reciprocal is the
    Gamma_eg+Gamma_ge contribution used by fitting and channel curves.
    """
    if not np.isfinite(Temp) or Temp <= 0.0:
        raise ValueError("Temp must be positive and finite")
    return np.asarray(
        _calculate_purcell_t1_limit_cached(
            _float_tuple_cache_key(context.params),
            _array_cache_key(np.asarray(fluxs, dtype=np.float64), name="fluxs"),
            _float_cache_key(purcell.bare_rf),
            _float_cache_key(purcell.kappa_ghz),
            _float_cache_key(purcell.g),
            _float_cache_key(Temp),
        ),
        dtype=np.float64,
    )


def clear_t1_purcell_cache() -> None:
    """Release all cached Purcell sweeps so the next calculation recomputes.

    This clears the single bounded cache used by every T1 Notebook stage.
    It takes no inputs and returns None.
    """
    _calculate_purcell_t1_limit_cached.cache_clear()


def estimate_purcell_Temp_upper_bound(
    data: T1PreparedData,
    purcell: PurcellEffectParams,
    *,
    Temp_range: tuple[float, float] = (
        DEFAULT_TEMP_LOWER_BOUND,
        DEFAULT_TEMP_UPPER_BOUND,
    ),
    tolerance: float = 1e-3,
    max_iter: int = 12,
    progress: bool = False,
) -> float:
    """Estimate the largest Temp in K compatible with observed ns T1 data.

    data supplies normalized fit flux and fluxonium energies in GHz; purcell
    supplies resonator GHz inputs. Temp_range is a positive finite K bracket,
    tolerance is a positive finite log-margin tolerance and max_iter is a
    positive bisection budget. progress controls only the outer search display.
    Return a feasible temperature, or raise ValueError for invalid controls,
    missing positive fit data, invalid calculated limits or an infeasible floor.

    The constraint is ``T1_Purcell(flux, Temp) >= T1_observed`` for every fit point.
    A lower bound is not inferred from Purcell; ``Temp_range[0]`` is only the search
    floor used to verify that a feasible point exists.
    """
    _validate_temperature_search(Temp_range, tolerance, max_iter)
    lower, upper = Temp_range

    fluxs = np.asarray(data.fit.fluxs, dtype=np.float64)
    observed_T1s = np.asarray(data.fit.T1_ns, dtype=np.float64)
    valid = np.isfinite(fluxs) & np.isfinite(observed_T1s) & (observed_T1s > 0.0)
    if not np.any(valid):
        raise ValueError("data.fit must contain at least one finite positive T1 point")
    fit_fluxs = fluxs[valid]
    fit_T1s = observed_T1s[valid]

    pbar = (
        make_pbar(desc="Purcell Temp upper bound", total=max_iter + 2, leave=False)
        if progress
        else None
    )

    def log_margin(Temp: float) -> float:
        purcell_T1s = calculate_purcell_t1_limit(
            data.calibration.context,
            fit_fluxs,
            purcell,
            Temp=Temp,
        )
        if (
            purcell_T1s.shape != fit_T1s.shape
            or np.any(~np.isfinite(purcell_T1s))
            or np.any(purcell_T1s <= 0.0)
        ):
            raise ValueError("Purcell T1 calculation returned invalid values")
        margin = float(np.min(np.log(purcell_T1s / fit_T1s)))
        if pbar is not None:
            pbar.set_description(
                f"Purcell Temp={Temp * 1e3:.2f} mK margin={margin:.3g}"
            )
            pbar.update()
        return margin

    try:
        lower_margin = log_margin(float(lower))
        if lower_margin < 0.0:
            raise ValueError(
                "Purcell T1 is already below observed T1 at Temp_range lower bound"
            )
        upper_margin = log_margin(float(upper))
        if upper_margin >= 0.0:
            return float(upper)

        feasible_Temp = float(lower)
        infeasible_Temp = float(upper)
        for _ in range(max_iter):
            mid_Temp = 0.5 * (feasible_Temp + infeasible_Temp)
            mid_margin = log_margin(mid_Temp)
            if abs(mid_margin) <= tolerance:
                return float(mid_Temp)
            if mid_margin >= 0.0:
                feasible_Temp = mid_Temp
            else:
                infeasible_Temp = mid_Temp
        return float(feasible_Temp)
    finally:
        if pbar is not None:
            pbar.close()


def subtract_relaxation_limit(
    observed_T1_ns: NDArray[np.float64],
    observed_T1err_ns: NDArray[np.float64] | None,
    limit_T1_ns: NDArray[np.float64],
) -> PurcellCorrectionResult:
    """Subtract a relaxation limit from observed T1 in the ns^-1 rate domain.

    observed_T1_ns and limit_T1_ns must be positive finite 1D ns arrays with
    equal shape. observed_T1err_ns supplies matching ns errors, or None for NaN
    errors. Return the inputs, limit rates, propagated intrinsic errors and a
    valid_mask; nonpositive intrinsic rates produce NaN T1/errors, not a raise.
    Invalid T1 values or mismatched shapes raise ValueError. Inputs are not
    mutated, and nonfinite/nonpositive errors propagate as NaN.
    """
    observed_T1s = np.asarray(observed_T1_ns, dtype=np.float64)
    observed_T1errs = (
        np.full_like(observed_T1s, np.nan, dtype=np.float64)
        if observed_T1err_ns is None
        else np.asarray(observed_T1err_ns, dtype=np.float64)
    )
    limit_T1s = np.asarray(limit_T1_ns, dtype=np.float64)
    if observed_T1s.ndim != 1 or limit_T1s.ndim != 1:
        raise ValueError("observed_T1_ns and limit_T1_ns must be 1D arrays")
    if observed_T1errs.shape != observed_T1s.shape:
        raise ValueError("observed_T1err_ns must have the same shape as observed_T1_ns")
    if limit_T1s.shape != observed_T1s.shape:
        raise ValueError("limit_T1_ns must have the same shape as observed_T1_ns")

    observed_rate = relaxation_rates_from_t1(observed_T1s, name="observed T1")
    limit_rate = relaxation_rates_from_t1(limit_T1s, name="limit T1")
    intrinsic_rate = observed_rate - limit_rate
    valid_mask = np.isfinite(intrinsic_rate) & (intrinsic_rate > 0.0)

    intrinsic_T1s = np.divide(
        1.0,
        intrinsic_rate,
        out=np.full_like(intrinsic_rate, np.nan, dtype=np.float64),
        where=valid_mask,
    )
    rate_err = np.divide(
        observed_T1errs,
        observed_T1s**2,
        out=np.full_like(observed_T1errs, np.nan, dtype=np.float64),
        where=np.isfinite(observed_T1errs) & (observed_T1errs > 0.0),
    )
    intrinsic_T1errs = np.divide(
        rate_err,
        intrinsic_rate**2,
        out=np.full_like(rate_err, np.nan, dtype=np.float64),
        where=valid_mask & np.isfinite(rate_err) & (rate_err > 0.0),
    )
    return PurcellCorrectionResult(
        observed_T1_ns=observed_T1s,
        observed_T1err_ns=observed_T1errs,
        intrinsic_T1_ns=intrinsic_T1s,
        intrinsic_T1err_ns=intrinsic_T1errs,
        purcell_T1_ns=limit_T1s,
        purcell_rate_per_ns=limit_rate,
        valid_mask=valid_mask.astype(bool, copy=False),
    )


def _validate_t1_limit(
    T1_ns: NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    values = np.asarray(T1_ns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return values


def relaxation_rates_from_t1(
    T1_ns: NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Convert positive finite 1D T1_ns values to rates in ns^-1.

    name labels ValueError messages for invalid dimensions or T1 values.
    Return a new float64 array with the same shape and order; do not mutate
    T1_ns. No unit rescaling is applied beyond taking the reciprocal.
    """
    T1s = _validate_t1_limit(T1_ns, name=name)
    return np.divide(
        1.0,
        T1s,
        out=np.full_like(T1s, np.nan, dtype=np.float64),
        where=np.isfinite(T1s) & (T1s > 0.0),
    )


def combine_t1_limits(
    *limits: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Combine one or more independent T1 limits by summing ns^-1 rates.

    Each limit is a positive finite 1D ns array of the same shape. Return the
    effective T1 in ns without modifying the inputs. Raise ValueError for no
    limits, invalid values or mismatched shapes; nonfinite/zero total rates
    yield NaN under the existing rate-domain division policy.
    """
    if not limits:
        raise ValueError("at least one T1 limit is required")
    first = np.asarray(limits[0], dtype=np.float64)
    total_rate = np.zeros_like(first, dtype=np.float64)
    for limit in limits:
        rates = relaxation_rates_from_t1(np.asarray(limit, dtype=np.float64), name="T1")
        if rates.shape != first.shape:
            raise ValueError("all T1 limits must have the same shape")
        total_rate += rates
    return np.divide(
        1.0,
        total_rate,
        out=np.full_like(total_rate, np.nan, dtype=np.float64),
        where=np.isfinite(total_rate) & (total_rate > 0.0),
    )


@lru_cache(maxsize=_PURCELL_CACHE_SIZE)
def _calculate_purcell_t1_limit_cached(
    params: tuple[float, float, float],
    fluxs: tuple[float, ...],
    bare_rf: float,
    kappa: float,
    g: float,
    Temp: float,
) -> tuple[float, ...]:
    values = _validate_t1_limit(
        calculate_purcell_t1_vs_flux(
            np.asarray(fluxs, dtype=np.float64),
            bare_rf=bare_rf,
            kappa=kappa,
            g=g,
            Temp=Temp,
            params=params,
            progress=False,
        ),
        name="Purcell T1",
    )
    return tuple(float(value) for value in values)


def _float_tuple_cache_key(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(_float_cache_key(value) for value in values)


def _array_cache_key(values: NDArray[np.float64], *, name: str) -> tuple[float, ...]:
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return tuple(_float_cache_key(value) for value in values)


def _float_cache_key(value: float) -> float:
    return round(float(value), _CACHE_FLOAT_DECIMALS)


def _validate_temperature_search(
    Temp_range: tuple[float, float], tolerance: float, max_iter: int
) -> None:
    """Require a positive temperature bracket, finite tolerance and iteration budget."""
    lower, upper = Temp_range
    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or lower <= 0.0
        or upper <= lower
    ):
        raise ValueError("Temp_range must be positive finite (lower, upper)")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

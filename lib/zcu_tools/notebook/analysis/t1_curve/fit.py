from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from zcu_tools.notebook.analysis.fit_tools import (
    ErrorResolutionResult,
    FluxResidualWeighting,
    FluxResidualWeights,
    MeasurementErrorPolicy,
    build_flux_residual_weights,
    least_squares_cost,
    reduced_chi2_from_cost,
    resolve_measurement_errors,
)
from zcu_tools.progress_bar import make_pbar
from zcu_tools.progress_bar.base import BaseProgressBar
from zcu_tools.simulate.fluxonium import calculate_eff_t1_vs_flux_fast

NoiseParameterName = Literal["Q_cap", "x_qp", "Q_ind"]
ParameterName = Literal["Q_cap", "x_qp", "Q_ind", "Temp"]
ResidualMode = Literal["log", "linear"]

_NOISE_PARAMETER_NAMES: tuple[NoiseParameterName, ...] = ("Q_cap", "x_qp", "Q_ind")
_PARAMETER_NAMES: tuple[ParameterName, ...] = ("Q_cap", "x_qp", "Q_ind", "Temp")
_NOISE_CHANNELS: dict[NoiseParameterName, tuple[str, str]] = {
    "Q_cap": ("t1_capacitive", "Q_cap"),
    "x_qp": ("t1_quasiparticle_tunneling", "x_qp"),
    "Q_ind": ("t1_inductive", "Q_ind"),
}
_TINY_POSITIVE = np.finfo(np.float64).tiny
_DEFAULT_BOUNDS: dict[ParameterName, tuple[float, float]] = {
    "Q_cap": (_TINY_POSITIVE, np.inf),
    "x_qp": (_TINY_POSITIVE, np.inf),
    "Q_ind": (_TINY_POSITIVE, np.inf),
    "Temp": (10e-3, 300e-3),
}


def _default_flux_weights() -> FluxResidualWeights:
    return FluxResidualWeights(
        residual_weights=np.ones(0, dtype=np.float64),
        bin_indices=np.zeros(0, dtype=np.int64),
        bin_counts=np.zeros(0, dtype=np.int64),
        effective_observation_count=0.0,
        mode="sample",
        bin_width=None,
        bin_count=None,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class T1FitParams:
    Temp: float
    Q_cap: float | None = None
    x_qp: float | None = None
    Q_ind: float | None = None


ExtraRelaxationRateFn = Callable[[T1FitParams], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class T1FitResult:
    params: T1FitParams
    stderr: T1FitParams
    fixed: tuple[str, ...]
    free: tuple[str, ...]
    model_T1s: NDArray[np.float64]
    residuals: NDArray[np.float64]
    cost: float
    reduced_chi2: float
    success: bool
    message: str
    optimizer_result: OptimizeResult | None
    T1_error_resolution: ErrorResolutionResult | None = None
    flux_weights: FluxResidualWeights = field(default_factory=_default_flux_weights)


def fit_t1_noise_params(
    fluxs: NDArray[np.float64],
    T1s: NDArray[np.float64],
    params: tuple[float, float, float],
    *,
    init: T1FitParams,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    fixed: Iterable[str] = (),
    T1errs: NDArray[np.float64] | None = None,
    T1_error_policy: MeasurementErrorPolicy | None = None,
    flux_weighting: FluxResidualWeighting | None = None,
    residual_mode: ResidualMode = "log",
    loss: str = "linear",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    cutoff: int = 40,
    qub_dim: int = 20,
    i: int = 1,
    j: int = 0,
    extra_relaxation_rate_fn: ExtraRelaxationRateFn | None = None,
    progress: bool = False,
) -> T1FitResult:
    """Fit active T1 noise parameters against measured T1 vs normalized flux.

    ``fluxs`` are normalized flux values (phi_ext/phi0), ``T1s`` and ``T1errs`` are
    in ns, and ``params`` is ``(EJ, EC, EL)`` in GHz. ``Temp`` is always required;
    ``Q_cap``, ``x_qp``, and ``Q_ind`` are white-listed by providing non-``None``
    values in ``init``.
    """
    data = _validate_data(fluxs, T1s, T1errs, T1_error_policy, flux_weighting)
    if residual_mode not in ("log", "linear"):
        raise ValueError("residual_mode must be 'log' or 'linear'")
    active_names = _active_parameter_names(init)
    fixed_names = _validate_fixed(fixed, active_names)
    free_names: tuple[ParameterName, ...] = tuple(
        name for name in active_names if name not in fixed_names
    )
    lower, upper = _validate_bounds(bounds, active_names)
    init_values = _params_to_values(init, active_names)
    _validate_init(init_values, lower, upper)

    lower_log = np.log(np.array([lower[name] for name in free_names]))
    upper_log = np.log(np.array([upper[name] for name in free_names]))
    init_log = np.log(np.array([init_values[name] for name in free_names]))

    def model(current: T1FitParams) -> NDArray[np.float64]:
        noise_channels = _noise_channels_from_params(current, active_names)
        intrinsic_T1s = calculate_eff_t1_vs_flux_fast(
            params,
            data.fluxs,
            noise_channels,
            current.Temp,
            cutoff=cutoff,
            qub_dim=qub_dim,
            i=i,
            j=j,
        )
        if extra_relaxation_rate_fn is None:
            return intrinsic_T1s
        extra_rates = _validate_extra_relaxation_rates(
            extra_relaxation_rate_fn(current),
            shape=data.T1s.shape,
        )
        return _combine_t1_with_extra_rates(intrinsic_T1s, extra_rates)

    def residual_from_params(current: T1FitParams) -> NDArray[np.float64]:
        return _calc_residuals(
            model(current),
            data,
            residual_mode=residual_mode,
        )

    if not free_names:
        fixed_model = model(init)
        residuals = _calc_residuals(
            fixed_model,
            data,
            residual_mode=residual_mode,
        )
        cost = least_squares_cost(residuals)
        return T1FitResult(
            params=init,
            stderr=_zero_stderr(active_names),
            fixed=fixed_names,
            free=free_names,
            T1_error_resolution=data.T1_error_resolution,
            flux_weights=data.flux_weights,
            model_T1s=fixed_model,
            residuals=residuals,
            cost=cost,
            reduced_chi2=reduced_chi2_from_cost(
                cost, data.flux_weights.effective_observation_count, 0
            ),
            success=True,
            message="all parameters fixed",
            optimizer_result=None,
        )

    pbar: BaseProgressBar | None = None
    best_cost = np.inf

    def residual_from_free(log_free_values: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal best_cost

        current_values = init_values.copy()
        for name, value in zip(free_names, np.exp(log_free_values), strict=True):
            current_values[name] = float(value)
        residuals = residual_from_params(_values_to_params(current_values))
        if pbar is not None:
            best_cost = min(best_cost, least_squares_cost(residuals))
            pbar.set_description(f"T1 fit cost={best_cost:.3g}")
            pbar.update()
        return residuals

    if progress:
        pbar = make_pbar(desc="T1 fit cost=nan", total=max_nfev, leave=False)

    try:
        opt = least_squares(
            residual_from_free,
            init_log,
            bounds=(lower_log, upper_log),
            loss=loss,
            max_nfev=max_nfev,
            ftol=1e-8 if ftol is None else ftol,
            xtol=1e-8 if xtol is None else xtol,
            gtol=1e-8 if gtol is None else gtol,
        )
    finally:
        if pbar is not None:
            pbar.close()

    fit_values = init_values.copy()
    for name, value in zip(free_names, np.exp(opt.x), strict=True):
        fit_values[name] = float(value)
    fit_params = _values_to_params(fit_values)
    fit_model = model(fit_params)
    residuals = _calc_residuals(
        fit_model,
        data,
        residual_mode=residual_mode,
    )
    cost = least_squares_cost(residuals)
    stderr = _estimate_stderr(
        opt,
        fit_values,
        active_names,
        free_names,
        data.flux_weights.effective_observation_count,
    )

    return T1FitResult(
        params=fit_params,
        stderr=stderr,
        fixed=fixed_names,
        free=free_names,
        T1_error_resolution=data.T1_error_resolution,
        flux_weights=data.flux_weights,
        model_T1s=fit_model,
        residuals=residuals,
        cost=cost,
        reduced_chi2=reduced_chi2_from_cost(
            cost, data.flux_weights.effective_observation_count, len(free_names)
        ),
        success=bool(opt.success),
        message=str(opt.message),
        optimizer_result=opt,
    )


@dataclass(frozen=True, slots=True)
class _T1FitData:
    fluxs: NDArray[np.float64]
    T1s: NDArray[np.float64]
    T1errs: NDArray[np.float64] | None
    T1_error_resolution: ErrorResolutionResult | None
    flux_weights: FluxResidualWeights


def _validate_data(
    fluxs: NDArray[np.float64],
    T1s: NDArray[np.float64],
    T1errs: NDArray[np.float64] | None,
    T1_error_policy: MeasurementErrorPolicy | None,
    flux_weighting: FluxResidualWeighting | None,
) -> _T1FitData:
    fluxs_arr = np.asarray(fluxs, dtype=np.float64)
    T1s_arr = np.asarray(T1s, dtype=np.float64)
    if fluxs_arr.ndim != 1 or T1s_arr.ndim != 1:
        raise ValueError("fluxs and T1s must be 1D arrays")
    if fluxs_arr.shape != T1s_arr.shape:
        raise ValueError("fluxs and T1s must have the same shape")
    if fluxs_arr.size == 0:
        raise ValueError("at least one T1 sample is required")
    if not np.all(np.isfinite(fluxs_arr)):
        raise ValueError("fluxs must be finite")
    if not np.all(np.isfinite(T1s_arr)) or np.any(T1s_arr <= 0.0):
        raise ValueError("T1s must be finite and positive")

    flux_weights = build_flux_residual_weights(
        fluxs_arr,
        flux_weighting,
        sample_count=len(T1s_arr),
    )
    if T1errs is None:
        return _T1FitData(
            fluxs=fluxs_arr,
            T1s=T1s_arr,
            T1errs=None,
            T1_error_resolution=None,
            flux_weights=flux_weights,
        )

    T1errs_arr = np.asarray(T1errs, dtype=np.float64)
    if T1errs_arr.shape != T1s_arr.shape:
        raise ValueError("T1errs must have the same shape as T1s")
    valid_err = np.isnan(T1errs_arr) | (np.isfinite(T1errs_arr) & (T1errs_arr > 0.0))
    if not np.all(valid_err):
        raise ValueError("T1errs must be positive finite values or NaN")
    T1_error_resolution = resolve_measurement_errors(
        T1s_arr,
        T1errs_arr,
        policy=T1_error_policy,
        flux_weights=flux_weights,
        name="T1errs",
    )
    return _T1FitData(
        fluxs=fluxs_arr,
        T1s=T1s_arr,
        T1errs=T1_error_resolution.effective_errors,
        T1_error_resolution=T1_error_resolution,
        flux_weights=flux_weights,
    )


def _active_parameter_names(init: T1FitParams) -> tuple[ParameterName, ...]:
    active_names: list[ParameterName] = []
    for name in _PARAMETER_NAMES:
        if name == "Temp" or _param_value(init, name) is not None:
            active_names.append(name)
    if len(active_names) == 1:
        raise ValueError("at least one T1 noise parameter must be provided")
    return tuple(active_names)


def _validate_fixed(
    fixed: Iterable[str], active_names: tuple[ParameterName, ...]
) -> tuple[ParameterName, ...]:
    fixed_list = list(fixed)
    if len(set(fixed_list)) != len(fixed_list):
        raise ValueError("fixed contains duplicate parameter names")
    unknown = set(fixed_list) - set(_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown fixed parameter(s): {sorted(unknown)}")
    inactive = set(fixed_list) - set(active_names)
    if inactive:
        raise ValueError(f"fixed contains inactive parameter(s): {sorted(inactive)}")
    return tuple(name for name in active_names if name in fixed_list)


def _validate_bounds(
    bounds: Mapping[str, tuple[float, float]] | None,
    active_names: tuple[ParameterName, ...],
) -> tuple[dict[ParameterName, float], dict[ParameterName, float]]:
    merged = dict(_DEFAULT_BOUNDS)
    if bounds is not None:
        unknown = set(bounds) - set(_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"unknown bound parameter(s): {sorted(unknown)}")
        inactive = set(bounds) - set(active_names)
        if inactive:
            raise ValueError(
                f"bounds contain inactive parameter(s): {sorted(inactive)}"
            )
        merged.update(bounds)  # type: ignore[arg-type]

    lower: dict[ParameterName, float] = {}
    upper: dict[ParameterName, float] = {}
    for name in active_names:
        lower[name] = float(merged[name][0])
        upper[name] = float(merged[name][1])
    lower_arr = np.array(list(lower.values()), dtype=np.float64)
    upper_arr = np.array(list(upper.values()), dtype=np.float64)
    if np.any(~np.isfinite(lower_arr)) or np.any(lower_arr <= 0.0):
        raise ValueError("all lower bounds must be finite and positive")
    if np.any(np.isnan(upper_arr)) or np.any(upper_arr <= 0.0):
        raise ValueError("all upper bounds must be positive or infinity")
    if np.any(lower_arr >= upper_arr):
        raise ValueError("lower bounds must be lower than upper bounds")
    return lower, upper


def _validate_init(
    init_values: Mapping[ParameterName, float],
    lower: Mapping[ParameterName, float],
    upper: Mapping[ParameterName, float],
) -> None:
    values = np.array(list(init_values.values()), dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("initial T1 fit parameters must be finite and positive")
    for name, value in init_values.items():
        if value < lower[name] or value > upper[name]:
            raise ValueError("initial T1 fit parameters must be within bounds")


def _calc_residuals(
    model_T1s: NDArray[np.float64],
    data: _T1FitData,
    *,
    residual_mode: ResidualMode,
) -> NDArray[np.float64]:
    model_T1s = np.asarray(model_T1s, dtype=np.float64)
    if model_T1s.shape != data.T1s.shape:
        raise ValueError("model T1 output shape does not match T1s")
    if np.any(~np.isfinite(model_T1s)) or np.any(model_T1s <= 0.0):
        return np.full_like(data.T1s, 1e12, dtype=np.float64)

    if residual_mode == "log":
        residuals = np.log(model_T1s) - np.log(data.T1s)
        if data.T1errs is not None:
            weights = _nan_as_unweighted(data.T1errs / data.T1s)
            residuals = residuals / weights
    elif residual_mode == "linear":
        residuals = model_T1s - data.T1s
        if data.T1errs is not None:
            residuals = residuals / _nan_as_unweighted(data.T1errs)
    else:
        raise ValueError("residual_mode must be 'log' or 'linear'")

    return (residuals * data.flux_weights.residual_weights).astype(
        np.float64, copy=False
    )


def _combine_t1_with_extra_rates(
    intrinsic_T1s: NDArray[np.float64],
    extra_rates: NDArray[np.float64],
) -> NDArray[np.float64]:
    intrinsic_arr = np.asarray(intrinsic_T1s, dtype=np.float64)
    rates = np.divide(
        1.0,
        intrinsic_arr,
        out=np.full_like(intrinsic_arr, np.nan, dtype=np.float64),
        where=np.isfinite(intrinsic_arr) & (intrinsic_arr > 0.0),
    )
    rates += extra_rates
    return np.divide(
        1.0,
        rates,
        out=np.full_like(rates, np.nan, dtype=np.float64),
        where=np.isfinite(rates) & (rates > 0.0),
    )


def _validate_extra_relaxation_rates(
    rates: NDArray[np.float64],
    *,
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    rates_arr = np.asarray(rates, dtype=np.float64)
    if rates_arr.shape != shape:
        raise ValueError("extra relaxation rate output shape does not match T1s")
    if np.any(~np.isfinite(rates_arr)) or np.any(rates_arr < 0.0):
        raise ValueError("extra relaxation rates must be finite and non-negative")
    return rates_arr


def _nan_as_unweighted(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(np.isnan(values), 1.0, values)


def _estimate_stderr(
    opt: OptimizeResult,
    fit_values: Mapping[ParameterName, float],
    active_names: tuple[ParameterName, ...],
    free_names: tuple[ParameterName, ...],
    observation_count: float,
) -> T1FitParams:
    stderr_values: dict[ParameterName, float] = {name: 0.0 for name in active_names}
    n_free = len(free_names)
    if n_free == 0:
        return _values_to_params(stderr_values)

    try:
        jac = np.asarray(opt.jac, dtype=np.float64)
        cov_log = np.linalg.pinv(jac.T @ jac)
        cov_log *= reduced_chi2_from_cost(float(opt.cost), observation_count, n_free)
        log_stderr = np.sqrt(np.maximum(np.diag(cov_log), 0.0))
        for name, value in zip(free_names, log_stderr, strict=True):
            stderr_values[name] = fit_values[name] * float(value)
    except Exception:
        for name in free_names:
            stderr_values[name] = np.inf

    return _values_to_params(stderr_values)


def _noise_channels_from_params(
    params: T1FitParams, active_names: tuple[ParameterName, ...]
) -> list[tuple[str, dict[str, float]]]:
    noise_channels: list[tuple[str, dict[str, float]]] = []
    for name in _NOISE_PARAMETER_NAMES:
        if name not in active_names:
            continue
        value = _param_value(params, name)
        if value is None:
            raise ValueError(f"active T1 fit parameter {name} is missing")
        channel_name, option_name = _NOISE_CHANNELS[name]
        noise_channels.append((channel_name, {option_name: value}))
    return noise_channels


def _param_value(params: T1FitParams, name: ParameterName) -> float | None:
    if name == "Q_cap":
        return params.Q_cap
    if name == "x_qp":
        return params.x_qp
    if name == "Q_ind":
        return params.Q_ind
    return params.Temp


def _params_to_values(
    params: T1FitParams, active_names: tuple[ParameterName, ...]
) -> dict[ParameterName, float]:
    values: dict[ParameterName, float] = {}
    for name in active_names:
        value = _param_value(params, name)
        if value is None:
            raise ValueError(f"active T1 fit parameter {name} is missing")
        values[name] = float(value)
    return values


def _values_to_params(values: Mapping[ParameterName, float]) -> T1FitParams:
    return T1FitParams(
        Temp=float(values["Temp"]),
        Q_cap=_optional_value(values, "Q_cap"),
        x_qp=_optional_value(values, "x_qp"),
        Q_ind=_optional_value(values, "Q_ind"),
    )


def _optional_value(
    values: Mapping[ParameterName, float], name: NoiseParameterName
) -> float | None:
    value = values.get(name)
    return None if value is None else float(value)


def _zero_stderr(active_names: tuple[ParameterName, ...]) -> T1FitParams:
    values: dict[ParameterName, float] = {name: 0.0 for name in active_names}
    return _values_to_params(values)


__all__ = ["T1FitParams", "T1FitResult", "fit_t1_noise_params"]

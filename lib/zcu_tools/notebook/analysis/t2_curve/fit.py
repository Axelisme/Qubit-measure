from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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

ParameterName = Literal["A_phi", "n_th"]
ResidualMode = Literal["gamma", "log_t2"]

_PARAMETER_NAMES: tuple[ParameterName, ...] = ("A_phi", "n_th")
_TINY_POSITIVE = np.finfo(np.float64).tiny
_DEFAULT_BOUNDS: dict[ParameterName, tuple[float, float]] = {
    "A_phi": (_TINY_POSITIVE, np.inf),
    "n_th": (_TINY_POSITIVE, np.inf),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class T2FitParams:
    A_phi: float | None = None
    n_th: float | None = None


@dataclass(frozen=True, slots=True)
class T2FitResult:
    params: T2FitParams
    stderr: T2FitParams
    fixed: tuple[str, ...]
    free: tuple[str, ...]
    gamma_phi_obs: NDArray[np.float64]
    gamma_phi_err: NDArray[np.float64] | None
    T1_error_resolution: ErrorResolutionResult | None
    T2_error_resolution: ErrorResolutionResult | None
    flux_weights: FluxResidualWeights
    gamma_phi_flux: NDArray[np.float64]
    gamma_phi_photon: NDArray[np.float64]
    model_gamma_phi: NDArray[np.float64]
    model_T2s: NDArray[np.float64]
    residuals: NDArray[np.float64]
    cost: float
    reduced_chi2: float
    success: bool
    message: str
    optimizer_result: OptimizeResult | None


def flux_noise_gamma_phi_per_us(
    A_phi: float,
    domega_dflux: NDArray[np.float64],
) -> NDArray[np.float64]:
    """First-order 1/f echo dephasing rate in 1/us."""
    return (
        np.sqrt(np.log(2.0))
        * float(A_phi)
        * np.abs(np.asarray(domega_dflux, dtype=np.float64))
    )


def thermal_photon_gamma_phi_per_us(
    n_th: float | NDArray[np.float64],
    *,
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """Readout thermal-photon shot-noise dephasing rate in 1/us.

    Inputs use the common ``kappa/2pi`` and ``chi/2pi`` MHz convention. Since
    1 MHz = 1 cycle/us, the angular rates are obtained by multiplying by 2*pi.
    """
    kappa = 2.0 * np.pi * float(kappa_over_2pi_mhz)
    chi = 2.0 * np.pi * np.asarray(chi_over_2pi_mhz, dtype=np.float64)
    gamma_per_photon = kappa * chi**2 / (kappa**2 + chi**2)
    gamma = np.asarray(n_th, dtype=np.float64) * gamma_per_photon
    if np.ndim(gamma) == 0:
        return float(gamma)
    return np.asarray(gamma, dtype=np.float64)


def thermal_photon_t2_limit_us(
    n_th: float | NDArray[np.float64],
    *,
    T1_us: float | NDArray[np.float64],
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    gamma_phi = thermal_photon_gamma_phi_per_us(
        n_th,
        kappa_over_2pi_mhz=kappa_over_2pi_mhz,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
    )
    t2_limit = 1.0 / (1.0 / (2.0 * np.asarray(T1_us)) + gamma_phi)
    if np.ndim(t2_limit) == 0:
        return float(t2_limit)
    return np.asarray(t2_limit, dtype=np.float64)


def equivalent_n_th_from_t2(
    *,
    T1_us: float,
    T2_us: float,
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: float,
) -> float:
    gamma_phi = 1.0 / float(T2_us) - 1.0 / (2.0 * float(T1_us))
    gamma_per_photon = thermal_photon_gamma_phi_per_us(
        1.0,
        kappa_over_2pi_mhz=kappa_over_2pi_mhz,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
    )
    if gamma_phi <= 0.0:
        return float("nan")
    return float(gamma_phi / gamma_per_photon)


def fit_t2_noise_params(
    T1s: NDArray[np.float64],
    T2s: NDArray[np.float64],
    domega_dflux: NDArray[np.float64],
    chi_over_2pi_mhz: NDArray[np.float64],
    *,
    kappa_over_2pi_mhz: float,
    init: T2FitParams,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    fixed: Iterable[str] = (),
    T1errs: NDArray[np.float64] | None = None,
    T2errs: NDArray[np.float64] | None = None,
    fluxs: NDArray[np.float64] | None = None,
    T1_error_policy: MeasurementErrorPolicy | None = None,
    T2_error_policy: MeasurementErrorPolicy | None = None,
    flux_weighting: FluxResidualWeighting | None = None,
    residual_mode: ResidualMode = "gamma",
    loss: str = "linear",
    max_nfev: int | None = None,
    ftol: float | None = None,
    xtol: float | None = None,
    gtol: float | None = None,
    progress: bool = False,
) -> T2FitResult:
    """Fit active T2 pure-dephasing noise parameters.

    ``T1s``, ``T2s``, and their errors are in us. ``domega_dflux`` is in
    rad/us/Phi0. ``kappa_over_2pi_mhz`` and ``chi_over_2pi_mhz`` use MHz.
    Active parameters are white-listed by non-``None`` values in ``init``.
    """
    data = _validate_data(
        T1s,
        T2s,
        domega_dflux,
        chi_over_2pi_mhz,
        T1errs,
        T2errs,
        fluxs,
        T1_error_policy,
        T2_error_policy,
        flux_weighting,
    )
    if residual_mode not in ("gamma", "log_t2"):
        raise ValueError("residual_mode must be 'gamma' or 'log_t2'")
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

    def model(
        current: T2FitParams,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        values = _params_to_values(current, active_names)
        gamma_flux = (
            flux_noise_gamma_phi_per_us(values["A_phi"], data.domega_dflux)
            if "A_phi" in active_names
            else np.zeros_like(data.T2s)
        )
        gamma_photon = (
            np.asarray(
                thermal_photon_gamma_phi_per_us(
                    values["n_th"],
                    kappa_over_2pi_mhz=kappa_over_2pi_mhz,
                    chi_over_2pi_mhz=data.chi_over_2pi_mhz,
                ),
                dtype=np.float64,
            )
            if "n_th" in active_names
            else np.zeros_like(data.T2s)
        )
        gamma_model = gamma_flux + gamma_photon
        model_T2s = 1.0 / (1.0 / (2.0 * data.T1s) + gamma_model)
        return gamma_flux, gamma_photon, gamma_model, model_T2s

    def residual_from_params(current: T2FitParams) -> NDArray[np.float64]:
        _, _, gamma_model, model_T2s = model(current)
        return _calc_residuals(
            gamma_model,
            model_T2s,
            data,
            residual_mode=residual_mode,
        )

    if not free_names:
        gamma_flux, gamma_photon, gamma_model, model_T2s = model(init)
        residuals = _calc_residuals(
            gamma_model,
            model_T2s,
            data,
            residual_mode=residual_mode,
        )
        cost = least_squares_cost(residuals)
        return T2FitResult(
            params=init,
            stderr=_zero_stderr(active_names),
            fixed=fixed_names,
            free=free_names,
            gamma_phi_obs=data.gamma_phi_obs,
            gamma_phi_err=data.gamma_phi_err,
            T1_error_resolution=data.T1_error_resolution,
            T2_error_resolution=data.T2_error_resolution,
            flux_weights=data.flux_weights,
            gamma_phi_flux=gamma_flux,
            gamma_phi_photon=gamma_photon,
            model_gamma_phi=gamma_model,
            model_T2s=model_T2s,
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
            pbar.set_description(f"T2 fit cost={best_cost:.3g}")
            pbar.update()
        return residuals

    if progress:
        pbar = make_pbar(desc="T2 fit cost=nan", total=max_nfev, leave=False)

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
    gamma_flux, gamma_photon, gamma_model, model_T2s = model(fit_params)
    residuals = _calc_residuals(
        gamma_model,
        model_T2s,
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

    return T2FitResult(
        params=fit_params,
        stderr=stderr,
        fixed=fixed_names,
        free=free_names,
        gamma_phi_obs=data.gamma_phi_obs,
        gamma_phi_err=data.gamma_phi_err,
        T1_error_resolution=data.T1_error_resolution,
        T2_error_resolution=data.T2_error_resolution,
        flux_weights=data.flux_weights,
        gamma_phi_flux=gamma_flux,
        gamma_phi_photon=gamma_photon,
        model_gamma_phi=gamma_model,
        model_T2s=model_T2s,
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
class _T2FitData:
    T1s: NDArray[np.float64]
    T2s: NDArray[np.float64]
    domega_dflux: NDArray[np.float64]
    chi_over_2pi_mhz: NDArray[np.float64]
    T1errs: NDArray[np.float64] | None
    T2errs: NDArray[np.float64] | None
    gamma_phi_obs: NDArray[np.float64]
    gamma_phi_err: NDArray[np.float64] | None
    T1_error_resolution: ErrorResolutionResult | None
    T2_error_resolution: ErrorResolutionResult | None
    flux_weights: FluxResidualWeights


def _validate_data(
    T1s: NDArray[np.float64],
    T2s: NDArray[np.float64],
    domega_dflux: NDArray[np.float64],
    chi_over_2pi_mhz: NDArray[np.float64],
    T1errs: NDArray[np.float64] | None,
    T2errs: NDArray[np.float64] | None,
    fluxs: NDArray[np.float64] | None,
    T1_error_policy: MeasurementErrorPolicy | None,
    T2_error_policy: MeasurementErrorPolicy | None,
    flux_weighting: FluxResidualWeighting | None,
) -> _T2FitData:
    T1s_arr = np.asarray(T1s, dtype=np.float64)
    T2s_arr = np.asarray(T2s, dtype=np.float64)
    domega_arr = np.asarray(domega_dflux, dtype=np.float64)
    chi_arr = np.asarray(chi_over_2pi_mhz, dtype=np.float64)
    arrays = {
        "T1s": T1s_arr,
        "T2s": T2s_arr,
        "domega_dflux": domega_arr,
        "chi_over_2pi_mhz": chi_arr,
    }
    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be a 1D array")
        if values.shape != T1s_arr.shape:
            raise ValueError("all T2 fit arrays must have the same shape")
    if T1s_arr.size == 0:
        raise ValueError("at least one T2 sample is required")
    if not np.all(np.isfinite(T1s_arr)) or np.any(T1s_arr <= 0.0):
        raise ValueError("T1s must be finite and positive")
    if not np.all(np.isfinite(T2s_arr)) or np.any(T2s_arr <= 0.0):
        raise ValueError("T2s must be finite and positive")
    if not np.all(np.isfinite(domega_arr)):
        raise ValueError("domega_dflux must be finite")
    if not np.all(np.isfinite(chi_arr)) or np.any(chi_arr <= 0.0):
        raise ValueError("chi_over_2pi_mhz must be finite and positive")

    fluxs_arr = _validate_fluxs(fluxs, T1s_arr.shape)
    flux_weights = build_flux_residual_weights(
        fluxs_arr,
        flux_weighting,
        sample_count=len(T1s_arr),
    )
    T1errs_arr = _validate_error("T1errs", T1errs, T1s_arr.shape)
    T2errs_arr = _validate_error("T2errs", T2errs, T2s_arr.shape)
    T1_error_resolution = _resolve_optional_errors(
        "T1errs",
        T1s_arr,
        T1errs_arr,
        T1_error_policy,
        flux_weights,
    )
    T2_error_resolution = _resolve_optional_errors(
        "T2errs",
        T2s_arr,
        T2errs_arr,
        T2_error_policy,
        flux_weights,
    )
    effective_T1errs = (
        None if T1_error_resolution is None else T1_error_resolution.effective_errors
    )
    effective_T2errs = (
        None if T2_error_resolution is None else T2_error_resolution.effective_errors
    )
    gamma_phi_obs = 1.0 / T2s_arr - 1.0 / (2.0 * T1s_arr)
    if not np.all(np.isfinite(gamma_phi_obs)) or np.any(gamma_phi_obs <= 0.0):
        raise ValueError("observed pure-dephasing rates must be finite and positive")
    gamma_phi_err = _gamma_phi_err(T1s_arr, T2s_arr, effective_T1errs, effective_T2errs)
    return _T2FitData(
        T1s=T1s_arr,
        T2s=T2s_arr,
        domega_dflux=domega_arr,
        chi_over_2pi_mhz=chi_arr,
        T1errs=effective_T1errs,
        T2errs=effective_T2errs,
        gamma_phi_obs=gamma_phi_obs,
        gamma_phi_err=gamma_phi_err,
        T1_error_resolution=T1_error_resolution,
        T2_error_resolution=T2_error_resolution,
        flux_weights=flux_weights,
    )


def _validate_error(
    name: str,
    values: NDArray[np.float64] | None,
    shape: tuple[int, ...],
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    err = np.asarray(values, dtype=np.float64)
    if err.shape != shape:
        raise ValueError(f"{name} must have the same shape as T2s")
    valid = np.isnan(err) | (np.isfinite(err) & (err > 0.0))
    if not np.all(valid):
        raise ValueError(f"{name} must be positive finite values or NaN")
    return err


def _validate_fluxs(
    values: NDArray[np.float64] | None,
    shape: tuple[int, ...],
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    fluxs = np.asarray(values, dtype=np.float64)
    if fluxs.shape != shape:
        raise ValueError("fluxs must have the same shape as T2s")
    if fluxs.ndim != 1:
        raise ValueError("fluxs must be a 1D array")
    if not np.all(np.isfinite(fluxs)):
        raise ValueError("fluxs must be finite")
    return fluxs


def _resolve_optional_errors(
    name: str,
    values: NDArray[np.float64],
    errors: NDArray[np.float64] | None,
    policy: MeasurementErrorPolicy | None,
    flux_weights: FluxResidualWeights,
) -> ErrorResolutionResult | None:
    if errors is None:
        return None
    return resolve_measurement_errors(
        values,
        errors,
        policy=policy,
        flux_weights=flux_weights,
        name=name,
    )


def _gamma_phi_err(
    T1s: NDArray[np.float64],
    T2s: NDArray[np.float64],
    T1errs: NDArray[np.float64] | None,
    T2errs: NDArray[np.float64] | None,
) -> NDArray[np.float64] | None:
    if T1errs is None and T2errs is None:
        return None
    err_sq = np.zeros_like(T2s, dtype=np.float64)
    if T2errs is not None:
        err_sq += (T2errs / T2s**2) ** 2
    if T1errs is not None:
        err_sq += (0.5 * T1errs / T1s**2) ** 2
    return np.sqrt(err_sq)


def _active_parameter_names(init: T2FitParams) -> tuple[ParameterName, ...]:
    active_names: list[ParameterName] = []
    for name in _PARAMETER_NAMES:
        if _param_value(init, name) is not None:
            active_names.append(name)
    if not active_names:
        raise ValueError("at least one T2 noise parameter must be provided")
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
        raise ValueError("initial T2 fit parameters must be finite and positive")
    for name, value in init_values.items():
        if value < lower[name] or value > upper[name]:
            raise ValueError("initial T2 fit parameters must be within bounds")


def _calc_residuals(
    model_gamma_phi: NDArray[np.float64],
    model_T2s: NDArray[np.float64],
    data: _T2FitData,
    *,
    residual_mode: ResidualMode,
) -> NDArray[np.float64]:
    if np.any(~np.isfinite(model_gamma_phi)) or np.any(model_gamma_phi < 0.0):
        return np.full_like(data.T2s, 1e12, dtype=np.float64)
    if np.any(~np.isfinite(model_T2s)) or np.any(model_T2s <= 0.0):
        return np.full_like(data.T2s, 1e12, dtype=np.float64)

    if residual_mode == "gamma":
        residuals = model_gamma_phi - data.gamma_phi_obs
        if data.gamma_phi_err is not None:
            residuals = residuals / _nan_as_unweighted(data.gamma_phi_err)
    elif residual_mode == "log_t2":
        residuals = np.log(model_T2s) - np.log(data.T2s)
        if data.T2errs is not None:
            residuals = residuals / _nan_as_unweighted(data.T2errs / data.T2s)
    else:
        raise ValueError("residual_mode must be 'gamma' or 'log_t2'")
    return (residuals * data.flux_weights.residual_weights).astype(
        np.float64, copy=False
    )


def _nan_as_unweighted(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(np.isnan(values), 1.0, values)


def _estimate_stderr(
    opt: OptimizeResult,
    fit_values: Mapping[ParameterName, float],
    active_names: tuple[ParameterName, ...],
    free_names: tuple[ParameterName, ...],
    observation_count: float,
) -> T2FitParams:
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


def _param_value(params: T2FitParams, name: ParameterName) -> float | None:
    if name == "A_phi":
        return params.A_phi
    return params.n_th


def _params_to_values(
    params: T2FitParams, active_names: tuple[ParameterName, ...]
) -> dict[ParameterName, float]:
    values: dict[ParameterName, float] = {}
    for name in active_names:
        value = _param_value(params, name)
        if value is None:
            raise ValueError(f"active T2 fit parameter {name} is missing")
        values[name] = float(value)
    return values


def _values_to_params(values: Mapping[ParameterName, float]) -> T2FitParams:
    return T2FitParams(
        A_phi=_optional_value(values, "A_phi"),
        n_th=_optional_value(values, "n_th"),
    )


def _optional_value(
    values: Mapping[ParameterName, float], name: ParameterName
) -> float | None:
    value = values.get(name)
    return None if value is None else float(value)


def _zero_stderr(active_names: tuple[ParameterName, ...]) -> T2FitParams:
    values: dict[ParameterName, float] = {name: 0.0 for name in active_names}
    return _values_to_params(values)


__all__ = [
    "T2FitParams",
    "T2FitResult",
    "equivalent_n_th_from_t2",
    "fit_t2_noise_params",
    "flux_noise_gamma_phi_per_us",
    "thermal_photon_gamma_phi_per_us",
    "thermal_photon_t2_limit_us",
]

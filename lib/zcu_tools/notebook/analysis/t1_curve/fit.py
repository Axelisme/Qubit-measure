from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, least_squares

from zcu_tools.progress_bar import make_pbar
from zcu_tools.progress_bar.base import BaseProgressBar
from zcu_tools.simulate.fluxonium import calculate_eff_t1_vs_flux_fast

ParameterName = Literal["Q_cap", "x_qp", "Q_ind", "Temp"]
ResidualMode = Literal["log", "linear"]

_PARAMETER_NAMES: tuple[ParameterName, ...] = ("Q_cap", "x_qp", "Q_ind", "Temp")
_TINY_POSITIVE = np.finfo(np.float64).tiny
_DEFAULT_BOUNDS: dict[ParameterName, tuple[float, float]] = {
    "Q_cap": (_TINY_POSITIVE, np.inf),
    "x_qp": (_TINY_POSITIVE, np.inf),
    "Q_ind": (_TINY_POSITIVE, np.inf),
    "Temp": (10e-3, 300e-3),
}


@dataclass(frozen=True, slots=True)
class T1FitParams:
    Q_cap: float
    x_qp: float
    Q_ind: float
    Temp: float


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


def fit_t1_noise_params(
    fluxs: NDArray[np.float64],
    T1s: NDArray[np.float64],
    params: tuple[float, float, float],
    *,
    init: T1FitParams,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    fixed: Iterable[str] = (),
    T1errs: NDArray[np.float64] | None = None,
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
    progress: bool = False,
) -> T1FitResult:
    """Fit Q_cap, x_qp, Q_ind, and Temp against measured T1 vs normalized flux.

    ``fluxs`` are normalized flux values (phi_ext/phi0), ``T1s`` and ``T1errs`` are
    in ns, and ``params`` is ``(EJ, EC, EL)`` in GHz.
    """
    fluxs_arr, T1s_arr, T1errs_arr = _validate_data(fluxs, T1s, T1errs)
    if residual_mode not in ("log", "linear"):
        raise ValueError("residual_mode must be 'log' or 'linear'")
    fixed_names = _validate_fixed(fixed)
    free_names = tuple(name for name in _PARAMETER_NAMES if name not in fixed_names)
    lower, upper = _validate_bounds(bounds)
    init_values = _params_to_array(init)
    _validate_init(init_values, lower, upper)

    free_mask = np.array([name in free_names for name in _PARAMETER_NAMES])
    lower_log = np.log(lower[free_mask])
    upper_log = np.log(upper[free_mask])
    init_log = np.log(init_values[free_mask])

    def model(current: T1FitParams) -> NDArray[np.float64]:
        noise_channels = [
            ("t1_capacitive", {"Q_cap": current.Q_cap}),
            ("t1_quasiparticle_tunneling", {"x_qp": current.x_qp}),
            ("t1_inductive", {"Q_ind": current.Q_ind}),
        ]
        return calculate_eff_t1_vs_flux_fast(
            params,
            fluxs_arr,
            noise_channels,
            current.Temp,
            cutoff=cutoff,
            qub_dim=qub_dim,
            i=i,
            j=j,
        )

    def residual_from_params(current: T1FitParams) -> NDArray[np.float64]:
        return _calc_residuals(
            model(current), T1s_arr, T1errs_arr, residual_mode=residual_mode
        )

    if not free_names:
        fixed_model = model(init)
        residuals = _calc_residuals(
            fixed_model, T1s_arr, T1errs_arr, residual_mode=residual_mode
        )
        cost = _cost(residuals)
        return T1FitResult(
            params=init,
            stderr=T1FitParams(0.0, 0.0, 0.0, 0.0),
            fixed=fixed_names,
            free=free_names,
            model_T1s=fixed_model,
            residuals=residuals,
            cost=cost,
            reduced_chi2=_reduced_chi2(cost, len(residuals), 0),
            success=True,
            message="all parameters fixed",
            optimizer_result=None,
        )

    pbar: BaseProgressBar | None = None
    best_cost = np.inf

    def residual_from_free(log_free_values: NDArray[np.float64]) -> NDArray[np.float64]:
        nonlocal best_cost

        current_values = init_values.copy()
        current_values[free_mask] = np.exp(log_free_values)
        residuals = residual_from_params(_array_to_params(current_values))
        if pbar is not None:
            best_cost = min(best_cost, _cost(residuals))
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
    fit_values[free_mask] = np.exp(opt.x)
    fit_params = _array_to_params(fit_values)
    fit_model = model(fit_params)
    residuals = _calc_residuals(
        fit_model, T1s_arr, T1errs_arr, residual_mode=residual_mode
    )
    cost = _cost(residuals)
    stderr = _estimate_stderr(opt, fit_values, free_mask, len(residuals))

    return T1FitResult(
        params=fit_params,
        stderr=stderr,
        fixed=fixed_names,
        free=free_names,
        model_T1s=fit_model,
        residuals=residuals,
        cost=cost,
        reduced_chi2=_reduced_chi2(cost, len(residuals), len(free_names)),
        success=bool(opt.success),
        message=str(opt.message),
        optimizer_result=opt,
    )


def _validate_data(
    fluxs: NDArray[np.float64],
    T1s: NDArray[np.float64],
    T1errs: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
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

    if T1errs is None:
        return fluxs_arr, T1s_arr, None

    T1errs_arr = np.asarray(T1errs, dtype=np.float64)
    if T1errs_arr.shape != T1s_arr.shape:
        raise ValueError("T1errs must have the same shape as T1s")
    valid_err = np.isnan(T1errs_arr) | (np.isfinite(T1errs_arr) & (T1errs_arr > 0.0))
    if not np.all(valid_err):
        raise ValueError("T1errs must be positive finite values or NaN")
    return fluxs_arr, T1s_arr, T1errs_arr


def _validate_fixed(fixed: Iterable[str]) -> tuple[ParameterName, ...]:
    fixed_list = list(fixed)
    if len(set(fixed_list)) != len(fixed_list):
        raise ValueError("fixed contains duplicate parameter names")
    unknown = set(fixed_list) - set(_PARAMETER_NAMES)
    if unknown:
        raise ValueError(f"unknown fixed parameter(s): {sorted(unknown)}")
    return tuple(name for name in _PARAMETER_NAMES if name in fixed_list)


def _validate_bounds(
    bounds: Mapping[str, tuple[float, float]] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    merged = dict(_DEFAULT_BOUNDS)
    if bounds is not None:
        unknown = set(bounds) - set(_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"unknown bound parameter(s): {sorted(unknown)}")
        merged.update(bounds)  # type: ignore[arg-type]

    lower = np.array([merged[name][0] for name in _PARAMETER_NAMES], dtype=np.float64)
    upper = np.array([merged[name][1] for name in _PARAMETER_NAMES], dtype=np.float64)
    if np.any(~np.isfinite(lower)) or np.any(lower <= 0.0):
        raise ValueError("all lower bounds must be finite and positive")
    if np.any(np.isnan(upper)) or np.any(upper <= 0.0):
        raise ValueError("all upper bounds must be positive or infinity")
    if np.any(lower >= upper):
        raise ValueError("lower bounds must be lower than upper bounds")
    return lower, upper


def _validate_init(
    init_values: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> None:
    if np.any(~np.isfinite(init_values)) or np.any(init_values <= 0.0):
        raise ValueError("initial T1 fit parameters must be finite and positive")
    if np.any(init_values < lower) or np.any(init_values > upper):
        raise ValueError("initial T1 fit parameters must be within bounds")


def _calc_residuals(
    model_T1s: NDArray[np.float64],
    T1s: NDArray[np.float64],
    T1errs: NDArray[np.float64] | None,
    *,
    residual_mode: ResidualMode,
) -> NDArray[np.float64]:
    model_T1s = np.asarray(model_T1s, dtype=np.float64)
    if model_T1s.shape != T1s.shape:
        raise ValueError("model T1 output shape does not match T1s")
    if np.any(~np.isfinite(model_T1s)) or np.any(model_T1s <= 0.0):
        return np.full_like(T1s, 1e12, dtype=np.float64)

    if residual_mode == "log":
        residuals = np.log(model_T1s) - np.log(T1s)
        if T1errs is not None:
            weights = _nan_as_unweighted(T1errs / T1s)
            residuals = residuals / weights
    elif residual_mode == "linear":
        residuals = model_T1s - T1s
        if T1errs is not None:
            residuals = residuals / _nan_as_unweighted(T1errs)
    else:
        raise ValueError("residual_mode must be 'log' or 'linear'")

    return residuals.astype(np.float64, copy=False)


def _nan_as_unweighted(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(np.isnan(values), 1.0, values)


def _estimate_stderr(
    opt: OptimizeResult,
    fit_values: NDArray[np.float64],
    free_mask: NDArray[np.bool_],
    n_residuals: int,
) -> T1FitParams:
    stderr = np.zeros(len(_PARAMETER_NAMES), dtype=np.float64)
    n_free = int(np.sum(free_mask))
    if n_free == 0:
        return _array_to_params(stderr)

    try:
        jac = np.asarray(opt.jac, dtype=np.float64)
        cov_log = np.linalg.pinv(jac.T @ jac)
        cov_log *= _reduced_chi2(float(opt.cost), n_residuals, n_free)
        log_stderr = np.sqrt(np.maximum(np.diag(cov_log), 0.0))
        stderr[free_mask] = fit_values[free_mask] * log_stderr
    except Exception:
        stderr[free_mask] = np.inf

    return _array_to_params(stderr)


def _params_to_array(params: T1FitParams) -> NDArray[np.float64]:
    return np.array(
        [params.Q_cap, params.x_qp, params.Q_ind, params.Temp], dtype=np.float64
    )


def _array_to_params(values: NDArray[np.float64]) -> T1FitParams:
    return T1FitParams(
        Q_cap=float(values[0]),
        x_qp=float(values[1]),
        Q_ind=float(values[2]),
        Temp=float(values[3]),
    )


def _cost(residuals: NDArray[np.float64]) -> float:
    return float(0.5 * np.sum(residuals**2))


def _reduced_chi2(cost: float, n_residuals: int, n_free: int) -> float:
    dof = max(n_residuals - n_free, 1)
    return float(2.0 * cost / dof)


__all__ = ["T1FitParams", "T1FitResult", "fit_t1_noise_params"]

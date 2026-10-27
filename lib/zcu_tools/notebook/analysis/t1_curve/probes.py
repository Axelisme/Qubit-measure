"""Pointwise mechanism probes, intrinsic Q extraction and temperature search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.simulate.fluxonium import (
    calculate_n_oper_vs_flux,
    calculate_phi_oper_vs_flux,
)

from .base import (
    DEFAULT_TEMP_LOWER_BOUND,
    DEFAULT_TEMP_UPPER_BOUND,
    TempBounds,
    find_proper_Temp,
    format_parameter,
    omega_mask,
    positive_override,
    resolve_Temp_bounds,
)
from .models import (
    MECHANISM_TO_PARAM,
    MechanismName,
    ProbeStatistic,
    PurcellCorrectionResult,
    PurcellEffectParams,
    T1MechanismOptions,
    T1MechanismProbe,
    T1PreparedData,
    validate_t1_mechanisms,
)
from .purcell import (
    calculate_purcell_t1_limit,
    subtract_relaxation_limit,
)
from .Qcap import calc_cap_dipole, calc_Qcap_vs_omega
from .Qind import calc_ind_dipole, calc_Qind_vs_omega
from .Qqp import calc_qp_dipole, calc_qp_oper, calc_Qqp_vs_omega


# tolerated: Notebook keywords expose the capacitive window, temperature search and Q estimate.
def analyze_t1_capacitive_limit(  # noqa: PLR0913
    data: T1PreparedData,
    *,
    Temp: float,
    purcell: PurcellEffectParams | None = None,
    omega_range: tuple[float | None, float | None] = (None, None),
    fit_temperature: bool = False,
    Temp_bounds: TempBounds = (DEFAULT_TEMP_LOWER_BOUND, DEFAULT_TEMP_UPPER_BOUND),
    fit_constant: bool = True,
    statistic: ProbeStatistic = "median",
    parameter_init: float | None = None,
) -> T1MechanismProbe:
    """Analyze the capacitive limit using prepared ns T1/error data.

    Temp/Temp_bounds are in K and omega_range is an inclusive rad/ns window.
    purcell optionally supplies GHz resonator inputs for rate subtraction.
    fit_temperature selects a temperature search, fit_constant selects constant
    rather than power-law Q, and statistic selects the fitted-Q summary.
    parameter_init overrides the suggested dimensionless Q_cap.
    Return the full mechanism probe without saving or writeback. Invalid bounds,
    overrides or an empty positive-Q window raise ValueError.
    """
    return analyze_t1_mechanism_limit(
        data,
        mechanism="capacitive",
        options=T1MechanismOptions(
            Temp=Temp,
            purcell=purcell,
            omega_range=omega_range,
            fit_temperature=fit_temperature,
            Temp_bounds=Temp_bounds,
            fit_constant=fit_constant,
            statistic=statistic,
            parameter_init=parameter_init,
        ),
    )


# tolerated: Notebook keywords retain the quasiparticle window and reciprocal-Q estimate controls.
def analyze_t1_quasiparticle_limit(  # noqa: PLR0913
    data: T1PreparedData,
    *,
    Temp: float,
    purcell: PurcellEffectParams | None = None,
    omega_range: tuple[float | None, float | None] = (6.0, None),
    fit_temperature: bool = False,
    Temp_bounds: TempBounds = (DEFAULT_TEMP_LOWER_BOUND, DEFAULT_TEMP_UPPER_BOUND),
    fit_constant: bool = True,
    statistic: ProbeStatistic = "median",
    parameter_init: float | None = None,
) -> T1MechanismProbe:
    """Analyze the quasiparticle limit using prepared ns T1/error data.

    Temp/Temp_bounds are in K and omega_range is an inclusive rad/ns window.
    purcell optionally supplies GHz resonator inputs for rate subtraction.
    fit_temperature selects a temperature search, fit_constant selects constant
    rather than power-law Q, and statistic selects the fitted-Q summary.
    parameter_init overrides the suggested dimensionless x_qp.
    Return the full mechanism probe without saving or writeback. Invalid bounds,
    overrides or an empty positive-Q window raise ValueError.
    """
    return analyze_t1_mechanism_limit(
        data,
        mechanism="quasiparticle",
        options=T1MechanismOptions(
            Temp=Temp,
            purcell=purcell,
            omega_range=omega_range,
            fit_temperature=fit_temperature,
            Temp_bounds=Temp_bounds,
            fit_constant=fit_constant,
            statistic=statistic,
            parameter_init=parameter_init,
        ),
    )


# tolerated: Notebook keywords expose the inductive window, temperature search and Q estimate.
def analyze_t1_inductive_limit(  # noqa: PLR0913
    data: T1PreparedData,
    *,
    Temp: float,
    purcell: PurcellEffectParams | None = None,
    omega_range: tuple[float | None, float | None] = (None, 4.0),
    fit_temperature: bool = False,
    Temp_bounds: TempBounds = (DEFAULT_TEMP_LOWER_BOUND, DEFAULT_TEMP_UPPER_BOUND),
    fit_constant: bool = True,
    statistic: ProbeStatistic = "median",
    parameter_init: float | None = None,
) -> T1MechanismProbe:
    """Analyze the inductive limit using prepared ns T1/error data.

    Temp/Temp_bounds are in K and omega_range is an inclusive rad/ns window.
    purcell optionally supplies GHz resonator inputs for rate subtraction.
    fit_temperature selects a temperature search, fit_constant selects constant
    rather than power-law Q, and statistic selects the fitted-Q summary.
    parameter_init overrides the suggested dimensionless Q_ind.
    Return the full mechanism probe without saving or writeback. Invalid bounds,
    overrides or an empty positive-Q window raise ValueError.
    """
    return analyze_t1_mechanism_limit(
        data,
        mechanism="inductive",
        options=T1MechanismOptions(
            Temp=Temp,
            purcell=purcell,
            omega_range=omega_range,
            fit_temperature=fit_temperature,
            Temp_bounds=Temp_bounds,
            fit_constant=fit_constant,
            statistic=statistic,
            parameter_init=parameter_init,
        ),
    )


def analyze_t1_mechanism_limit(
    data: T1PreparedData,
    *,
    mechanism: MechanismName,
    options: T1MechanismOptions,
) -> T1MechanismProbe:
    """Extract intrinsic Q and suggest a parameter for the selected mechanism.

    Prepared T1/T1err data stay in ns; options select temperature, optional
    Purcell rate subtraction and the rad/ns Q-fit window. Return pointwise
    arrays, fitted Q values, parameter bounds and diagnostic tables without
    saving or writing back. Raise ValueError for an unknown mechanism, an empty
    positive-Q window, invalid temperature-search bounds or an invalid override.
    """
    Temp = options.Temp
    purcell = options.purcell
    omega_range = options.omega_range
    fit_temperature = options.fit_temperature
    Temp_bounds = options.Temp_bounds
    fit_constant = options.fit_constant
    statistic = options.statistic
    parameter_init = options.parameter_init
    validate_t1_mechanisms((mechanism,))
    temp = (
        _fit_mechanism_temperature(
            data,
            mechanism,
            Temp,
            omega_range,
            purcell,
            Temp_bounds=Temp_bounds,
        )
        if fit_temperature
        else Temp
    )
    purcell_correction = (
        _subtract_purcell_from_fit_data(data, purcell, purcell.resolve_Temp(temp))
        if purcell is not None
        else None
    )
    T1_for_q = (
        data.fit.T1_ns
        if purcell_correction is None
        else purcell_correction.intrinsic_T1_ns
    )
    T1err_for_q = (
        data.fit.T1err_ns
        if purcell_correction is None
        else purcell_correction.intrinsic_T1err_ns
    )
    q_values, q_errors, dipoles = _calculate_mechanism_arrays(
        data,
        mechanism,
        temp,
        T1_ns=T1_for_q,
        T1err_ns=T1err_for_q,
    )
    fit_mask = omega_mask(data.fit.omegas, omega_range)
    if purcell_correction is not None:
        fit_mask &= purcell_correction.valid_mask
    fit_mask &= np.isfinite(q_values) & (q_values > 0.0)
    if not np.any(fit_mask):
        raise ValueError(f"No finite positive {mechanism} Q values in omega_range")
    fit_omegas = data.fit.omegas[fit_mask]
    q_for_fit = q_values[fit_mask]
    fit_q_values = _fit_q_values(fit_omegas, q_for_fit, fit_constant=fit_constant)
    q_center = _positive_statistic(fit_q_values, statistic)
    q_lower, q_upper = _log_spread_bounds(q_for_fit)

    parameter_name = MECHANISM_TO_PARAM[mechanism]
    parameter_values = _parameter_values_from_q(mechanism, q_values)
    auto_init = _parameter_from_q(mechanism, q_center)
    param_lower = _parameter_from_q(
        mechanism, q_upper if mechanism == "quasiparticle" else q_lower
    )
    param_upper = _parameter_from_q(
        mechanism, q_lower if mechanism == "quasiparticle" else q_upper
    )
    resolved_init = positive_override(parameter_init, auto_init, parameter_name)
    pointwise_columns: dict[str, object] = {
        "flux": data.fit.fluxs,
        "omega (rad/ns)": data.fit.omegas,
        "T1 (ns)": data.fit.T1_ns,
        "T1err (ns)": data.fit.T1err_ns,
        "T1 used for Q (ns)": T1_for_q,
        "T1err used for Q (ns)": T1err_for_q,
        "dipole": dipoles,
        "Q": q_values,
        parameter_name: parameter_values,
    }
    if purcell_correction is not None:
        pointwise_columns.update(
            {
                "Purcell T1 (ns)": purcell_correction.purcell_T1_ns,
                "Purcell rate (1/ns)": purcell_correction.purcell_rate_per_ns,
                "Purcell subtraction valid": purcell_correction.valid_mask,
            }
        )
    pointwise_table = pd.DataFrame(pointwise_columns)
    purcell_summary_rows = _purcell_probe_summary_rows(data, purcell_correction)
    summary_table = pd.DataFrame(
        [
            ("mechanism", mechanism),
            ("parameter", parameter_name),
            ("Temp", f"{temp * 1e3:.2f} mK"),
            ("Temp bounds", _format_temperature_bounds(Temp_bounds)),
            *purcell_summary_rows,
            ("omega range", _format_range(omega_range)),
            ("fit constant", str(fit_constant)),
            ("fit points", str(int(np.count_nonzero(fit_mask)))),
            ("parameter init", format_parameter(parameter_name, resolved_init)),
            ("parameter lower", format_parameter(parameter_name, param_lower)),
            ("parameter upper", format_parameter(parameter_name, param_upper)),
        ],
        columns=["metric", "value"],
    )
    return T1MechanismProbe(
        data=data,
        mechanism=mechanism,
        parameter_name=parameter_name,
        temperature=temp,
        omega_range=omega_range,
        fit_constant=fit_constant,
        q_values=q_values,
        q_errors=q_errors,
        fit_omegas=fit_omegas,
        fit_q_values=fit_q_values,
        dipoles=dipoles,
        parameter_values=parameter_values,
        parameter_init=resolved_init,
        parameter_lower=param_lower,
        parameter_upper=param_upper,
        pointwise_table=pointwise_table,
        summary_table=summary_table,
        purcell=purcell,
        purcell_correction=purcell_correction,
    )


def _subtract_purcell_from_fit_data(
    data: T1PreparedData,
    purcell: PurcellEffectParams,
    Temp: float,
) -> PurcellCorrectionResult:
    purcell_T1s = calculate_purcell_t1_limit(
        data.calibration.context,
        data.fit.fluxs,
        purcell,
        Temp=Temp,
    )
    return subtract_relaxation_limit(data.fit.T1_ns, data.fit.T1err_ns, purcell_T1s)


def _purcell_probe_summary_rows(
    data: T1PreparedData,
    correction: PurcellCorrectionResult | None,
) -> list[tuple[str, str]]:
    if correction is None:
        return [("Purcell", "off")]
    invalid_count = len(data.fit.fluxs) - int(np.count_nonzero(correction.valid_mask))
    return [
        ("Purcell", "on"),
        ("Purcell invalid rows", str(invalid_count)),
    ]


def _calculate_mechanism_arrays(
    data: T1PreparedData,
    mechanism: MechanismName,
    Temp: float,
    *,
    T1_ns: NDArray[np.float64] | None = None,
    T1err_ns: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    params = data.calibration.context.params
    T1s = data.fit.T1_ns if T1_ns is None else np.asarray(T1_ns, dtype=np.float64)
    T1errs = (
        data.fit.T1err_ns
        if T1err_ns is None
        else np.asarray(T1err_ns, dtype=np.float64)
    )
    if T1s.shape != data.fit.T1_ns.shape:
        raise ValueError("T1_ns must have the same shape as fit T1 data")
    if T1errs.shape != data.fit.T1err_ns.shape:
        raise ValueError("T1err_ns must have the same shape as fit T1err data")
    if mechanism == "capacitive":
        _, n_elements = calculate_n_oper_vs_flux(params, data.fit.fluxs)
        q_values, q_errors = calc_Qcap_vs_omega(
            params, data.fit.omegas, T1s, n_elements, T1errs, Temp
        )
        dipoles = calc_cap_dipole(params, n_elements, data.fit.omegas, Temp)
    elif mechanism == "quasiparticle":
        sin2_elements = np.asarray(
            [calc_qp_oper(params, float(flux)) for flux in data.fit.fluxs],
            dtype=np.complex128,
        )
        q_values, q_errors = calc_Qqp_vs_omega(
            params,
            data.fit.omegas,
            T1s,
            sin2_elements,
            T1errs,
            Temp,
        )
        dipoles = calc_qp_dipole(params, sin2_elements, data.fit.omegas, Temp)
    elif mechanism == "inductive":
        _, phi_elements = calculate_phi_oper_vs_flux(params, data.fit.fluxs)
        q_values, q_errors = calc_Qind_vs_omega(
            params,
            data.fit.omegas,
            T1s,
            phi_elements,
            T1errs,
            Temp,
        )
        dipoles = calc_ind_dipole(params, phi_elements, data.fit.omegas, Temp)
    else:
        raise ValueError(f"unknown T1 mechanism: {mechanism}")
    return (
        np.asarray(q_values, dtype=np.float64),
        np.asarray(q_errors, dtype=np.float64),
        np.asarray(dipoles, dtype=np.float64),
    )


def _fit_mechanism_temperature(
    data: T1PreparedData,
    mechanism: MechanismName,
    Temp: float,
    omega_range: tuple[float | None, float | None],
    purcell: PurcellEffectParams | None,
    *,
    Temp_bounds: TempBounds,
) -> float:
    mask = omega_mask(data.fit.omegas, omega_range)

    def calc_Q_fn(candidate_Temp: float) -> NDArray[np.float64]:
        purcell_correction = (
            _subtract_purcell_from_fit_data(
                data, purcell, purcell.resolve_Temp(candidate_Temp)
            )
            if purcell is not None
            else None
        )
        T1_for_q = (
            data.fit.T1_ns
            if purcell_correction is None
            else purcell_correction.intrinsic_T1_ns
        )
        T1err_for_q = (
            data.fit.T1err_ns
            if purcell_correction is None
            else purcell_correction.intrinsic_T1err_ns
        )
        q_values, _, _ = _calculate_mechanism_arrays(
            data,
            mechanism,
            candidate_Temp,
            T1_ns=T1_for_q,
            T1err_ns=T1err_for_q,
        )
        finite = mask & np.isfinite(q_values) & (q_values > 0.0)
        if purcell_correction is not None:
            finite &= purcell_correction.valid_mask
        return q_values[finite]

    return find_proper_Temp(Temp, calc_Q_fn, Temp_bounds=Temp_bounds)


def _fit_q_values(
    omegas: NDArray[np.float64],
    q_values: NDArray[np.float64],
    *,
    fit_constant: bool,
) -> NDArray[np.float64]:
    if fit_constant:
        return np.full_like(omegas, _geometric_mean(q_values), dtype=np.float64)
    slope, intercept = np.polyfit(np.log(omegas), np.log(q_values), 1)
    return np.asarray(np.exp(intercept) * omegas**slope, dtype=np.float64)


def _geometric_mean(values: NDArray[np.float64]) -> float:
    finite = _finite_positive(values)
    return float(np.exp(np.nanmean(np.log(finite))))


def _log_spread_bounds(values: NDArray[np.float64]) -> tuple[float, float]:
    finite = _finite_positive(values)
    logs = np.log(finite)
    center = float(np.nanmean(logs))
    spread = float(np.nanstd(logs))
    return float(np.exp(center - 2.0 * spread)), float(np.exp(center + 2.0 * spread))


def _parameter_values_from_q(
    mechanism: MechanismName,
    q_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    if mechanism == "quasiparticle":
        return np.divide(
            1.0,
            q_values,
            out=np.full_like(q_values, np.nan, dtype=np.float64),
            where=np.isfinite(q_values) & (q_values > 0.0),
        )
    return q_values


def _parameter_from_q(mechanism: MechanismName, q_value: float) -> float:
    if mechanism == "quasiparticle":
        return float(1.0 / q_value)
    return float(q_value)


def _positive_statistic(
    values: NDArray[np.float64], statistic: ProbeStatistic
) -> float:
    finite = _finite_positive(values)
    if statistic == "median":
        return float(np.nanmedian(finite))
    if statistic == "mean":
        return float(np.nanmean(finite))
    if statistic == "min":
        return float(np.nanmin(finite))
    if statistic == "max":
        return float(np.nanmax(finite))
    if statistic == "p10":
        return float(np.nanpercentile(finite, 10.0))
    if statistic == "p90":
        return float(np.nanpercentile(finite, 90.0))
    raise ValueError(f"unknown probe statistic: {statistic}")


def _finite_positive(values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr) & (arr > 0.0)]
    if finite.size == 0:
        raise ValueError("no positive finite values")
    return finite


def _format_range(value_range: tuple[float | None, float | None]) -> str:
    lower, upper = value_range
    lower_text = "-inf" if lower is None else f"{lower:g}"
    upper_text = "inf" if upper is None else f"{upper:g}"
    return f"{lower_text}..{upper_text}"


def _format_temperature_bounds(Temp_bounds: TempBounds) -> str:
    lower_raw, _upper_raw = Temp_bounds
    lower, upper = resolve_Temp_bounds(Temp_bounds)
    lower_text = "default" if lower_raw is None else f"{lower * 1e3:.2f}"
    return f"{lower_text}..{upper * 1e3:.2f} mK"

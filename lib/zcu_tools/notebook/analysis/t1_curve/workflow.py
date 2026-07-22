from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.meta_tool import (
    QubitParams,
    T1CurveFit,
    T1CurveFitParams,
    T1CurveFitUncertainty,
)
from zcu_tools.notebook.analysis.fit_tools import (
    FluxResidualWeighting,
    MeasurementErrorPolicy,
    choose_current_scale_from_f01,
    correct_flux_from_f01,
)
from zcu_tools.simulate import value2flux
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1_vs_flux_fast,
    calculate_n_oper_vs_flux,
    calculate_phi_oper_vs_flux,
    calculate_purcell_t1_vs_flux,
)

from .base import (
    find_proper_Temp,
    plot_eff_t1_with_sample,
    plot_Q_vs_omega,
    plot_sample_t1,
    plot_t1_vs_elements,
)
from .fit import (
    NoiseParameterName,
    ParameterName,
    ResidualMode,
    T1FitParams,
    T1FitResult,
    fit_t1_noise_params,
)
from .Qcap import calc_cap_dipole, calc_Qcap_vs_omega
from .Qind import calc_ind_dipole, calc_Qind_vs_omega
from .Qqp import calc_qp_dipole, calc_qp_oper, calc_Qqp_vs_omega
from .utils import freq2omega

MechanismName = Literal["capacitive", "quasiparticle", "inductive"]
MechanismOrParamName = Literal[
    "capacitive",
    "quasiparticle",
    "inductive",
    "Q_cap",
    "x_qp",
    "Q_ind",
    "Temp",
]
ProbeStatistic = Literal["median", "mean", "min", "max", "p10", "p90"]

_REQUIRED_COLUMNS = ("calibrated mA", "Freq (MHz)", "T1 (us)")
_MECHANISM_TO_PARAM: dict[MechanismName, NoiseParameterName] = {
    "capacitive": "Q_cap",
    "quasiparticle": "x_qp",
    "inductive": "Q_ind",
}
_PARAM_TO_MECHANISM: dict[NoiseParameterName, MechanismName] = {
    value: key for key, value in _MECHANISM_TO_PARAM.items()
}
_NOISE_CHANNELS: dict[NoiseParameterName, tuple[str, str]] = {
    "Q_cap": ("t1_capacitive", "Q_cap"),
    "x_qp": ("t1_quasiparticle_tunneling", "x_qp"),
    "Q_ind": ("t1_inductive", "Q_ind"),
}
_Q_LABELS: dict[MechanismName, str] = {
    "capacitive": r"$Q_{cap}$",
    "quasiparticle": r"$Q_{qp}=1/x_{qp}$",
    "inductive": r"$Q_{ind}$",
}
_CURVE_LABELS: dict[MechanismName, str] = {
    "capacitive": "capacitive",
    "quasiparticle": "quasiparticle",
    "inductive": "inductive",
}
_PURCELL_CACHE_SIZE = 64
_CACHE_FLOAT_DECIMALS = 12


@dataclass(frozen=True, slots=True)
class PurcellEffectParams:
    kappa_ghz: float
    bare_rf: float
    g: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.kappa_ghz) or self.kappa_ghz <= 0.0:
            raise ValueError("kappa_ghz must be positive and finite")
        if not np.isfinite(self.bare_rf) or self.bare_rf <= 0.0:
            raise ValueError("bare_rf must be positive and finite")
        if not np.isfinite(self.g) or self.g <= 0.0:
            raise ValueError("g must be positive and finite")


@dataclass(frozen=True, slots=True)
class T1CurveAnalysisConfig:
    result_dir: str
    analysis_flux_range: tuple[float, float] = (0.0, 1.0)
    image_dir: str | None = None
    samples_filename: str = "samples.csv"
    current_scale_candidates: tuple[float, ...] = (1.0, 1000.0)
    max_abs_flux_correction: float = 0.03
    max_rel_t1_err: float = 0.25
    use_weighted_points_only: bool = False
    T1_error_policy: MeasurementErrorPolicy = field(
        default_factory=lambda: MeasurementErrorPolicy(
            nan_policy="bin_median",
            relative_floor=0.05,
            fallback_error=1000.0,
        )
    )
    flux_weighting: FluxResidualWeighting = field(
        default_factory=lambda: FluxResidualWeighting(
            mode="equal_flux_bin",
            bin_width=0.01,
        )
    )
    Temp: float = 60e-3
    purcell: PurcellEffectParams | None = None
    active_mechanisms: tuple[MechanismName, ...] = (
        "capacitive",
        "quasiparticle",
        "inductive",
    )
    fixed_mechanisms: tuple[MechanismOrParamName, ...] = ()
    fit_Temp_override: float | None = None
    fit_Q_cap_override: float | None = None
    fit_x_qp_override: float | None = None
    fit_Q_ind_override: float | None = None
    fit_bounds: Mapping[str, tuple[float, float]] | None = None
    residual_mode: ResidualMode = "log"
    loss: str = "linear"
    max_nfev: int = 10000
    t_flux_count: int = 1000
    progress: bool = True
    save_figures: bool = True
    show_figures: bool = True


@dataclass(frozen=True, slots=True)
class T1CurveContext:
    result_dir: str
    image_dir: str
    samples_filename: str
    params: tuple[float, float, float]
    flux_half: float
    flux_int: float
    flux_period: float
    samples_df: pd.DataFrame
    params_table: pd.DataFrame
    samples_preview: pd.DataFrame
    available_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class T1FluxCalibration:
    context: T1CurveContext
    current_scale: float
    scale_report: pd.DataFrame
    t1_df: pd.DataFrame
    freq_rows: pd.DataFrame
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class T1CurveData:
    current_raw: NDArray[np.float64]
    values: NDArray[np.float64]
    raw_fluxs: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    f01_mhz: NDArray[np.float64]
    T1_ns: NDArray[np.float64]
    T1err_ns: NDArray[np.float64]
    omegas: NDArray[np.float64]
    f01_correction_accepted: NDArray[np.bool_]
    flux_corrections: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class T1PreparedData:
    calibration: T1FluxCalibration
    analysis_flux_range: tuple[float, float]
    max_abs_flux_correction: float
    max_rel_t1_err: float
    use_weighted_points_only: bool
    sample: T1CurveData
    fit: T1CurveData
    kept_rows: int
    source_rows: int
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PurcellCorrectionResult:
    observed_T1_ns: NDArray[np.float64]
    observed_T1err_ns: NDArray[np.float64]
    intrinsic_T1_ns: NDArray[np.float64]
    intrinsic_T1err_ns: NDArray[np.float64]
    purcell_T1_ns: NDArray[np.float64]
    purcell_rate_per_ns: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class T1MechanismProbe:
    data: T1PreparedData
    mechanism: MechanismName
    parameter_name: NoiseParameterName
    temperature: float
    omega_range: tuple[float | None, float | None]
    fit_constant: bool
    q_values: NDArray[np.float64]
    q_errors: NDArray[np.float64]
    fit_omegas: NDArray[np.float64]
    fit_q_values: NDArray[np.float64]
    dipoles: NDArray[np.float64]
    parameter_values: NDArray[np.float64]
    parameter_init: float
    parameter_lower: float
    parameter_upper: float
    pointwise_table: pd.DataFrame
    summary_table: pd.DataFrame
    purcell: PurcellEffectParams | None = None
    purcell_correction: PurcellCorrectionResult | None = None


@dataclass(frozen=True, slots=True)
class T1CombinedFit:
    data: T1PreparedData
    init: T1FitParams
    bounds: Mapping[str, tuple[float, float]] | None
    fixed: tuple[ParameterName, ...]
    residual_mode: ResidualMode
    loss: str
    max_nfev: int | None
    fit_result: T1FitResult
    params_table: pd.DataFrame
    summary_table: pd.DataFrame
    purcell: PurcellEffectParams | None = None


@dataclass(frozen=True, slots=True)
class T1ChannelCurves:
    fluxs: NDArray[np.float64]
    T1_effective_ns: NDArray[np.float64]
    component_T1s_ns: Mapping[str, NDArray[np.float64]]
    purcell_T1_ns: NDArray[np.float64] | None
    display_T1_ns: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class T1ChannelAnalysis:
    combined_fit: T1CombinedFit
    curves: T1ChannelCurves
    flux_range: tuple[float, float]
    t_flux_count: int
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class T1CurveAnalysisResult:
    context: T1CurveContext
    calibration: T1FluxCalibration
    data: T1PreparedData
    cap_probe: T1MechanismProbe | None
    qp_probe: T1MechanismProbe | None
    ind_probe: T1MechanismProbe | None
    combined_fit: T1CombinedFit
    channel_analysis: T1ChannelAnalysis
    summary_table: pd.DataFrame
    figure_paths: Mapping[str, str]


def load_t1_curve_context(
    *,
    result_dir: str,
    samples_filename: str = "samples.csv",
    image_dir: str | None = None,
    preview_rows: int = 10,
) -> T1CurveContext:
    resolved_image_dir = image_dir or os.path.join(result_dir, "t1_curve")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(resolved_image_dir, exist_ok=True)

    params_file = QubitParams(os.path.join(result_dir, "params.json"), readonly=True)
    fit_inputs = params_file.require_fluxdep_fit()
    samples_df = pd.read_csv(os.path.join(result_dir, samples_filename))
    _validate_required_columns(samples_df)

    params = fit_inputs.params
    params_table = pd.DataFrame(
        [
            ("EJ (GHz)", params[0]),
            ("EC (GHz)", params[1]),
            ("EL (GHz)", params[2]),
            ("flux_half", fit_inputs.flux_half),
            ("flux_int", fit_inputs.flux_int),
            ("flux_period", fit_inputs.flux_period),
        ],
        columns=["parameter", "value"],
    )

    return T1CurveContext(
        result_dir=result_dir,
        image_dir=resolved_image_dir,
        samples_filename=samples_filename,
        params=params,
        flux_half=fit_inputs.flux_half,
        flux_int=fit_inputs.flux_int,
        flux_period=fit_inputs.flux_period,
        samples_df=samples_df,
        params_table=params_table,
        samples_preview=samples_df.head(preview_rows),
        available_columns=tuple(str(column) for column in samples_df.columns),
    )


def calibrate_t1_flux(
    context: T1CurveContext,
    *,
    current_scale_candidates: tuple[float, ...] = (1.0, 1000.0),
) -> T1FluxCalibration:
    samples_df = context.samples_df
    finite_t1 = np.isfinite(_float_column(samples_df, "T1 (us)"))
    finite_freq = np.isfinite(_float_column(samples_df, "calibrated mA")) & np.isfinite(
        _float_column(samples_df, "Freq (MHz)")
    )
    t1_df = cast(pd.DataFrame, samples_df.loc[finite_t1 & finite_freq].copy())
    freq_rows = cast(pd.DataFrame, samples_df.loc[finite_freq].copy())
    if freq_rows.empty:
        raise ValueError("samples.csv has no finite f01 rows for flux calibration")
    current_scale, scale_report = choose_current_scale_from_f01(
        _float_column(freq_rows, "calibrated mA"),
        _float_column(freq_rows, "Freq (MHz)"),
        params=context.params,
        flux_half=context.flux_half,
        flux_period=context.flux_period,
        candidates=current_scale_candidates,
    )
    summary_table = pd.DataFrame(
        [
            ("samples.csv rows", str(len(samples_df))),
            ("finite f01 rows", str(len(freq_rows))),
            ("finite T1 rows", str(len(t1_df))),
            ("current scale", f"{current_scale:g}"),
        ],
        columns=["metric", "value"],
    )
    return T1FluxCalibration(
        context=context,
        current_scale=current_scale,
        scale_report=scale_report,
        t1_df=t1_df,
        freq_rows=freq_rows,
        summary_table=summary_table,
    )


def prepare_t1_curve_data(
    calibration: T1FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float] = (0.0, 1.0),
    max_abs_flux_correction: float = 0.03,
    max_rel_t1_err: float = 0.25,
    use_weighted_points_only: bool = False,
    correct_flux_from_f01_enabled: bool = True,
) -> T1PreparedData:
    context = calibration.context
    sample = _prepare_t1_data(
        calibration.t1_df,
        params=context.params,
        flux_half=context.flux_half,
        flux_period=context.flux_period,
        current_scale=calibration.current_scale,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    fit = _filter_fit_data(
        sample,
        max_rel_t1_err=max_rel_t1_err,
        use_weighted_points_only=use_weighted_points_only,
    )
    if len(fit.fluxs) == 0:
        raise ValueError("No T1 fit rows remain after filtering")

    summary_table = _prepared_summary_table(
        calibration=calibration,
        sample=sample,
        fit=fit,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        max_rel_t1_err=max_rel_t1_err,
        use_weighted_points_only=use_weighted_points_only,
    )
    return T1PreparedData(
        calibration=calibration,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        max_rel_t1_err=max_rel_t1_err,
        use_weighted_points_only=use_weighted_points_only,
        sample=sample,
        fit=fit,
        kept_rows=len(sample.fluxs),
        source_rows=len(calibration.t1_df),
        summary_table=summary_table,
    )


def calculate_purcell_t1_limit(
    context: T1CurveContext,
    fluxs: NDArray[np.float64],
    purcell: PurcellEffectParams,
    *,
    Temp: float,
) -> NDArray[np.float64]:
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
    _calculate_purcell_t1_limit_cached.cache_clear()


def subtract_relaxation_limit(
    observed_T1_ns: NDArray[np.float64],
    observed_T1err_ns: NDArray[np.float64] | None,
    limit_T1_ns: NDArray[np.float64],
) -> PurcellCorrectionResult:
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

    observed_rate = _relaxation_rates_from_t1(observed_T1s, name="observed T1")
    limit_rate = _relaxation_rates_from_t1(limit_T1s, name="limit T1")
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


def analyze_t1_capacitive_limit(
    data: T1PreparedData,
    *,
    Temp: float,
    purcell: PurcellEffectParams | None = None,
    omega_range: tuple[float | None, float | None] = (None, None),
    fit_temperature: bool = False,
    fit_constant: bool = True,
    statistic: ProbeStatistic = "median",
    parameter_init: float | None = None,
) -> T1MechanismProbe:
    return analyze_t1_mechanism_limit(
        data,
        mechanism="capacitive",
        Temp=Temp,
        purcell=purcell,
        omega_range=omega_range,
        fit_temperature=fit_temperature,
        fit_constant=fit_constant,
        statistic=statistic,
        parameter_init=parameter_init,
    )


def analyze_t1_quasiparticle_limit(
    data: T1PreparedData,
    *,
    Temp: float,
    purcell: PurcellEffectParams | None = None,
    omega_range: tuple[float | None, float | None] = (6.0, None),
    fit_temperature: bool = False,
    fit_constant: bool = True,
    statistic: ProbeStatistic = "median",
    parameter_init: float | None = None,
) -> T1MechanismProbe:
    return analyze_t1_mechanism_limit(
        data,
        mechanism="quasiparticle",
        Temp=Temp,
        purcell=purcell,
        omega_range=omega_range,
        fit_temperature=fit_temperature,
        fit_constant=fit_constant,
        statistic=statistic,
        parameter_init=parameter_init,
    )


def analyze_t1_inductive_limit(
    data: T1PreparedData,
    *,
    Temp: float,
    purcell: PurcellEffectParams | None = None,
    omega_range: tuple[float | None, float | None] = (None, 4.0),
    fit_temperature: bool = False,
    fit_constant: bool = True,
    statistic: ProbeStatistic = "median",
    parameter_init: float | None = None,
) -> T1MechanismProbe:
    return analyze_t1_mechanism_limit(
        data,
        mechanism="inductive",
        Temp=Temp,
        purcell=purcell,
        omega_range=omega_range,
        fit_temperature=fit_temperature,
        fit_constant=fit_constant,
        statistic=statistic,
        parameter_init=parameter_init,
    )


def analyze_t1_mechanism_limit(
    data: T1PreparedData,
    *,
    mechanism: MechanismName,
    Temp: float,
    purcell: PurcellEffectParams | None = None,
    omega_range: tuple[float | None, float | None] = (None, None),
    fit_temperature: bool = False,
    fit_constant: bool = True,
    statistic: ProbeStatistic = "median",
    parameter_init: float | None = None,
) -> T1MechanismProbe:
    _validate_mechanisms((mechanism,))
    temp = (
        _fit_mechanism_temperature(data, mechanism, Temp, omega_range, purcell)
        if fit_temperature
        else Temp
    )
    purcell_correction = (
        _subtract_purcell_from_fit_data(data, purcell, temp)
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
    fit_mask = _omega_mask(data.fit.omegas, omega_range)
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

    parameter_name = _MECHANISM_TO_PARAM[mechanism]
    parameter_values = _parameter_values_from_q(mechanism, q_values)
    auto_init = _parameter_from_q(mechanism, q_center)
    param_lower = _parameter_from_q(
        mechanism, q_upper if mechanism == "quasiparticle" else q_lower
    )
    param_upper = _parameter_from_q(
        mechanism, q_lower if mechanism == "quasiparticle" else q_upper
    )
    resolved_init = _positive_override(parameter_init, auto_init, parameter_name)
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
            *purcell_summary_rows,
            ("omega range", _format_range(omega_range)),
            ("fit constant", str(fit_constant)),
            ("fit points", str(int(np.count_nonzero(fit_mask)))),
            ("parameter init", _format_parameter(parameter_name, resolved_init)),
            ("parameter lower", _format_parameter(parameter_name, param_lower)),
            ("parameter upper", _format_parameter(parameter_name, param_upper)),
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


def make_t1_fit_init(
    *,
    active_mechanisms: tuple[MechanismName, ...],
    Temp: float,
    cap_probe: T1MechanismProbe | None = None,
    qp_probe: T1MechanismProbe | None = None,
    ind_probe: T1MechanismProbe | None = None,
    Q_cap: float | None = None,
    x_qp: float | None = None,
    Q_ind: float | None = None,
) -> T1FitParams:
    _validate_mechanisms(active_mechanisms)
    probe_by_param = {
        "Q_cap": cap_probe,
        "x_qp": qp_probe,
        "Q_ind": ind_probe,
    }
    return T1FitParams(
        Temp=_positive_override(None, Temp, "Temp"),
        Q_cap=(
            _fit_init_value(Q_cap, probe_by_param["Q_cap"], "Q_cap")
            if "capacitive" in active_mechanisms
            else None
        ),
        x_qp=(
            _fit_init_value(x_qp, probe_by_param["x_qp"], "x_qp")
            if "quasiparticle" in active_mechanisms
            else None
        ),
        Q_ind=(
            _fit_init_value(Q_ind, probe_by_param["Q_ind"], "Q_ind")
            if "inductive" in active_mechanisms
            else None
        ),
    )


def mechanisms_to_fixed_params(
    fixed_mechanisms: tuple[MechanismOrParamName, ...],
) -> tuple[ParameterName, ...]:
    fixed: list[ParameterName] = []
    for name in fixed_mechanisms:
        if name in ("capacitive", "Q_cap"):
            fixed.append("Q_cap")
        elif name in ("quasiparticle", "x_qp"):
            fixed.append("x_qp")
        elif name in ("inductive", "Q_ind"):
            fixed.append("Q_ind")
        elif name == "Temp":
            fixed.append("Temp")
        else:
            raise ValueError(f"unknown T1 mechanism or parameter: {name}")
    if len(set(fixed)) != len(fixed):
        raise ValueError("fixed_mechanisms contains duplicate parameters")
    return tuple(fixed)


def make_t1_fit_bounds(
    init: T1FitParams,
    *,
    factor: float = 100.0,
    Temp_bounds: tuple[float, float] = (10e-3, 300e-3),
    Q_lower_floor: float = 1.0,
    x_qp_lower_floor: float = 1e-12,
    Q_upper_cap: float = np.inf,
    x_qp_upper_cap: float = 1.0,
) -> dict[str, tuple[float, float]]:
    if factor <= 1.0:
        raise ValueError("factor must be larger than 1")
    bounds: dict[str, tuple[float, float]] = {"Temp": Temp_bounds}
    if init.Q_cap is not None:
        bounds["Q_cap"] = (
            max(Q_lower_floor, init.Q_cap / factor),
            min(Q_upper_cap, init.Q_cap * factor),
        )
    if init.x_qp is not None:
        bounds["x_qp"] = (
            max(x_qp_lower_floor, init.x_qp / factor),
            min(x_qp_upper_cap, init.x_qp * factor),
        )
    if init.Q_ind is not None:
        bounds["Q_ind"] = (
            max(Q_lower_floor, init.Q_ind / factor),
            min(Q_upper_cap, init.Q_ind * factor),
        )
    return bounds


def fit_t1_curve(
    data: T1PreparedData,
    *,
    init: T1FitParams,
    purcell: PurcellEffectParams | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    fixed: tuple[ParameterName, ...] = (),
    T1_error_policy: MeasurementErrorPolicy | None = None,
    flux_weighting: FluxResidualWeighting | None = None,
    residual_mode: ResidualMode = "log",
    loss: str = "linear",
    max_nfev: int | None = 10000,
    progress: bool = False,
) -> T1CombinedFit:
    context = data.calibration.context
    extra_relaxation_rate_fn = (
        _purcell_rate_fn(data, purcell) if purcell is not None else None
    )
    fit_result = fit_t1_noise_params(
        data.fit.fluxs,
        data.fit.T1_ns,
        context.params,
        init=init,
        bounds=bounds,
        fixed=fixed,
        T1errs=data.fit.T1err_ns,
        T1_error_policy=T1_error_policy,
        flux_weighting=flux_weighting,
        residual_mode=residual_mode,
        loss=loss,
        max_nfev=max_nfev,
        extra_relaxation_rate_fn=extra_relaxation_rate_fn,
        progress=progress,
    )
    params_table = _fit_params_table(init, fit_result)
    purcell_rows = _purcell_fit_summary_rows(purcell)
    summary_table = pd.DataFrame(
        [
            ("fit success", str(fit_result.success)),
            ("fit message", fit_result.message),
            *purcell_rows,
            ("fixed", str(fit_result.fixed)),
            ("free", str(fit_result.free)),
            ("fit rows", str(len(data.fit.fluxs))),
            (
                "effective flux bins",
                f"{fit_result.flux_weights.effective_observation_count:g}",
            ),
            ("flux weighting", fit_result.flux_weights.mode),
            ("reduced chi2", f"{fit_result.reduced_chi2:.3g}"),
            ("T1err NaN filled", _error_fill_summary(fit_result.T1_error_resolution)),
        ],
        columns=["metric", "value"],
    )
    return T1CombinedFit(
        data=data,
        init=init,
        bounds=bounds,
        fixed=fixed,
        residual_mode=residual_mode,
        loss=loss,
        max_nfev=max_nfev,
        fit_result=fit_result,
        params_table=params_table,
        summary_table=summary_table,
        purcell=purcell,
    )


def build_t1_channel_curves(
    combined_fit: T1CombinedFit,
    *,
    t_flux_count: int = 1000,
    flux_range: tuple[float, float] | None = None,
    purcell: PurcellEffectParams | None = None,
    include_purcell: bool | None = None,
) -> T1ChannelAnalysis:
    data = combined_fit.data
    context = data.calibration.context
    resolved_purcell = _resolve_channel_purcell(
        combined_fit,
        purcell=purcell,
        include_purcell=include_purcell,
    )
    resolved_flux_range = flux_range or data.analysis_flux_range
    grid_fluxs = np.linspace(
        resolved_flux_range[0], resolved_flux_range[1], t_flux_count, dtype=np.float64
    )
    fit_params = combined_fit.fit_result.params
    noise_channels = t1_noise_channels_from_params(fit_params)
    component_t1s = {
        _CURVE_LABELS[_PARAM_TO_MECHANISM[param_name]]: calculate_eff_t1_vs_flux_fast(
            context.params,
            grid_fluxs,
            [(channel_name, dict(options))],
            fit_params.Temp,
        )
        for param_name, channel_name, options in _iter_noise_channels(fit_params)
    }
    mechanism_T1s = calculate_eff_t1_vs_flux_fast(
        context.params,
        grid_fluxs,
        noise_channels,
        fit_params.Temp,
    )
    purcell_T1s = None
    display_T1s = mechanism_T1s
    if resolved_purcell is not None:
        purcell_T1s = calculate_purcell_t1_limit(
            context,
            grid_fluxs,
            resolved_purcell,
            Temp=fit_params.Temp,
        )
        component_t1s = {**component_t1s, "Purcell": purcell_T1s}
        display_T1s = _combine_t1_limits(mechanism_T1s, purcell_T1s)

    curves = T1ChannelCurves(
        fluxs=grid_fluxs,
        T1_effective_ns=mechanism_T1s,
        component_T1s_ns=component_t1s,
        purcell_T1_ns=purcell_T1s,
        display_T1_ns=display_T1s,
    )
    summary_table = pd.DataFrame(
        [
            (
                "flux range",
                f"{resolved_flux_range[0]:.3f}..{resolved_flux_range[1]:.3f}",
            ),
            ("t_flux_count", str(t_flux_count)),
            ("include Purcell", str(resolved_purcell is not None)),
            ("active fit params", str(combined_fit.fit_result.free)),
            ("fixed fit params", str(combined_fit.fit_result.fixed)),
        ],
        columns=["metric", "value"],
    )
    return T1ChannelAnalysis(
        combined_fit=combined_fit,
        curves=curves,
        flux_range=resolved_flux_range,
        t_flux_count=t_flux_count,
        summary_table=summary_table,
    )


def collect_t1_curve_result(
    *,
    context: T1CurveContext,
    calibration: T1FluxCalibration,
    data: T1PreparedData,
    cap_probe: T1MechanismProbe | None,
    qp_probe: T1MechanismProbe | None,
    ind_probe: T1MechanismProbe | None,
    combined_fit: T1CombinedFit,
    channel_analysis: T1ChannelAnalysis,
    figure_paths: Mapping[str, str] | None = None,
) -> T1CurveAnalysisResult:
    probe_tables: list[pd.DataFrame] = []
    if cap_probe is not None:
        probe_tables.append(_stage_table("Q_cap_probe", cap_probe.summary_table))
    if qp_probe is not None:
        probe_tables.append(_stage_table("x_qp_probe", qp_probe.summary_table))
    if ind_probe is not None:
        probe_tables.append(_stage_table("Q_ind_probe", ind_probe.summary_table))
    summary_table = pd.concat(
        [
            _stage_table("load", calibration.summary_table),
            _stage_table("prepare", data.summary_table),
            *probe_tables,
            _stage_table("combined_fit", combined_fit.summary_table),
            _stage_table("channel_curves", channel_analysis.summary_table),
        ],
        ignore_index=True,
    )
    return T1CurveAnalysisResult(
        context=context,
        calibration=calibration,
        data=data,
        cap_probe=cap_probe,
        qp_probe=qp_probe,
        ind_probe=ind_probe,
        combined_fit=combined_fit,
        channel_analysis=channel_analysis,
        summary_table=summary_table,
        figure_paths={} if figure_paths is None else dict(figure_paths),
    )


def plot_t1_curve_data(data: T1PreparedData) -> tuple[Figure, Axes]:
    context = data.calibration.context
    return plot_sample_t1(
        data.sample.values,
        data.sample.T1_ns,
        data.sample.T1err_ns,
        context.flux_half,
        context.flux_period,
    )


def plot_t1_flux_calibration(data: T1PreparedData) -> tuple[Figure, Axes]:
    sample = data.sample
    if len(sample.fluxs) == 0:
        raise ValueError("No T1 samples are available for flux calibration plotting")

    raw_fluxs = sample.raw_fluxs
    corrected_fluxs = sample.fluxs
    accepted = sample.f01_correction_accepted
    f01_mhz = sample.f01_mhz
    fig_height = float(np.clip(1.8 + 0.18 * len(corrected_fluxs), 3.0, 8.0))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    lower, upper = data.analysis_flux_range
    ax.axvspan(lower, upper, color="tab:green", alpha=0.08, label="analysis window")

    for f01_freq_mhz, raw_flux, corrected_flux, is_accepted in zip(
        f01_mhz,
        raw_fluxs,
        corrected_fluxs,
        accepted,
        strict=True,
    ):
        line_color = "tab:blue" if is_accepted else "tab:gray"
        ax.plot(
            [raw_flux, corrected_flux],
            [f01_freq_mhz, f01_freq_mhz],
            color=line_color,
            alpha=0.55,
            linewidth=1.0,
        )

    ax.scatter(
        raw_fluxs,
        f01_mhz,
        marker="o",
        facecolors="none",
        edgecolors="tab:gray",
        s=34,
        label="raw flux",
        zorder=3,
    )
    if np.any(accepted):
        ax.scatter(
            corrected_fluxs[accepted],
            f01_mhz[accepted],
            marker="o",
            color="tab:blue",
            s=20,
            label="f01 corrected flux",
            zorder=4,
        )
    rejected = ~accepted
    if np.any(rejected):
        ax.scatter(
            corrected_fluxs[rejected],
            f01_mhz[rejected],
            marker="x",
            color="tab:orange",
            s=38,
            label="kept raw flux",
            zorder=4,
        )

    ax.set_xlabel(r"Flux quanta ($\Phi_{ext}/\Phi_0$)")
    ax.set_ylabel("f01 frequency (MHz)")
    ax.set_title("f01 flux correction")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_t1_mechanism_probe(probe: T1MechanismProbe) -> tuple[Figure, Axes]:
    fig, ax = plot_Q_vs_omega(
        probe.data.fit.omegas,
        probe.q_values,
        probe.q_errors,
        Qname=_Q_LABELS[probe.mechanism],
    )
    ax.plot(
        probe.fit_omegas,
        probe.fit_q_values,
        label=f"{_CURVE_LABELS[probe.mechanism]} probe",
    )
    ax.set_title(f"Temp = {probe.temperature * 1e3:.2f} mK")
    ax.legend(fontsize="small")
    return fig, ax


def plot_t1_mechanism_dipole(probe: T1MechanismProbe) -> tuple[Figure, Axes]:
    T1_for_q = (
        probe.data.fit.T1_ns
        if probe.purcell_correction is None
        else probe.purcell_correction.intrinsic_T1_ns
    )
    T1err_for_q = (
        probe.data.fit.T1err_ns
        if probe.purcell_correction is None
        else probe.purcell_correction.intrinsic_T1err_ns
    )
    valid = _omega_mask(probe.data.fit.omegas, probe.omega_range)
    if probe.purcell_correction is not None:
        valid &= probe.purcell_correction.valid_mask
    valid &= (
        np.isfinite(probe.dipoles)
        & (probe.dipoles > 0.0)
        & np.isfinite(T1_for_q)
        & (T1_for_q > 0.0)
        & np.isfinite(probe.q_values)
        & (probe.q_values > 0.0)
    )
    if not np.any(valid):
        raise ValueError(f"No finite positive {probe.mechanism} dipole points to plot")

    Q_name = (
        r"$x_{qp}$"
        if probe.mechanism == "quasiparticle"
        else _Q_LABELS[probe.mechanism]
    )
    product2val: Callable[[float], float] = (
        (lambda product: 1.0 / product)
        if probe.mechanism == "quasiparticle"
        else (lambda product: product)
    )
    fig, ax = plot_t1_vs_elements(
        probe.dipoles[valid],
        T1_for_q[valid],
        T1err_for_q[valid],
        Q_name=Q_name,
        product2val=product2val,
    )
    title_suffix = " after Purcell subtraction" if probe.purcell_correction else ""
    ax.set_title(f"Temp = {probe.temperature * 1e3:.2f} mK{title_suffix}")
    return fig, ax


def plot_t1_mechanism_limit(
    probe: T1MechanismProbe,
    *,
    t_flux_count: int = 1000,
    flux_range: tuple[float, float] | None = None,
    purcell: PurcellEffectParams | None = None,
) -> tuple[Figure, Axes]:
    data = probe.data
    context = data.calibration.context
    resolved_purcell = probe.purcell if purcell is None else purcell
    resolved_flux_range = flux_range or data.analysis_flux_range
    t_fluxs = np.linspace(
        resolved_flux_range[0], resolved_flux_range[1], t_flux_count, dtype=np.float64
    )
    channel_name, option_name = _NOISE_CHANNELS[probe.parameter_name]
    values = {
        "lower": probe.parameter_lower,
        "probe": probe.parameter_init,
        "upper": probe.parameter_upper,
    }
    mechanism_curves = {
        f"{_CURVE_LABELS[probe.mechanism]} {label}": calculate_eff_t1_vs_flux_fast(
            context.params,
            t_fluxs,
            [(channel_name, {option_name: value})],
            probe.temperature,
        )
        for label, value in values.items()
    }
    probe_curve_name = f"{_CURVE_LABELS[probe.mechanism]} probe"
    display_T1s = mechanism_curves[probe_curve_name]
    component_t1s = {
        key: value
        for key, value in mechanism_curves.items()
        if not key.endswith(" probe")
    }
    plot_label = _CURVE_LABELS[probe.mechanism]
    parameter_lines = [
        _format_parameter(probe.parameter_name, probe.parameter_init),
        f"lower = {_format_parameter_value(probe.parameter_name, probe.parameter_lower)}",
        f"upper = {_format_parameter_value(probe.parameter_name, probe.parameter_upper)}",
    ]
    if resolved_purcell is not None:
        purcell_T1s = calculate_purcell_t1_limit(
            context,
            t_fluxs,
            resolved_purcell,
            Temp=probe.temperature,
        )
        combined_curves = {
            key: _combine_t1_limits(value, purcell_T1s)
            for key, value in mechanism_curves.items()
        }
        display_T1s = combined_curves[probe_curve_name]
        component_t1s = {
            key: value
            for key, value in combined_curves.items()
            if not key.endswith(" probe")
        }
        component_t1s = {**component_t1s, "Purcell": purcell_T1s}
        plot_label = f"{plot_label} + Purcell"
        parameter_lines.extend(_purcell_parameter_text_lines(resolved_purcell))
    fig, ax = plot_eff_t1_with_sample(
        data.sample.values,
        data.sample.T1_ns,
        data.sample.T1err_ns,
        display_T1s,
        context.flux_half,
        context.flux_period,
        t_fluxs,
        label=plot_label,
        title=f"Temp = {probe.temperature * 1e3:.2f} mK",
        component_t1s=component_t1s,
        parameter_text="\n".join(parameter_lines),
    )
    ax.set_xlim(*resolved_flux_range)
    return fig, ax


def plot_t1_channel_analysis(
    channel_analysis: T1ChannelAnalysis,
    *,
    parameter_text: str | None = None,
) -> tuple[Figure, Axes]:
    combined_fit = channel_analysis.combined_fit
    data = combined_fit.data
    context = data.calibration.context
    if parameter_text is None:
        parameter_text = t1_parameter_text(
            combined_fit.fit_result,
            extra_lines=_purcell_parameter_text_lines(combined_fit.purcell),
        )
    fig, ax = plot_eff_t1_with_sample(
        data.sample.values,
        data.sample.T1_ns,
        data.sample.T1err_ns,
        channel_analysis.curves.display_T1_ns,
        context.flux_half,
        context.flux_period,
        channel_analysis.curves.fluxs,
        label="effective T1",
        title=f"Temperature = {combined_fit.fit_result.params.Temp * 1e3:.2f} mK",
        component_t1s=channel_analysis.curves.component_T1s_ns,
        parameter_text=parameter_text,
    )
    ax.set_xlim(*channel_analysis.flux_range)
    return fig, ax


def save_t1_curve_figure(
    fig: Figure,
    context: T1CurveContext,
    filename: str,
    *,
    show: bool = True,
    close: bool = True,
    bbox_inches: str | None = None,
    dpi: int = 160,
) -> str:
    path = os.path.join(context.image_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return path


def write_t1_curve_fit(combined_fit: T1CombinedFit) -> None:
    result = combined_fit.fit_result
    params = result.params
    stderr = result.stderr
    QubitParams(
        os.path.join(combined_fit.data.calibration.context.result_dir, "params.json")
    ).set_t1_curve_fit(
        T1CurveFit(
            params=T1CurveFitParams(
                Temp=params.Temp,
                Q_cap=params.Q_cap,
                x_qp=params.x_qp,
                Q_ind=params.Q_ind,
            ),
            stderr=T1CurveFitUncertainty(
                Temp=stderr.Temp,
                Q_cap=stderr.Q_cap,
                x_qp=stderr.x_qp,
                Q_ind=stderr.Q_ind,
            ),
            fixed=result.fixed,
            free=result.free,
            cost=result.cost,
            reduced_chi2=result.reduced_chi2,
            success=result.success,
            message=result.message,
            residual_mode=combined_fit.residual_mode,
            loss=combined_fit.loss,
            max_nfev=combined_fit.max_nfev,
            init=T1CurveFitParams(
                Temp=combined_fit.init.Temp,
                Q_cap=combined_fit.init.Q_cap,
                x_qp=combined_fit.init.x_qp,
                Q_ind=combined_fit.init.Q_ind,
            ),
            bounds=combined_fit.bounds or {},
        )
    )


def t1_parameter_text(
    fit_result: T1FitResult,
    *,
    extra_lines: tuple[str, ...] = (),
) -> str:
    lines: list[str] = []
    params = fit_result.params
    if params.Q_cap is not None:
        lines.append(_format_parameter("Q_cap", params.Q_cap))
    if params.x_qp is not None:
        lines.append(_format_parameter("x_qp", params.x_qp))
    if params.Q_ind is not None:
        lines.append(_format_parameter("Q_ind", params.Q_ind))
    lines.append(f"Temp = {params.Temp * 1e3:.2f} mK")
    lines.append(f"reduced chi2 = {fit_result.reduced_chi2:.3g}")
    lines.extend(extra_lines)
    return "\n".join(lines)


def _purcell_parameter_text_lines(
    purcell: PurcellEffectParams | None,
) -> tuple[str, ...]:
    if purcell is None:
        return ()
    return (
        f"Purcell kappa = {purcell.kappa_ghz:.3e} GHz",
        f"Purcell bare_rf = {purcell.bare_rf:.6g} GHz",
        f"Purcell g = {purcell.g:.3e} GHz",
    )


def run_t1_curve_analysis(
    config: T1CurveAnalysisConfig,
) -> T1CurveAnalysisResult:
    context = load_t1_curve_context(
        result_dir=config.result_dir,
        samples_filename=config.samples_filename,
        image_dir=config.image_dir,
    )
    calibration = calibrate_t1_flux(
        context,
        current_scale_candidates=config.current_scale_candidates,
    )
    data = prepare_t1_curve_data(
        calibration,
        analysis_flux_range=config.analysis_flux_range,
        max_abs_flux_correction=config.max_abs_flux_correction,
        max_rel_t1_err=config.max_rel_t1_err,
        use_weighted_points_only=config.use_weighted_points_only,
    )
    cap_probe = (
        analyze_t1_capacitive_limit(data, Temp=config.Temp, purcell=config.purcell)
        if "capacitive" in config.active_mechanisms
        else None
    )
    qp_probe = (
        analyze_t1_quasiparticle_limit(data, Temp=config.Temp, purcell=config.purcell)
        if "quasiparticle" in config.active_mechanisms
        else None
    )
    ind_probe = (
        analyze_t1_inductive_limit(data, Temp=config.Temp, purcell=config.purcell)
        if "inductive" in config.active_mechanisms
        else None
    )
    fit_init = make_t1_fit_init(
        active_mechanisms=config.active_mechanisms,
        Temp=config.fit_Temp_override or config.Temp,
        cap_probe=cap_probe,
        qp_probe=qp_probe,
        ind_probe=ind_probe,
        Q_cap=config.fit_Q_cap_override,
        x_qp=config.fit_x_qp_override,
        Q_ind=config.fit_Q_ind_override,
    )
    combined_fit = fit_t1_curve(
        data,
        init=fit_init,
        bounds=config.fit_bounds or make_t1_fit_bounds(fit_init),
        fixed=mechanisms_to_fixed_params(config.fixed_mechanisms),
        T1_error_policy=config.T1_error_policy,
        flux_weighting=config.flux_weighting,
        residual_mode=config.residual_mode,
        loss=config.loss,
        max_nfev=config.max_nfev,
        purcell=config.purcell,
        progress=config.progress,
    )
    channel_analysis = build_t1_channel_curves(
        combined_fit,
        t_flux_count=config.t_flux_count,
        flux_range=config.analysis_flux_range,
        purcell=config.purcell,
    )
    figure_paths: dict[str, str] = {}
    if config.save_figures:
        for name, fig_name, figure in [
            (
                "flux_calibration",
                "flux_calibration.png",
                plot_t1_flux_calibration(data)[0],
            ),
            ("samples", "T1s.png", plot_t1_curve_data(data)[0]),
            (
                "channel_overlay",
                "T1s_fit_eff.png",
                plot_t1_channel_analysis(channel_analysis)[0],
            ),
        ]:
            figure_paths[name] = save_t1_curve_figure(
                figure,
                context,
                fig_name,
                show=config.show_figures,
                bbox_inches="tight" if name == "channel_overlay" else None,
            )
    return collect_t1_curve_result(
        context=context,
        calibration=calibration,
        data=data,
        cap_probe=cap_probe,
        qp_probe=qp_probe,
        ind_probe=ind_probe,
        combined_fit=combined_fit,
        channel_analysis=channel_analysis,
        figure_paths=figure_paths,
    )


def _validate_required_columns(samples_df: pd.DataFrame) -> None:
    missing = [
        column for column in _REQUIRED_COLUMNS if column not in samples_df.columns
    ]
    if missing:
        raise ValueError(f"samples.csv missing required columns: {missing}")


def _float_column(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float64)
    values = pd.to_numeric(cast(pd.Series, frame[column]), errors="coerce")
    return np.asarray(values, dtype=np.float64)


def _prepare_t1_data(
    frame: pd.DataFrame,
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    current_scale: float,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    correct_flux_from_f01_enabled: bool,
) -> T1CurveData:
    current_raw = _float_column(frame, "calibrated mA")
    values = current_raw * current_scale
    f01_mhz = _float_column(frame, "Freq (MHz)")
    T1_ns = 1e3 * _float_column(frame, "T1 (us)")
    T1err_ns = 1e3 * _float_column(frame, "T1err (us)")
    finite = (
        np.isfinite(values) & np.isfinite(f01_mhz) & np.isfinite(T1_ns) & (T1_ns > 0.0)
    )
    values = values[finite]
    current_raw = current_raw[finite]
    f01_mhz = f01_mhz[finite]
    T1_ns = T1_ns[finite]
    T1err_ns = T1err_ns[finite]
    raw_fluxs = np.asarray(value2flux(values, flux_half, flux_period), dtype=np.float64)
    if correct_flux_from_f01_enabled:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            correction = correct_flux_from_f01(
                values,
                1e-3 * f01_mhz,
                params,
                flux_half,
                flux_period,
                max_abs_flux_correction=max_abs_flux_correction,
            )
        corrected_values = correction.corrected_dev_values
        corrected_fluxs = correction.corrected_fluxs
        accepted = correction.accepted
        flux_corrections = correction.applied_flux_corrections
    else:
        corrected_values = values
        corrected_fluxs = raw_fluxs
        accepted = np.ones_like(values, dtype=bool)
        flux_corrections = np.zeros_like(values, dtype=np.float64)
    in_window = (
        np.isfinite(corrected_fluxs)
        & (corrected_fluxs >= analysis_flux_range[0])
        & (corrected_fluxs <= analysis_flux_range[1])
    )
    order = np.argsort(corrected_fluxs[in_window])
    return T1CurveData(
        current_raw=current_raw[in_window][order],
        values=corrected_values[in_window][order],
        raw_fluxs=raw_fluxs[in_window][order],
        fluxs=corrected_fluxs[in_window][order],
        f01_mhz=f01_mhz[in_window][order],
        T1_ns=T1_ns[in_window][order],
        T1err_ns=T1err_ns[in_window][order],
        omegas=np.asarray(
            freq2omega(1e-3 * f01_mhz[in_window][order]), dtype=np.float64
        ),
        f01_correction_accepted=accepted[in_window][order],
        flux_corrections=flux_corrections[in_window][order],
    )


def _filter_fit_data(
    sample: T1CurveData,
    *,
    max_rel_t1_err: float,
    use_weighted_points_only: bool,
) -> T1CurveData:
    finite_err = np.isfinite(sample.T1err_ns) & (sample.T1err_ns > 0.0)
    bounded_err = finite_err & (sample.T1err_ns <= max_rel_t1_err * sample.T1_ns)
    if use_weighted_points_only:
        valid = bounded_err
    else:
        valid = np.isnan(sample.T1err_ns) | bounded_err
    return _take_t1_data(sample, valid)


def _take_t1_data(data: T1CurveData, mask: NDArray[np.bool_]) -> T1CurveData:
    return T1CurveData(
        current_raw=data.current_raw[mask],
        values=data.values[mask],
        raw_fluxs=data.raw_fluxs[mask],
        fluxs=data.fluxs[mask],
        f01_mhz=data.f01_mhz[mask],
        T1_ns=data.T1_ns[mask],
        T1err_ns=data.T1err_ns[mask],
        omegas=data.omegas[mask],
        f01_correction_accepted=data.f01_correction_accepted[mask],
        flux_corrections=data.flux_corrections[mask],
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


def _purcell_rate_fn(
    data: T1PreparedData,
    purcell: PurcellEffectParams,
) -> Callable[[T1FitParams], NDArray[np.float64]]:
    def extra_rate(params: T1FitParams) -> NDArray[np.float64]:
        purcell_T1s = calculate_purcell_t1_limit(
            data.calibration.context,
            data.fit.fluxs,
            purcell,
            Temp=float(params.Temp),
        )
        return _relaxation_rates_from_t1(
            purcell_T1s,
            name="Purcell T1",
        )

    return extra_rate


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


def _purcell_fit_summary_rows(
    purcell: PurcellEffectParams | None,
) -> list[tuple[str, str]]:
    if purcell is None:
        return [("Purcell", "off")]
    return [
        ("Purcell", "on"),
        ("Purcell kappa (GHz)", f"{purcell.kappa_ghz:.6g}"),
        ("Purcell bare_rf (GHz)", f"{purcell.bare_rf:.6g}"),
        ("Purcell g (GHz)", f"{purcell.g:.6g}"),
    ]


def _resolve_channel_purcell(
    combined_fit: T1CombinedFit,
    *,
    purcell: PurcellEffectParams | None,
    include_purcell: bool | None,
) -> PurcellEffectParams | None:
    if purcell is not None:
        return purcell
    if include_purcell is True:
        if combined_fit.purcell is None:
            raise ValueError(
                "pass purcell=PurcellEffectParams(...) when include_purcell=True"
            )
        return combined_fit.purcell
    if include_purcell is False:
        return None
    return combined_fit.purcell


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


def _relaxation_rates_from_t1(
    T1_ns: NDArray[np.float64],
    *,
    name: str,
) -> NDArray[np.float64]:
    T1s = _validate_t1_limit(T1_ns, name=name)
    return np.divide(
        1.0,
        T1s,
        out=np.full_like(T1s, np.nan, dtype=np.float64),
        where=np.isfinite(T1s) & (T1s > 0.0),
    )


def _combine_t1_limits(
    *limits: NDArray[np.float64],
) -> NDArray[np.float64]:
    if not limits:
        raise ValueError("at least one T1 limit is required")
    first = np.asarray(limits[0], dtype=np.float64)
    total_rate = np.zeros_like(first, dtype=np.float64)
    for limit in limits:
        rates = _relaxation_rates_from_t1(
            np.asarray(limit, dtype=np.float64), name="T1"
        )
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
) -> float:
    mask = _omega_mask(data.fit.omegas, omega_range)

    def calc_Q_fn(candidate_Temp: float) -> NDArray[np.float64]:
        purcell_correction = (
            _subtract_purcell_from_fit_data(data, purcell, candidate_Temp)
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

    return find_proper_Temp(Temp, calc_Q_fn)


def _omega_mask(
    omegas: NDArray[np.float64],
    omega_range: tuple[float | None, float | None],
) -> NDArray[np.bool_]:
    mask = np.ones_like(omegas, dtype=bool)
    lower, upper = omega_range
    if lower is not None:
        mask &= omegas >= lower
    if upper is not None:
        mask &= omegas <= upper
    return mask


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


def _positive_override(override: float | None, fallback: float, name: str) -> float:
    value = fallback if override is None else override
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _fit_init_value(
    override: float | None,
    probe: T1MechanismProbe | None,
    name: NoiseParameterName,
) -> float:
    probe_value = None if probe is None else probe.parameter_init
    value = override if override is not None else probe_value
    if value is None:
        raise ValueError(
            f"{name} needs an explicit value or a completed mechanism probe"
        )
    return _positive_override(None, value, name)


def _validate_mechanisms(active_mechanisms: tuple[MechanismName, ...]) -> None:
    if not active_mechanisms:
        raise ValueError("at least one T1 mechanism must be active")
    if len(set(active_mechanisms)) != len(active_mechanisms):
        raise ValueError("active_mechanisms contains duplicates")
    known = {"capacitive", "quasiparticle", "inductive"}
    unknown = set(active_mechanisms) - known
    if unknown:
        raise ValueError(f"unknown active mechanism(s): {sorted(unknown)}")


def t1_noise_channels_from_params(
    params: T1FitParams,
) -> list[tuple[str, dict[str, float]]]:
    return [
        (channel_name, {option_name: value})
        for _param_name, channel_name, option_name, value in _iter_noise_channel_values(
            params
        )
    ]


def _iter_noise_channels(
    params: T1FitParams,
) -> list[tuple[NoiseParameterName, str, dict[str, float]]]:
    return [
        (param_name, channel_name, {option_name: value})
        for param_name, channel_name, option_name, value in _iter_noise_channel_values(
            params
        )
    ]


def _iter_noise_channel_values(
    params: T1FitParams,
) -> list[tuple[NoiseParameterName, str, str, float]]:
    values: list[tuple[NoiseParameterName, str, str, float]] = []
    if params.Q_cap is not None:
        channel_name, option_name = _NOISE_CHANNELS["Q_cap"]
        values.append(("Q_cap", channel_name, option_name, params.Q_cap))
    if params.x_qp is not None:
        channel_name, option_name = _NOISE_CHANNELS["x_qp"]
        values.append(("x_qp", channel_name, option_name, params.x_qp))
    if params.Q_ind is not None:
        channel_name, option_name = _NOISE_CHANNELS["Q_ind"]
        values.append(("Q_ind", channel_name, option_name, params.Q_ind))
    return values


def _fit_params_table(init: T1FitParams, fit_result: T1FitResult) -> pd.DataFrame:
    rows = []
    for name in ("Q_cap", "x_qp", "Q_ind", "Temp"):
        fit_value = _param_value(fit_result.params, name)
        if fit_value is None:
            continue
        init_value = _param_value(init, name)
        stderr = _param_value(fit_result.stderr, name)
        rows.append(
            {
                "parameter": name,
                "init": init_value,
                "fit": fit_value,
                "stderr": stderr,
                "display": _format_parameter(cast(ParameterName, name), fit_value),
            }
        )
    return pd.DataFrame(rows)


def _param_value(params: T1FitParams, name: str) -> float | None:
    if name == "Q_cap":
        return params.Q_cap
    if name == "x_qp":
        return params.x_qp
    if name == "Q_ind":
        return params.Q_ind
    if name == "Temp":
        return params.Temp
    raise ValueError(f"unknown T1 parameter: {name}")


def _error_fill_summary(result: object | None) -> str:
    if result is None:
        return "no error column"
    bin_fill_mask = getattr(result, "bin_fill_mask")
    global_fill_mask = getattr(result, "global_fill_mask")
    fallback_fill_mask = getattr(result, "fallback_fill_mask")
    filled = int(
        np.count_nonzero(bin_fill_mask | global_fill_mask | fallback_fill_mask)
    )
    if filled == 0:
        return "0"
    return (
        f"{filled} (bin={int(np.count_nonzero(bin_fill_mask))}, "
        f"global={int(np.count_nonzero(global_fill_mask))}, "
        f"fallback={int(np.count_nonzero(fallback_fill_mask))})"
    )


def _prepared_summary_table(
    *,
    calibration: T1FluxCalibration,
    sample: T1CurveData,
    fit: T1CurveData,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    max_rel_t1_err: float,
    use_weighted_points_only: bool,
) -> pd.DataFrame:
    peak_idx = int(np.nanargmax(sample.T1_ns))
    return pd.DataFrame(
        [
            (
                "analysis flux range",
                f"{analysis_flux_range[0]:.3f}..{analysis_flux_range[1]:.3f}",
            ),
            ("max abs flux correction", f"{max_abs_flux_correction:.3g}"),
            ("max relative T1err", f"{max_rel_t1_err:.3g}"),
            ("use weighted points only", str(use_weighted_points_only)),
            ("window T1 rows", f"{len(sample.fluxs)}/{len(calibration.t1_df)}"),
            ("fit rows", str(len(fit.fluxs))),
            ("current scale", f"{calibration.current_scale:g}"),
            (
                "sample peak",
                f"flux={sample.fluxs[peak_idx]:.6f}, T1={1e-3 * sample.T1_ns[peak_idx]:.3f} us",
            ),
            (
                "f01 corrections accepted",
                f"{int(np.count_nonzero(sample.f01_correction_accepted))}/{len(sample.fluxs)}",
            ),
        ],
        columns=["metric", "value"],
    )


def _stage_table(stage: str, table: pd.DataFrame) -> pd.DataFrame:
    stage_table = table.copy()
    stage_table.insert(0, "stage", stage)
    return stage_table


def _format_range(value_range: tuple[float | None, float | None]) -> str:
    lower, upper = value_range
    lower_text = "-inf" if lower is None else f"{lower:g}"
    upper_text = "inf" if upper is None else f"{upper:g}"
    return f"{lower_text}..{upper_text}"


def _format_parameter(name: str, value: float) -> str:
    return f"{name} = {_format_parameter_value(name, value)}"


def _format_parameter_value(name: str, value: float) -> str:
    if name == "Temp":
        return f"{value * 1e3:.2f} mK"
    return f"{value:.3e}"


__all__ = [
    "MechanismName",
    "MechanismOrParamName",
    "ProbeStatistic",
    "PurcellCorrectionResult",
    "PurcellEffectParams",
    "T1ChannelAnalysis",
    "T1ChannelCurves",
    "T1CombinedFit",
    "T1CurveAnalysisConfig",
    "T1CurveAnalysisResult",
    "T1CurveContext",
    "T1CurveData",
    "T1FluxCalibration",
    "T1MechanismProbe",
    "T1PreparedData",
    "analyze_t1_capacitive_limit",
    "analyze_t1_inductive_limit",
    "analyze_t1_mechanism_limit",
    "analyze_t1_quasiparticle_limit",
    "build_t1_channel_curves",
    "calculate_purcell_t1_limit",
    "calibrate_t1_flux",
    "clear_t1_purcell_cache",
    "collect_t1_curve_result",
    "fit_t1_curve",
    "load_t1_curve_context",
    "make_t1_fit_bounds",
    "make_t1_fit_init",
    "mechanisms_to_fixed_params",
    "plot_t1_channel_analysis",
    "plot_t1_curve_data",
    "plot_t1_flux_calibration",
    "plot_t1_mechanism_dipole",
    "plot_t1_mechanism_limit",
    "plot_t1_mechanism_probe",
    "prepare_t1_curve_data",
    "run_t1_curve_analysis",
    "save_t1_curve_figure",
    "subtract_relaxation_limit",
    "t1_noise_channels_from_params",
    "t1_parameter_text",
    "write_t1_curve_fit",
]

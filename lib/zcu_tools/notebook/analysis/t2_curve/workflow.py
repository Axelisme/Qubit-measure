from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.meta_tool import QubitParams, T1CurveFit
from zcu_tools.notebook.analysis.fit_tools import (
    ErrorResolutionResult,
    FluxResidualWeighting,
    MeasurementErrorPolicy,
    correct_flux_from_f01,
)
from zcu_tools.simulate import value2flux

from .base import (
    T2ChannelCurves,
    calculate_t2_channel_curves,
    choose_current_scale,
    dispersive_chi01_over_2pi_mhz,
    make_thermal_limit_table,
    plot_flux_noise_sensitivity,
    plot_t2_channel_curves,
    plot_t2e_vs_flux,
    plot_thermal_photon_t2_limit,
    predict_domega_dflux,
    t2_parameter_text,
)
from .fit import (
    ParameterName,
    ResidualMode,
    T2FitParams,
    T2FitResult,
    equivalent_n_th_from_t2,
    fit_t2_noise_params,
    flux_noise_gamma_phi_per_us,
    thermal_photon_gamma_phi_per_us,
)

MechanismName = Literal["flux_noise", "photon_shot_noise"]
MechanismOrParamName = Literal["flux_noise", "photon_shot_noise", "A_phi", "n_th"]
ProbeStatistic = Literal["min", "median", "mean", "p90", "max", "half_flux"]

DisplayFn = Callable[[object], None]

_REQUIRED_COLUMNS = [
    "calibrated mA",
    "Freq (MHz)",
    "T1 (us)",
    "T1err (us)",
    "T2e (us)",
    "T2e err (us)",
]
_DEFAULT_THERMAL_PROBE_N_TH = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@dataclass(frozen=True, slots=True)
class T2CurveAnalysisConfig:
    result_dir: str
    analysis_flux_range: tuple[float, float] = (0.49, 0.53)
    readout_kappa_over_2pi_mhz: float = 14.75412896809815
    image_dir: str | None = None
    samples_filename: str = "samples.csv"
    default_bare_rf: float = 5.0
    t_flux_count: int = 1000
    current_scale_candidates: tuple[float, ...] = (1.0, 1000.0)
    max_abs_flux_correction: float = 0.03
    max_rel_t2e_err: float = 0.5
    use_weighted_points_only: bool = False
    T1_error_policy: MeasurementErrorPolicy = field(
        default_factory=lambda: MeasurementErrorPolicy(
            nan_policy="bin_median",
            fallback_error=1.0,
        )
    )
    T2e_error_policy: MeasurementErrorPolicy = field(
        default_factory=lambda: MeasurementErrorPolicy(
            nan_policy="bin_median",
            absolute_floor=0.2,
            relative_floor=0.05,
        )
    )
    flux_weighting: FluxResidualWeighting = field(
        default_factory=lambda: FluxResidualWeighting(
            mode="equal_flux_bin",
            bin_width=0.002,
        )
    )
    active_mechanisms: tuple[MechanismName, ...] = ("flux_noise", "photon_shot_noise")
    fixed_mechanisms: tuple[MechanismOrParamName, ...] = ()
    fit_A_phi_init: float | None = None
    fit_n_th_init: float | None = None
    fit_bounds: Mapping[str, tuple[float, float]] | None = None
    flux_probe_min_sensitivity_fraction: float = 1e-3
    residual_mode: ResidualMode = "gamma"
    loss: str = "linear"
    max_nfev: int = 10000
    thermal_probe_n_th: tuple[float, ...] = _DEFAULT_THERMAL_PROBE_N_TH
    thermal_n_th_range: tuple[float, float] = (1e-5, 1e-1)
    thermal_n_th_count: int = 400
    progress: bool = True
    verbose: bool = True
    display_tables: bool = True
    save_figures: bool = True
    show_figures: bool = True


@dataclass(frozen=True, slots=True)
class T2CurveContext:
    result_dir: str
    image_dir: str
    samples_filename: str
    params: tuple[float, float, float]
    flux_half: float
    flux_int: float
    flux_period: float
    bare_rf: float
    g: float
    samples_df: pd.DataFrame
    t1_curve_fit: T1CurveFit | None
    params_table: pd.DataFrame
    samples_preview: pd.DataFrame
    available_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class T2FluxCalibration:
    context: T2CurveContext
    current_scale: float
    scale_report: pd.DataFrame
    t2e_df: pd.DataFrame
    t2r_df: pd.DataFrame
    freq_rows: pd.DataFrame
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class T2CurveData:
    values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    f01_mhz: NDArray[np.float64]
    T1_us: NDArray[np.float64]
    T1_err_us: NDArray[np.float64]
    T2e_us: NDArray[np.float64]
    T2e_err_us: NDArray[np.float64]
    gamma_phi_per_us: NDArray[np.float64]
    gamma_phi_err_per_us: NDArray[np.float64] | None
    Tphi_us: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class T2WindowData:
    current_raw: NDArray[np.float64]
    values: NDArray[np.float64]
    raw_fluxs: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    f01_mhz: NDArray[np.float64]
    T2e_us: NDArray[np.float64]
    T2e_err_us: NDArray[np.float64]
    T1_us: NDArray[np.float64]
    T1_err_us: NDArray[np.float64]
    f01_correction_accepted: NDArray[np.bool_]
    flux_corrections: NDArray[np.float64]
    kept_rows: int
    source_rows: int


@dataclass(frozen=True, slots=True)
class T2DephasingAnalysis:
    calibration: T2FluxCalibration
    analysis_flux_range: tuple[float, float]
    max_abs_flux_correction: float
    max_rel_t2e_err: float
    use_weighted_points_only: bool
    window: T2WindowData
    sample: T2CurveData
    fit: T2CurveData
    branch_coverage: pd.DataFrame
    half_preview: pd.DataFrame
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class T2CurveThermalEstimate:
    half_flux: float
    half_T1_us: float
    half_T2e_us: float
    half_gamma_phi_per_us: float
    half_chi_over_2pi_mhz: float
    half_gamma_per_photon_us: float
    half_equivalent_n_th: float
    fit_peak_equivalent_n_th: float
    fit_peak_flux: float


@dataclass(frozen=True, slots=True)
class T2FluxNoiseProbe:
    data: T2DephasingAnalysis
    readout_kappa_over_2pi_mhz: float
    assumed_n_th: float
    A_phi_init: float
    A_phi_fit: float
    A_phi_stderr: float
    A_phi_upper: float
    domega_dflux: NDArray[np.float64]
    chi_over_2pi_mhz: NDArray[np.float64]
    residual_gamma_phi_per_us: NDArray[np.float64]
    pointwise_table: pd.DataFrame
    summary_table: pd.DataFrame
    fit_result: T2FitResult


@dataclass(frozen=True, slots=True)
class T2PhotonShotNoiseProbe:
    data: T2DephasingAnalysis
    readout_kappa_over_2pi_mhz: float
    assumed_A_phi: float
    n_th_init: float
    n_th_fit: float
    n_th_stderr: float
    n_th_upper: float
    thermal: T2CurveThermalEstimate
    thermal_limit_table: pd.DataFrame
    domega_dflux: NDArray[np.float64]
    chi_over_2pi_mhz: NDArray[np.float64]
    residual_gamma_phi_per_us: NDArray[np.float64]
    pointwise_table: pd.DataFrame
    summary_table: pd.DataFrame
    fit_result: T2FitResult


@dataclass(frozen=True, slots=True)
class T2CombinedFit:
    data: T2DephasingAnalysis
    readout_kappa_over_2pi_mhz: float
    init: T2FitParams
    bounds: Mapping[str, tuple[float, float]] | None
    fixed: tuple[ParameterName, ...]
    fit_result: T2FitResult
    fit_fluxs: NDArray[np.float64]
    fit_T1_us: NDArray[np.float64]
    fit_T1_err_us: NDArray[np.float64]
    fit_T2e_us: NDArray[np.float64]
    fit_T2e_err_us: NDArray[np.float64]
    domega_dflux: NDArray[np.float64]
    chi_over_2pi_mhz: NDArray[np.float64]
    summary_table: pd.DataFrame
    params_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class T2ChannelAnalysis:
    combined_fit: T2CombinedFit
    curves: T2ChannelCurves
    flux_range: tuple[float, float]
    t_flux_count: int
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class _CombinedFitArrays:
    fluxs: NDArray[np.float64]
    T1_us: NDArray[np.float64]
    T1_err_us: NDArray[np.float64]
    T2e_us: NDArray[np.float64]
    T2e_err_us: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class T2CurveAnalysisResult:
    context: T2CurveContext
    calibration: T2FluxCalibration
    data: T2DephasingAnalysis
    flux_probe: T2FluxNoiseProbe
    photon_probe: T2PhotonShotNoiseProbe
    combined_fit: T2CombinedFit
    channel_analysis: T2ChannelAnalysis
    summary_table: pd.DataFrame
    figure_paths: Mapping[str, str]


def load_t2_curve_context(
    *,
    result_dir: str,
    samples_filename: str = "samples.csv",
    image_dir: str | None = None,
    default_bare_rf: float = 5.0,
    preview_rows: int = 10,
) -> T2CurveContext:
    resolved_image_dir = image_dir or os.path.join(result_dir, "t2_curve")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(resolved_image_dir, exist_ok=True)

    params_file = QubitParams(os.path.join(result_dir, "params.json"), readonly=True)
    fit_inputs = params_file.require_dispersive_inputs(default_bare_rf=default_bare_rf)
    dispersive_fit = params_file.get_dispersive_fit()
    if dispersive_fit is None:
        raise RuntimeError("params.json needs a dispersive section with g and bare_rf.")

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
            ("bare_rf (GHz)", dispersive_fit.bare_rf),
            ("g (GHz)", dispersive_fit.g),
        ],
        columns=["parameter", "value"],
    )

    return T2CurveContext(
        result_dir=result_dir,
        image_dir=resolved_image_dir,
        samples_filename=samples_filename,
        params=params,
        flux_half=fit_inputs.flux_half,
        flux_int=fit_inputs.flux_int,
        flux_period=fit_inputs.flux_period,
        bare_rf=dispersive_fit.bare_rf,
        g=dispersive_fit.g,
        samples_df=samples_df,
        t1_curve_fit=params_file.get_t1_curve_fit(),
        params_table=params_table,
        samples_preview=samples_df.head(preview_rows),
        available_columns=tuple(str(column) for column in samples_df.columns),
    )


def calibrate_t2_flux(
    context: T2CurveContext,
    *,
    current_scale_candidates: tuple[float, ...] = (1.0, 1000.0),
) -> T2FluxCalibration:
    samples_df = context.samples_df
    t2e_df = cast(
        pd.DataFrame,
        samples_df.loc[np.isfinite(_float_column(samples_df, "T2e (us)"))].copy(),
    )
    if "T2r (us)" in samples_df.columns:
        t2r_df = cast(
            pd.DataFrame,
            samples_df.loc[np.isfinite(_float_column(samples_df, "T2r (us)"))].copy(),
        )
    else:
        t2r_df = samples_df.iloc[0:0].copy()

    freq_rows = cast(
        pd.DataFrame,
        samples_df.loc[
            np.isfinite(_float_column(samples_df, "calibrated mA"))
            & np.isfinite(_float_column(samples_df, "Freq (MHz)"))
        ].copy(),
    )
    current_scale, scale_report = choose_current_scale(
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
            ("finite T2e rows", str(len(t2e_df))),
            ("finite T2r rows", str(len(t2r_df))),
            ("current scale", f"{current_scale:g}"),
        ],
        columns=["metric", "value"],
    )
    return T2FluxCalibration(
        context=context,
        current_scale=current_scale,
        scale_report=scale_report,
        t2e_df=t2e_df,
        t2r_df=t2r_df,
        freq_rows=freq_rows,
        summary_table=summary_table,
    )


def prepare_t2_dephasing_data(
    calibration: T2FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float] = (0.49, 0.53),
    max_abs_flux_correction: float = 0.03,
    max_rel_t2e_err: float = 0.5,
    use_weighted_points_only: bool = False,
) -> T2DephasingAnalysis:
    context = calibration.context
    window = _prepare_window_data(
        calibration.t2e_df,
        params=context.params,
        flux_half=context.flux_half,
        flux_period=context.flux_period,
        current_scale=calibration.current_scale,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
    )
    branch_coverage = _branch_coverage_table(
        calibration.t2e_df,
        calibration.t2r_df,
        params=context.params,
        flux_half=context.flux_half,
        flux_period=context.flux_period,
        current_scale=calibration.current_scale,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
    )
    half_preview = _half_preview_table(
        calibration.t2e_df,
        params=context.params,
        flux_half=context.flux_half,
        flux_period=context.flux_period,
        current_scale=calibration.current_scale,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
    )
    sample, fit = _derive_dephasing(
        window,
        max_rel_t2e_err=max_rel_t2e_err,
        use_weighted_points_only=use_weighted_points_only,
    )
    if len(fit.T2e_us) == 0:
        raise ValueError("No T2e fit rows remain after filtering")

    summary_table = _dephasing_summary_table(
        calibration=calibration,
        window=window,
        sample=sample,
        fit=fit,
        analysis_flux_range=analysis_flux_range,
        max_rel_t2e_err=max_rel_t2e_err,
        use_weighted_points_only=use_weighted_points_only,
    )
    return T2DephasingAnalysis(
        calibration=calibration,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        max_rel_t2e_err=max_rel_t2e_err,
        use_weighted_points_only=use_weighted_points_only,
        window=window,
        sample=sample,
        fit=fit,
        branch_coverage=branch_coverage,
        half_preview=half_preview,
        summary_table=summary_table,
    )


def analyze_flux_noise_limit(
    data: T2DephasingAnalysis,
    *,
    readout_kappa_over_2pi_mhz: float,
    assumed_n_th: float = 0.0,
    A_phi_init: float | None = None,
    A_phi_bounds: tuple[float, float] | None = None,
    statistic: ProbeStatistic = "median",
    min_sensitivity_fraction: float = 1e-3,
    T1_error_policy: MeasurementErrorPolicy | None = None,
    T2_error_policy: MeasurementErrorPolicy | None = None,
    flux_weighting: FluxResidualWeighting | None = None,
    residual_mode: ResidualMode = "gamma",
    loss: str = "linear",
    max_nfev: int | None = 10000,
    progress: bool = False,
) -> T2FluxNoiseProbe:
    context = data.calibration.context
    domega_dflux, chi_over_2pi_mhz = _fit_model_axes(data)
    gamma_photon = _photon_gamma_like(
        assumed_n_th,
        kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
    )
    residual_gamma = data.fit.gamma_phi_per_us - gamma_photon
    sensitivity = np.sqrt(np.log(2.0)) * np.abs(domega_dflux)
    sensitivity_floor = _relative_floor(sensitivity, min_sensitivity_fraction)
    pointwise_A = _safe_divide_positive(
        residual_gamma, sensitivity, min_denominator=sensitivity_floor
    )
    A_init = float(
        A_phi_init or _positive_statistic(pointwise_A, data.fit.fluxs, statistic)
    )
    bounds = {"A_phi": A_phi_bounds or _auto_bounds(A_init)}
    init = T2FitParams(
        A_phi=A_init,
        n_th=float(assumed_n_th) if assumed_n_th > 0.0 else None,
    )
    fixed: tuple[ParameterName, ...] = ("n_th",) if assumed_n_th > 0.0 else ()

    fit_result = fit_t2_noise_params(
        data.fit.T1_us,
        data.fit.T2e_us,
        domega_dflux,
        chi_over_2pi_mhz,
        kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        init=init,
        bounds=bounds,
        fixed=fixed,
        T1errs=data.fit.T1_err_us,
        T2errs=data.fit.T2e_err_us,
        fluxs=data.fit.fluxs,
        T1_error_policy=T1_error_policy,
        T2_error_policy=T2_error_policy,
        flux_weighting=flux_weighting,
        residual_mode=residual_mode,
        loss=loss,
        max_nfev=max_nfev,
        progress=progress,
    )
    A_fit = _required_param(fit_result.params.A_phi, "A_phi")
    A_stderr = _optional_float(fit_result.stderr.A_phi)
    finite_pointwise = _finite_positive(pointwise_A)
    A_upper = float(np.nanmax(finite_pointwise))
    pointwise_table = pd.DataFrame(
        {
            "flux": data.fit.fluxs,
            "gamma_phi_obs (1/us)": data.fit.gamma_phi_per_us,
            "gamma_phi_photon_assumed (1/us)": gamma_photon,
            "gamma_phi_flux_target (1/us)": residual_gamma,
            "domega_dflux (rad/us/Phi0)": domega_dflux,
            "A_phi (Phi0/sqrtHz)": pointwise_A,
            "A_phi (uPhi0/sqrtHz)": 1e6 * pointwise_A,
        }
    )
    summary_table = pd.DataFrame(
        [
            ("mechanism", "flux_noise"),
            ("assumed n_th", f"{assumed_n_th:.3e}"),
            ("A_phi init", f"{A_init * 1e6:.3f} uPhi0/sqrtHz"),
            ("A_phi fit", f"{A_fit * 1e6:.3f} uPhi0/sqrtHz"),
            ("A_phi stderr", f"{A_stderr * 1e6:.3f} uPhi0/sqrtHz"),
            ("A_phi pointwise upper", f"{A_upper * 1e6:.3f} uPhi0/sqrtHz"),
            (
                "A_phi sensitivity floor",
                f"{sensitivity_floor:.3g} 1/us/Phi0",
            ),
            ("fit success", str(fit_result.success)),
            ("reduced chi2", f"{fit_result.reduced_chi2:.3g}"),
            ("params source", context.result_dir),
        ],
        columns=["metric", "value"],
    )
    return T2FluxNoiseProbe(
        data=data,
        readout_kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        assumed_n_th=assumed_n_th,
        A_phi_init=A_init,
        A_phi_fit=A_fit,
        A_phi_stderr=A_stderr,
        A_phi_upper=A_upper,
        domega_dflux=domega_dflux,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
        residual_gamma_phi_per_us=residual_gamma,
        pointwise_table=pointwise_table,
        summary_table=summary_table,
        fit_result=fit_result,
    )


def analyze_photon_shot_noise_limit(
    data: T2DephasingAnalysis,
    *,
    readout_kappa_over_2pi_mhz: float,
    assumed_A_phi: float = 0.0,
    n_th_init: float | None = None,
    n_th_bounds: tuple[float, float] | None = None,
    statistic: ProbeStatistic = "min",
    thermal_probe_n_th: tuple[float, ...] = _DEFAULT_THERMAL_PROBE_N_TH,
    T1_error_policy: MeasurementErrorPolicy | None = None,
    T2_error_policy: MeasurementErrorPolicy | None = None,
    flux_weighting: FluxResidualWeighting | None = None,
    residual_mode: ResidualMode = "gamma",
    loss: str = "linear",
    max_nfev: int | None = 10000,
    progress: bool = False,
) -> T2PhotonShotNoiseProbe:
    domega_dflux, chi_over_2pi_mhz = _fit_model_axes(data)
    gamma_flux = (
        flux_noise_gamma_phi_per_us(float(assumed_A_phi), domega_dflux)
        if assumed_A_phi > 0.0
        else np.zeros_like(data.fit.gamma_phi_per_us)
    )
    residual_gamma = data.fit.gamma_phi_per_us - gamma_flux
    gamma_per_photon = _photon_gamma_like(
        1.0,
        kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
    )
    pointwise_n_th = _safe_divide_positive(residual_gamma, gamma_per_photon)
    n_init = float(
        n_th_init or _positive_statistic(pointwise_n_th, data.fit.fluxs, statistic)
    )
    bounds = {"n_th": n_th_bounds or _auto_bounds(n_init, lower_floor=1e-8)}
    init = T2FitParams(
        A_phi=float(assumed_A_phi) if assumed_A_phi > 0.0 else None,
        n_th=n_init,
    )
    fixed: tuple[ParameterName, ...] = ("A_phi",) if assumed_A_phi > 0.0 else ()

    fit_result = fit_t2_noise_params(
        data.fit.T1_us,
        data.fit.T2e_us,
        domega_dflux,
        chi_over_2pi_mhz,
        kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        init=init,
        bounds=bounds,
        fixed=fixed,
        T1errs=data.fit.T1_err_us,
        T2errs=data.fit.T2e_err_us,
        fluxs=data.fit.fluxs,
        T1_error_policy=T1_error_policy,
        T2_error_policy=T2_error_policy,
        flux_weighting=flux_weighting,
        residual_mode=residual_mode,
        loss=loss,
        max_nfev=max_nfev,
        progress=progress,
    )
    n_fit = _required_param(fit_result.params.n_th, "n_th")
    n_stderr = _optional_float(fit_result.stderr.n_th)
    finite_pointwise = _finite_positive(pointwise_n_th)
    n_upper = float(np.nanmax(finite_pointwise))
    context = data.calibration.context
    thermal = _thermal_estimate(
        data.sample,
        data.fit,
        params=context.params,
        bare_rf=context.bare_rf,
        g=context.g,
        kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
    )
    thermal_limit_table = make_thermal_limit_table(
        np.asarray([*thermal_probe_n_th, thermal.half_equivalent_n_th, n_fit]),
        T1_us=thermal.half_T1_us,
        kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        chi_over_2pi_mhz=thermal.half_chi_over_2pi_mhz,
    )
    pointwise_table = pd.DataFrame(
        {
            "flux": data.fit.fluxs,
            "gamma_phi_obs (1/us)": data.fit.gamma_phi_per_us,
            "gamma_phi_flux_assumed (1/us)": gamma_flux,
            "gamma_phi_photon_target (1/us)": residual_gamma,
            "chi_over_2pi (MHz)": chi_over_2pi_mhz,
            "gamma_per_photon (1/us)": gamma_per_photon,
            "n_th": pointwise_n_th,
        }
    )
    summary_table = pd.DataFrame(
        [
            ("mechanism", "photon_shot_noise"),
            ("assumed A_phi", f"{assumed_A_phi * 1e6:.3f} uPhi0/sqrtHz"),
            ("n_th probe result", f"{n_init:.3e}"),
            ("n_th photon-only fit", f"{n_fit:.3e}"),
            ("n_th stderr", f"{n_stderr:.3e}"),
            ("n_th pointwise min", f"{np.nanmin(finite_pointwise):.3e}"),
            ("n_th pointwise max", f"{n_upper:.3e}"),
            ("half-flux equivalent n_th", f"{thermal.half_equivalent_n_th:.3e}"),
            ("fit success", str(fit_result.success)),
            ("reduced chi2", f"{fit_result.reduced_chi2:.3g}"),
        ],
        columns=["metric", "value"],
    )
    return T2PhotonShotNoiseProbe(
        data=data,
        readout_kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        assumed_A_phi=assumed_A_phi,
        n_th_init=n_init,
        n_th_fit=n_fit,
        n_th_stderr=n_stderr,
        n_th_upper=n_upper,
        thermal=thermal,
        thermal_limit_table=thermal_limit_table,
        domega_dflux=domega_dflux,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
        residual_gamma_phi_per_us=residual_gamma,
        pointwise_table=pointwise_table,
        summary_table=summary_table,
        fit_result=fit_result,
    )


def make_t2_fit_init(
    *,
    active_mechanisms: tuple[MechanismName, ...],
    flux_probe: T2FluxNoiseProbe | None = None,
    photon_probe: T2PhotonShotNoiseProbe | None = None,
    A_phi: float | None = None,
    n_th: float | None = None,
) -> T2FitParams:
    _validate_mechanisms(active_mechanisms)
    return T2FitParams(
        A_phi=(
            _fit_init_value(
                A_phi, flux_probe.A_phi_fit if flux_probe else None, "A_phi"
            )
            if "flux_noise" in active_mechanisms
            else None
        ),
        n_th=(
            _fit_init_value(
                n_th, photon_probe.n_th_init if photon_probe else None, "n_th"
            )
            if "photon_shot_noise" in active_mechanisms
            else None
        ),
    )


def mechanisms_to_fixed_params(
    fixed_mechanisms: tuple[MechanismOrParamName, ...],
) -> tuple[ParameterName, ...]:
    fixed: list[ParameterName] = []
    for name in fixed_mechanisms:
        if name in ("flux_noise", "A_phi"):
            fixed.append("A_phi")
        elif name in ("photon_shot_noise", "n_th"):
            fixed.append("n_th")
        else:
            raise ValueError(f"unknown T2 mechanism or parameter: {name}")
    if len(set(fixed)) != len(fixed):
        raise ValueError("fixed_mechanisms contains duplicate parameters")
    return tuple(fixed)


def make_t2_fit_bounds(
    init: T2FitParams,
    *,
    factor: float = 1000.0,
    A_phi_lower_floor: float = 1e-8,
    n_th_lower_floor: float = 1e-8,
    A_phi_upper_cap: float = 1e-4,
    n_th_upper_cap: float = 1e-1,
) -> dict[str, tuple[float, float]]:
    if factor <= 1.0:
        raise ValueError("factor must be larger than 1")

    bounds: dict[str, tuple[float, float]] = {}
    if init.A_phi is not None:
        bounds["A_phi"] = (
            max(A_phi_lower_floor, init.A_phi / factor),
            min(A_phi_upper_cap, init.A_phi * factor),
        )
    if init.n_th is not None:
        bounds["n_th"] = (
            max(n_th_lower_floor, init.n_th / factor),
            min(n_th_upper_cap, init.n_th * factor),
        )
    return bounds


def fit_t2_curve(
    data: T2DephasingAnalysis,
    *,
    readout_kappa_over_2pi_mhz: float,
    init: T2FitParams,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    fixed: tuple[ParameterName, ...] = (),
    residual_mode: ResidualMode = "gamma",
    loss: str = "linear",
    max_nfev: int | None = 10000,
    progress: bool = False,
    T1_error_policy: MeasurementErrorPolicy | None = None,
    T2_error_policy: MeasurementErrorPolicy | None = None,
    flux_weighting: FluxResidualWeighting | None = None,
) -> T2CombinedFit:
    fit_arrays = _combined_fit_arrays(data)
    domega_dflux, chi_over_2pi_mhz = _fit_model_axes(data, fit_arrays.fluxs)
    fit_result = fit_t2_noise_params(
        fit_arrays.T1_us,
        fit_arrays.T2e_us,
        domega_dflux,
        chi_over_2pi_mhz,
        kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        init=init,
        bounds=bounds,
        fixed=fixed,
        T1errs=fit_arrays.T1_err_us,
        T2errs=fit_arrays.T2e_err_us,
        fluxs=fit_arrays.fluxs,
        T1_error_policy=T1_error_policy,
        T2_error_policy=T2_error_policy,
        flux_weighting=flux_weighting,
        residual_mode=residual_mode,
        loss=loss,
        max_nfev=max_nfev,
        progress=progress,
    )
    params_table = _fit_params_table(init, fit_result)
    summary_table = pd.DataFrame(
        [
            ("fit success", str(fit_result.success)),
            ("fit message", fit_result.message),
            ("fixed", str(fit_result.fixed)),
            ("free", str(fit_result.free)),
            ("fit rows", str(len(fit_arrays.fluxs))),
            (
                "effective flux bins",
                f"{fit_result.flux_weights.effective_observation_count:g}",
            ),
            ("flux weighting", fit_result.flux_weights.mode),
            ("reduced chi2", f"{fit_result.reduced_chi2:.3g}"),
            (
                "T2e err NaN filled",
                _error_fill_summary(fit_result.T2_error_resolution),
            ),
        ],
        columns=["metric", "value"],
    )
    return T2CombinedFit(
        data=data,
        readout_kappa_over_2pi_mhz=readout_kappa_over_2pi_mhz,
        init=init,
        bounds=bounds,
        fixed=fixed,
        fit_result=fit_result,
        fit_fluxs=fit_arrays.fluxs,
        fit_T1_us=fit_arrays.T1_us,
        fit_T1_err_us=fit_arrays.T1_err_us,
        fit_T2e_us=fit_arrays.T2e_us,
        fit_T2e_err_us=fit_arrays.T2e_err_us,
        domega_dflux=domega_dflux,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
        summary_table=summary_table,
        params_table=params_table,
    )


def build_t2_channel_curves(
    combined_fit: T2CombinedFit,
    *,
    t_flux_count: int = 1000,
    flux_range: tuple[float, float] | None = None,
    use_t1_curve_fit: bool = True,
) -> T2ChannelAnalysis:
    data = combined_fit.data
    context = data.calibration.context
    resolved_flux_range = flux_range or data.analysis_flux_range
    grid_fluxs = np.linspace(
        resolved_flux_range[0], resolved_flux_range[1], t_flux_count
    )
    t1_curve_fit = context.t1_curve_fit if use_t1_curve_fit else None
    curves = calculate_t2_channel_curves(
        grid_fluxs,
        params=context.params,
        fit_result=combined_fit.fit_result,
        kappa_over_2pi_mhz=combined_fit.readout_kappa_over_2pi_mhz,
        chi_over_2pi_mhz=dispersive_chi01_over_2pi_mhz(
            context.params, grid_fluxs, context.bare_rf, context.g
        ),
        domega_dflux=predict_domega_dflux(context.params, grid_fluxs),
        t1_curve_fit=t1_curve_fit,
        fit_fluxs=combined_fit.fit_fluxs,
        fit_T1_us=combined_fit.fit_T1_us,
    )
    summary_table = pd.DataFrame(
        [
            (
                "flux range",
                f"{resolved_flux_range[0]:.3f}..{resolved_flux_range[1]:.3f}",
            ),
            ("t_flux_count", str(t_flux_count)),
            ("T1 source", curves.t1_label),
            ("active fit params", str(combined_fit.fit_result.free)),
            ("fixed fit params", str(combined_fit.fit_result.fixed)),
        ],
        columns=["metric", "value"],
    )
    return T2ChannelAnalysis(
        combined_fit=combined_fit,
        curves=curves,
        flux_range=resolved_flux_range,
        t_flux_count=t_flux_count,
        summary_table=summary_table,
    )


def collect_t2_curve_result(
    *,
    context: T2CurveContext,
    calibration: T2FluxCalibration,
    data: T2DephasingAnalysis,
    flux_probe: T2FluxNoiseProbe,
    photon_probe: T2PhotonShotNoiseProbe,
    combined_fit: T2CombinedFit,
    channel_analysis: T2ChannelAnalysis,
    figure_paths: Mapping[str, str] | None = None,
) -> T2CurveAnalysisResult:
    summary_table = pd.concat(
        [
            _stage_table("load", calibration.summary_table),
            _stage_table("dephasing", data.summary_table),
            _stage_table("flux_probe", flux_probe.summary_table),
            _stage_table("photon_probe", photon_probe.summary_table),
            _stage_table("combined_fit", combined_fit.summary_table),
            _stage_table("channel_curves", channel_analysis.summary_table),
        ],
        ignore_index=True,
    )
    return T2CurveAnalysisResult(
        context=context,
        calibration=calibration,
        data=data,
        flux_probe=flux_probe,
        photon_probe=photon_probe,
        combined_fit=combined_fit,
        channel_analysis=channel_analysis,
        summary_table=summary_table,
        figure_paths={} if figure_paths is None else dict(figure_paths),
    )


def plot_t2_dephasing_data(data: T2DephasingAnalysis) -> tuple[Figure, Axes]:
    return plot_t2e_vs_flux(
        data.sample.fluxs,
        data.sample.T2e_us,
        data.sample.T1_us,
        fit_fluxs=data.fit.fluxs,
        fit_T2e_us=data.fit.T2e_us,
        fit_T2e_err_us=data.fit.T2e_err_us,
    )


def plot_t2_flux_calibration(data: T2DephasingAnalysis) -> tuple[Figure, Axes]:
    window = data.window
    if len(window.fluxs) == 0:
        raise ValueError("No T2 samples are available for flux calibration plotting")

    raw_fluxs = window.raw_fluxs
    corrected_fluxs = window.fluxs
    accepted = window.f01_correction_accepted
    y_positions = np.arange(len(corrected_fluxs), dtype=np.float64)
    fig_height = float(np.clip(1.8 + 0.18 * len(corrected_fluxs), 3.0, 8.0))
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    lower, upper = data.analysis_flux_range
    ax.axvspan(lower, upper, color="tab:green", alpha=0.08, label="analysis window")

    for y_position, raw_flux, corrected_flux, is_accepted in zip(
        y_positions,
        raw_fluxs,
        corrected_fluxs,
        accepted,
        strict=True,
    ):
        line_color = "tab:blue" if is_accepted else "tab:gray"
        ax.plot(
            [raw_flux, corrected_flux],
            [y_position, y_position],
            color=line_color,
            alpha=0.55,
            linewidth=1.0,
        )

    ax.scatter(
        raw_fluxs,
        y_positions,
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
            y_positions[accepted],
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
            y_positions[rejected],
            marker="x",
            color="tab:orange",
            s=38,
            label="kept raw flux",
            zorder=4,
        )

    ax.set_xlabel(r"Flux quanta ($\Phi_{ext}/\Phi_0$)")
    ax.set_ylabel("sample index")
    ax.set_title("f01 flux correction")
    ax.set_ylim(-0.5, len(corrected_fluxs) - 0.5)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="best")
    return fig, ax


def plot_flux_noise_probe(probe: T2FluxNoiseProbe) -> tuple[Figure, Axes]:
    return plot_flux_noise_sensitivity(
        probe.domega_dflux,
        probe.residual_gamma_phi_per_us,
        A_phi=probe.A_phi_fit,
        gamma_phi_err=probe.data.fit.gamma_phi_err_per_us,
        title="Flux-noise-only dephasing probe",
    )


def plot_photon_shot_noise_probe(
    probe: T2PhotonShotNoiseProbe,
    *,
    n_th_range: tuple[float, float] = (1e-5, 1e-1),
    n_th_count: int = 400,
) -> tuple[Figure, Axes]:
    n_th_axis = np.logspace(
        np.log10(n_th_range[0]), np.log10(n_th_range[1]), n_th_count
    )
    return plot_thermal_photon_t2_limit(
        n_th_axis,
        T1_us=probe.thermal.half_T1_us,
        T2e_us=probe.thermal.half_T2e_us,
        kappa_over_2pi_mhz=probe.readout_kappa_over_2pi_mhz,
        chi_over_2pi_mhz=probe.thermal.half_chi_over_2pi_mhz,
        equivalent_n_th=probe.thermal.half_equivalent_n_th,
    )


def plot_nth_limit_vs_flux(probe: T2PhotonShotNoiseProbe) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        probe.pointwise_table["flux"],
        probe.pointwise_table["n_th"],
        "o",
        label="pointwise equivalent n_th",
    )
    ax.axhline(
        probe.n_th_init,
        color="tab:red",
        label=f"probe n_th={probe.n_th_init:.2e}",
    )
    ax.axhline(
        probe.n_th_fit,
        color="tab:gray",
        linestyle="--",
        alpha=0.7,
        label=f"photon-only fit={probe.n_th_fit:.2e}",
    )
    ax.set_xlabel(r"Flux $\Phi_\mathrm{ext}/\Phi_0$")
    ax.set_ylabel(r"Equivalent $n_{th}$")
    ax.set_title("Photon-shot-noise-only n_th probe")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_t2_channel_analysis(
    channel_analysis: T2ChannelAnalysis,
    *,
    parameter_text: str | None = None,
) -> tuple[Figure, Axes]:
    data = channel_analysis.combined_fit.data
    if parameter_text is None:
        parameter_text = t2_parameter_text(channel_analysis.combined_fit.fit_result)
    return plot_t2_channel_curves(
        data.sample.fluxs,
        data.sample.T2e_us,
        data.sample.T1_us,
        channel_analysis.curves,
        fit_fluxs=channel_analysis.combined_fit.fit_fluxs,
        fit_T2e_us=channel_analysis.combined_fit.fit_T2e_us,
        fit_T2e_err_us=channel_analysis.combined_fit.fit_T2e_err_us,
        parameter_text=parameter_text,
        xlim=channel_analysis.flux_range,
    )


def save_t2_curve_figure(
    fig: Figure,
    context: T2CurveContext,
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


def run_t2_curve_analysis(
    config: T2CurveAnalysisConfig,
    *,
    display_fn: DisplayFn | None = None,
) -> T2CurveAnalysisResult:
    context = load_t2_curve_context(
        result_dir=config.result_dir,
        samples_filename=config.samples_filename,
        image_dir=config.image_dir,
        default_bare_rf=config.default_bare_rf,
    )
    calibration = calibrate_t2_flux(
        context,
        current_scale_candidates=config.current_scale_candidates,
    )
    data = prepare_t2_dephasing_data(
        calibration,
        analysis_flux_range=config.analysis_flux_range,
        max_abs_flux_correction=config.max_abs_flux_correction,
        max_rel_t2e_err=config.max_rel_t2e_err,
        use_weighted_points_only=config.use_weighted_points_only,
    )
    flux_probe = analyze_flux_noise_limit(
        data,
        readout_kappa_over_2pi_mhz=config.readout_kappa_over_2pi_mhz,
        A_phi_init=config.fit_A_phi_init,
        min_sensitivity_fraction=config.flux_probe_min_sensitivity_fraction,
        residual_mode=config.residual_mode,
        loss=config.loss,
        max_nfev=config.max_nfev,
        T1_error_policy=config.T1_error_policy,
        T2_error_policy=config.T2e_error_policy,
        flux_weighting=config.flux_weighting,
        progress=config.progress,
    )
    photon_probe = analyze_photon_shot_noise_limit(
        data,
        readout_kappa_over_2pi_mhz=config.readout_kappa_over_2pi_mhz,
        n_th_init=config.fit_n_th_init,
        thermal_probe_n_th=config.thermal_probe_n_th,
        residual_mode=config.residual_mode,
        loss=config.loss,
        max_nfev=config.max_nfev,
        T1_error_policy=config.T1_error_policy,
        T2_error_policy=config.T2e_error_policy,
        flux_weighting=config.flux_weighting,
        progress=config.progress,
    )
    fit_init = make_t2_fit_init(
        active_mechanisms=config.active_mechanisms,
        flux_probe=flux_probe,
        photon_probe=photon_probe,
        A_phi=config.fit_A_phi_init,
        n_th=config.fit_n_th_init,
    )
    combined_fit = fit_t2_curve(
        data,
        readout_kappa_over_2pi_mhz=config.readout_kappa_over_2pi_mhz,
        init=fit_init,
        bounds=config.fit_bounds or make_t2_fit_bounds(fit_init),
        fixed=mechanisms_to_fixed_params(config.fixed_mechanisms),
        residual_mode=config.residual_mode,
        loss=config.loss,
        max_nfev=config.max_nfev,
        progress=config.progress,
        T1_error_policy=config.T1_error_policy,
        T2_error_policy=config.T2e_error_policy,
        flux_weighting=config.flux_weighting,
    )
    channel_analysis = build_t2_channel_curves(
        combined_fit,
        t_flux_count=config.t_flux_count,
        flux_range=config.analysis_flux_range,
    )

    figure_paths: dict[str, str] = {}
    if config.save_figures or config.show_figures:
        figure_paths = _save_standard_figures(
            context,
            data,
            flux_probe,
            photon_probe,
            channel_analysis,
            save_figures=config.save_figures,
            show_figures=config.show_figures,
            thermal_n_th_range=config.thermal_n_th_range,
            thermal_n_th_count=config.thermal_n_th_count,
        )
    result = collect_t2_curve_result(
        context=context,
        calibration=calibration,
        data=data,
        flux_probe=flux_probe,
        photon_probe=photon_probe,
        combined_fit=combined_fit,
        channel_analysis=channel_analysis,
        figure_paths=figure_paths,
    )
    _emit_outputs(
        result,
        display_fn=display_fn,
        verbose=config.verbose,
        tables=config.display_tables,
    )
    return result


def _validate_required_columns(samples_df: pd.DataFrame) -> None:
    missing_columns = [
        name for name in _REQUIRED_COLUMNS if name not in samples_df.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required sample columns: {missing_columns}")


def _float_column(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    return np.asarray(frame[column], dtype=np.float64)


def _prepare_window_data(
    t2e_df: pd.DataFrame,
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    current_scale: float,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
) -> T2WindowData:
    t2e_values = _float_column(t2e_df, "T2e (us)")
    valid_t2e = np.isfinite(t2e_values) & (t2e_values > 0.0)
    frame = cast(pd.DataFrame, t2e_df.loc[valid_t2e].copy())

    current_raw = _float_column(frame, "calibrated mA")
    values = current_raw * current_scale
    f01_mhz = _float_column(frame, "Freq (MHz)")
    T2e_us = _float_column(frame, "T2e (us)")
    T2e_err_us = _float_column(frame, "T2e err (us)")
    T1_us = _float_column(frame, "T1 (us)")
    T1_err_us = _float_column(frame, "T1err (us)")

    raw_fluxs = value2flux(values, flux_half, flux_period)
    corr_values = values.copy()
    corr_fluxs = np.asarray(raw_fluxs, dtype=np.float64).copy()
    correction_accepted = np.zeros_like(values, dtype=bool)

    finite_f01 = np.isfinite(f01_mhz)
    if np.any(finite_f01):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            correction = correct_flux_from_f01(
                values[finite_f01],
                1e-3 * f01_mhz[finite_f01],
                params,
                flux_half,
                flux_period,
                max_abs_flux_correction=max_abs_flux_correction,
            )
        corr_values[finite_f01] = correction.corrected_dev_values
        corr_fluxs[finite_f01] = correction.corrected_fluxs
        correction_accepted[finite_f01] = correction.accepted

    flux_window_mask = (
        np.isfinite(corr_fluxs)
        & (corr_fluxs >= analysis_flux_range[0])
        & (corr_fluxs <= analysis_flux_range[1])
    )
    sort_order = np.argsort(corr_fluxs[flux_window_mask])
    flux_corrections = corr_fluxs - raw_fluxs

    def take(values_arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return values_arr[flux_window_mask][sort_order]

    return T2WindowData(
        current_raw=take(current_raw),
        values=take(corr_values),
        raw_fluxs=take(np.asarray(raw_fluxs, dtype=np.float64)),
        fluxs=take(corr_fluxs),
        f01_mhz=take(f01_mhz),
        T2e_us=take(T2e_us),
        T2e_err_us=take(T2e_err_us),
        T1_us=take(T1_us),
        T1_err_us=take(T1_err_us),
        f01_correction_accepted=correction_accepted[flux_window_mask][sort_order],
        flux_corrections=take(flux_corrections),
        kept_rows=int(np.count_nonzero(flux_window_mask)),
        source_rows=len(corr_fluxs),
    )


def _derive_dephasing(
    window: T2WindowData,
    *,
    max_rel_t2e_err: float,
    use_weighted_points_only: bool,
) -> tuple[T2CurveData, T2CurveData]:
    sample_mask = (
        np.isfinite(window.fluxs)
        & np.isfinite(window.f01_mhz)
        & np.isfinite(window.T1_us)
        & np.isfinite(window.T2e_us)
        & (window.T1_us > 0.0)
        & (window.T2e_us > 0.0)
    )
    gamma_phi_all = 1.0 / window.T2e_us - 1.0 / (2.0 * window.T1_us)
    t1_err_valid = np.isnan(window.T1_err_us) | (
        np.isfinite(window.T1_err_us) & (window.T1_err_us >= 0.0)
    )
    t2_err_valid = np.isnan(window.T2e_err_us) | (
        np.isfinite(window.T2e_err_us)
        & (window.T2e_err_us > 0.0)
        & (window.T2e_err_us < max_rel_t2e_err * window.T2e_us)
    )
    if use_weighted_points_only:
        error_mask = np.isfinite(window.T1_err_us) & np.isfinite(window.T2e_err_us)
        error_mask &= t1_err_valid & t2_err_valid
    else:
        error_mask = t1_err_valid & t2_err_valid
    fit_mask = sample_mask & error_mask & (gamma_phi_all > 0.0)

    sample = _make_curve_data(window, sample_mask, include_gamma_err=False)
    fit = _make_curve_data(window, fit_mask, include_gamma_err=True)
    return sample, fit


def _make_curve_data(
    window: T2WindowData, mask: NDArray[np.bool_], *, include_gamma_err: bool
) -> T2CurveData:
    values = window.values[mask]
    fluxs = window.fluxs[mask]
    f01_mhz = window.f01_mhz[mask]
    T1_us = window.T1_us[mask]
    T1_err_us = window.T1_err_us[mask]
    T2e_us = window.T2e_us[mask]
    T2e_err_us = window.T2e_err_us[mask]
    gamma_phi = 1.0 / T2e_us - 1.0 / (2.0 * T1_us)
    gamma_phi_err = None
    if include_gamma_err:
        gamma_phi_err = np.sqrt(
            (T2e_err_us / T2e_us**2) ** 2 + (0.5 * T1_err_us / T1_us**2) ** 2
        )
    Tphi_us = np.where(gamma_phi > 0.0, 1.0 / gamma_phi, np.nan)
    return T2CurveData(
        values=values,
        fluxs=fluxs,
        f01_mhz=f01_mhz,
        T1_us=T1_us,
        T1_err_us=T1_err_us,
        T2e_us=T2e_us,
        T2e_err_us=T2e_err_us,
        gamma_phi_per_us=gamma_phi,
        gamma_phi_err_per_us=gamma_phi_err,
        Tphi_us=Tphi_us,
    )


def _branch_coverage_table(
    t2e_df: pd.DataFrame,
    t2r_df: pd.DataFrame,
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    current_scale: float,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
) -> pd.DataFrame:
    rows = [
        _coverage_row(
            "T2e rows",
            t2e_df,
            params=params,
            flux_half=flux_half,
            flux_period=flux_period,
            current_scale=current_scale,
            analysis_flux_range=analysis_flux_range,
            max_abs_flux_correction=max_abs_flux_correction,
        ),
        _coverage_row(
            "T2r rows",
            t2r_df,
            params=params,
            flux_half=flux_half,
            flux_period=flux_period,
            current_scale=current_scale,
            analysis_flux_range=analysis_flux_range,
            max_abs_flux_correction=max_abs_flux_correction,
        ),
    ]
    return pd.DataFrame(rows)


def _coverage_row(
    label: str,
    frame: pd.DataFrame,
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    current_scale: float,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
) -> dict[str, object]:
    if frame.empty:
        return _empty_coverage_row(label)

    finite = np.isfinite(_float_column(frame, "calibrated mA")) & np.isfinite(
        _float_column(frame, "Freq (MHz)")
    )
    usable = cast(pd.DataFrame, frame.loc[finite].copy())
    if usable.empty:
        return _empty_coverage_row(label)

    raw_values = _float_column(usable, "calibrated mA") * current_scale
    f01_freqs = 1e-3 * _float_column(usable, "Freq (MHz)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        correction = correct_flux_from_f01(
            raw_values,
            f01_freqs,
            params,
            flux_half,
            flux_period,
            max_abs_flux_correction=max_abs_flux_correction,
        )
    corrected_values = correction.corrected_dev_values
    corrected_fluxs = correction.corrected_fluxs
    in_window = (
        np.isfinite(corrected_fluxs)
        & (corrected_fluxs >= analysis_flux_range[0])
        & (corrected_fluxs <= analysis_flux_range[1])
    )
    window_values = corrected_values[in_window]
    return {
        "subset": label,
        "n_with_f01": len(usable),
        "n_in_flux_window": int(np.count_nonzero(in_window)),
        "accepted": int(np.count_nonzero(correction.accepted)),
        "skipped": correction.skipped_count,
        "raw_min_mA": float(np.nanmin(raw_values)),
        "raw_max_mA": float(np.nanmax(raw_values)),
        "window_corr_min_mA": (
            np.nan if len(window_values) == 0 else float(np.nanmin(window_values))
        ),
        "window_corr_max_mA": (
            np.nan if len(window_values) == 0 else float(np.nanmax(window_values))
        ),
        "window_below_flux_half": int(np.count_nonzero(window_values < flux_half)),
        "window_at_flux_half": int(
            np.count_nonzero(np.isclose(window_values, flux_half))
        ),
        "window_above_flux_half": int(np.count_nonzero(window_values > flux_half)),
    }


def _empty_coverage_row(label: str) -> dict[str, object]:
    return {
        "subset": label,
        "n_with_f01": 0,
        "n_in_flux_window": 0,
        "accepted": 0,
        "skipped": 0,
        "raw_min_mA": np.nan,
        "raw_max_mA": np.nan,
        "window_corr_min_mA": np.nan,
        "window_corr_max_mA": np.nan,
        "window_below_flux_half": 0,
        "window_at_flux_half": 0,
        "window_above_flux_half": 0,
    }


def _half_preview_table(
    t2e_df: pd.DataFrame,
    *,
    params: tuple[float, float, float],
    flux_half: float,
    flux_period: float,
    current_scale: float,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
) -> pd.DataFrame:
    finite = np.isfinite(_float_column(t2e_df, "calibrated mA")) & np.isfinite(
        _float_column(t2e_df, "Freq (MHz)")
    )
    frame = cast(pd.DataFrame, t2e_df.loc[finite].copy())
    if frame.empty:
        return frame
    raw_values = _float_column(frame, "calibrated mA") * current_scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        correction = correct_flux_from_f01(
            raw_values,
            1e-3 * _float_column(frame, "Freq (MHz)"),
            params,
            flux_half,
            flux_period,
            max_abs_flux_correction=max_abs_flux_correction,
        )
    preview_columns = [
        column
        for column in [
            "calibrated mA",
            "Freq (MHz)",
            "T2r (us)",
            "T2e (us)",
            "T2e err (us)",
        ]
        if column in frame.columns
    ]
    preview = cast(pd.DataFrame, frame.loc[:, preview_columns].copy())
    preview["raw current (mA)"] = raw_values
    preview["f01-corrected current (mA)"] = correction.corrected_dev_values
    preview["f01-corrected flux"] = correction.corrected_fluxs
    preview["f01 correction accepted"] = correction.accepted
    in_window = (
        (_float_column(preview, "f01-corrected flux") >= analysis_flux_range[0])
        & (_float_column(preview, "f01-corrected flux") <= analysis_flux_range[1])
        & np.isclose(_float_column(preview, "f01-corrected current (mA)"), flux_half)
    )
    return cast(pd.DataFrame, preview.loc[in_window].copy())


def _thermal_estimate(
    sample: T2CurveData,
    fit: T2CurveData,
    *,
    params: tuple[float, float, float],
    bare_rf: float,
    g: float,
    kappa_over_2pi_mhz: float,
) -> T2CurveThermalEstimate:
    half_sample_idx = int(np.nanargmin(np.abs(sample.fluxs - 0.5)))
    half_flux = float(sample.fluxs[half_sample_idx])
    half_T1_us = float(sample.T1_us[half_sample_idx])
    half_T2e_us = float(sample.T2e_us[half_sample_idx])
    half_gamma_phi_per_us = float(sample.gamma_phi_per_us[half_sample_idx])
    half_chi_over_2pi_mhz = float(
        dispersive_chi01_over_2pi_mhz(
            params, np.asarray([half_flux], dtype=np.float64), bare_rf, g
        )[0]
    )
    half_gamma_per_photon_us = float(
        thermal_photon_gamma_phi_per_us(
            1.0,
            kappa_over_2pi_mhz=kappa_over_2pi_mhz,
            chi_over_2pi_mhz=half_chi_over_2pi_mhz,
        )
    )
    half_equivalent_n_th = equivalent_n_th_from_t2(
        T1_us=half_T1_us,
        T2_us=half_T2e_us,
        kappa_over_2pi_mhz=kappa_over_2pi_mhz,
        chi_over_2pi_mhz=half_chi_over_2pi_mhz,
    )
    fit_peak_idx = int(np.nanargmax(fit.T2e_us))
    fit_peak_flux = float(fit.fluxs[fit_peak_idx])
    fit_peak_chi_over_2pi_mhz = float(
        dispersive_chi01_over_2pi_mhz(
            params, np.asarray([fit_peak_flux], dtype=np.float64), bare_rf, g
        )[0]
    )
    fit_peak_equivalent_n_th = equivalent_n_th_from_t2(
        T1_us=float(fit.T1_us[fit_peak_idx]),
        T2_us=float(fit.T2e_us[fit_peak_idx]),
        kappa_over_2pi_mhz=kappa_over_2pi_mhz,
        chi_over_2pi_mhz=fit_peak_chi_over_2pi_mhz,
    )
    return T2CurveThermalEstimate(
        half_flux=half_flux,
        half_T1_us=half_T1_us,
        half_T2e_us=half_T2e_us,
        half_gamma_phi_per_us=half_gamma_phi_per_us,
        half_chi_over_2pi_mhz=half_chi_over_2pi_mhz,
        half_gamma_per_photon_us=half_gamma_per_photon_us,
        half_equivalent_n_th=half_equivalent_n_th,
        fit_peak_equivalent_n_th=fit_peak_equivalent_n_th,
        fit_peak_flux=fit_peak_flux,
    )


def _combined_fit_arrays(data: T2DephasingAnalysis) -> _CombinedFitArrays:
    fit = data.fit
    return _CombinedFitArrays(
        fluxs=fit.fluxs,
        T1_us=fit.T1_us,
        T1_err_us=fit.T1_err_us,
        T2e_us=fit.T2e_us,
        T2e_err_us=fit.T2e_err_us,
    )


def _fit_model_axes(
    data: T2DephasingAnalysis,
    fluxs: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    context = data.calibration.context
    resolved_fluxs = (
        data.fit.fluxs if fluxs is None else np.asarray(fluxs, dtype=np.float64)
    )
    return (
        predict_domega_dflux(context.params, resolved_fluxs),
        dispersive_chi01_over_2pi_mhz(
            context.params, resolved_fluxs, context.bare_rf, context.g
        ),
    )


def _photon_gamma_like(
    n_th: float,
    *,
    kappa_over_2pi_mhz: float,
    chi_over_2pi_mhz: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.asarray(
        thermal_photon_gamma_phi_per_us(
            n_th,
            kappa_over_2pi_mhz=kappa_over_2pi_mhz,
            chi_over_2pi_mhz=chi_over_2pi_mhz,
        ),
        dtype=np.float64,
    )


def _safe_divide_positive(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    *,
    min_denominator: float = 0.0,
) -> NDArray[np.float64]:
    if not np.isfinite(min_denominator) or min_denominator < 0.0:
        raise ValueError("min_denominator must be finite and non-negative")
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=(denominator > min_denominator) & np.isfinite(denominator),
    )
    return np.where((ratio > 0.0) & np.isfinite(ratio), ratio, np.nan)


def _relative_floor(values: NDArray[np.float64], fraction: float) -> float:
    if not np.isfinite(fraction) or fraction < 0.0:
        raise ValueError("fraction must be finite and non-negative")
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return 0.0
    return float(fraction * np.nanmax(finite))


def _positive_statistic(
    values: NDArray[np.float64],
    fluxs: NDArray[np.float64],
    statistic: ProbeStatistic,
) -> float:
    if statistic == "half_flux":
        order = np.argsort(np.abs(fluxs - 0.5))
        for idx in order:
            value = float(values[idx])
            if np.isfinite(value) and value > 0.0:
                return value
        statistic = "median"

    finite = _finite_positive(values)
    if statistic == "min":
        return float(np.nanmin(finite))
    if statistic == "median":
        return float(np.nanmedian(finite))
    if statistic == "mean":
        return float(np.nanmean(finite))
    if statistic == "p90":
        return float(np.nanpercentile(finite, 90.0))
    if statistic == "max":
        return float(np.nanmax(finite))
    raise ValueError(f"unknown probe statistic: {statistic}")


def _finite_positive(values: NDArray[np.float64]) -> NDArray[np.float64]:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        raise ValueError("no positive finite pointwise probe values")
    return finite


def _auto_bounds(
    value: float,
    *,
    lower_floor: float = 1e-8,
    upper_cap: float = np.inf,
    factor: float = 100.0,
) -> tuple[float, float]:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("fit parameter initial value must be positive and finite")
    return max(lower_floor, value / factor), min(upper_cap, value * factor)


def _required_param(value: float | None, name: str) -> float:
    if value is None:
        raise RuntimeError(f"{name} was expected to be active in the T2 fit")
    return float(value)


def _optional_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _fit_init_value(
    override: float | None,
    probe_value: float | None,
    name: str,
) -> float:
    value = override if override is not None else probe_value
    if value is None:
        raise ValueError(
            f"{name} needs an explicit value or a completed mechanism probe"
        )
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} initial value must be positive and finite")
    return float(value)


def _validate_mechanisms(active_mechanisms: tuple[MechanismName, ...]) -> None:
    if not active_mechanisms:
        raise ValueError("at least one T2 mechanism must be active")
    if len(set(active_mechanisms)) != len(active_mechanisms):
        raise ValueError("active_mechanisms contains duplicates")
    known = {"flux_noise", "photon_shot_noise"}
    unknown = set(active_mechanisms) - known
    if unknown:
        raise ValueError(f"unknown active mechanism(s): {sorted(unknown)}")


def _fit_params_table(init: T2FitParams, fit_result: T2FitResult) -> pd.DataFrame:
    rows = []
    if fit_result.params.A_phi is not None:
        rows.append(
            {
                "parameter": "A_phi",
                "init": None if init.A_phi is None else init.A_phi,
                "fit": fit_result.params.A_phi,
                "stderr": fit_result.stderr.A_phi,
                "display": f"{fit_result.params.A_phi * 1e6:.3f} uPhi0/sqrtHz",
            }
        )
    if fit_result.params.n_th is not None:
        rows.append(
            {
                "parameter": "n_th",
                "init": None if init.n_th is None else init.n_th,
                "fit": fit_result.params.n_th,
                "stderr": fit_result.stderr.n_th,
                "display": f"{fit_result.params.n_th:.3e}",
            }
        )
    return pd.DataFrame(rows)


def _error_fill_summary(result: ErrorResolutionResult | None) -> str:
    if result is None:
        return "no error column"
    filled = int(
        np.count_nonzero(
            result.bin_fill_mask | result.global_fill_mask | result.fallback_fill_mask
        )
    )
    if filled == 0:
        return "0"
    bin_filled = int(np.count_nonzero(result.bin_fill_mask))
    global_filled = int(np.count_nonzero(result.global_fill_mask))
    fallback_filled = int(np.count_nonzero(result.fallback_fill_mask))
    return f"{filled} (bin={bin_filled}, global={global_filled}, fallback={fallback_filled})"


def _dephasing_summary_table(
    *,
    calibration: T2FluxCalibration,
    window: T2WindowData,
    sample: T2CurveData,
    fit: T2CurveData,
    analysis_flux_range: tuple[float, float],
    max_rel_t2e_err: float,
    use_weighted_points_only: bool,
) -> pd.DataFrame:
    sample_peak_idx = int(np.nanargmax(sample.T2e_us))
    fit_peak_idx = int(np.nanargmax(fit.T2e_us))
    return pd.DataFrame(
        [
            (
                "analysis flux range",
                f"{analysis_flux_range[0]:.3f}..{analysis_flux_range[1]:.3f}",
            ),
            ("max relative T2e err", f"{max_rel_t2e_err:.3g}"),
            ("use weighted points only", str(use_weighted_points_only)),
            ("window T2e rows", f"{window.kept_rows}/{window.source_rows}"),
            ("sample rows", str(len(sample.T2e_us))),
            ("fit rows", str(len(fit.T2e_us))),
            ("current scale", f"{calibration.current_scale:g}"),
            (
                "sample peak",
                f"flux={sample.fluxs[sample_peak_idx]:.6f}, T2e={sample.T2e_us[sample_peak_idx]:.3f} us",
            ),
            (
                "fit peak",
                f"flux={fit.fluxs[fit_peak_idx]:.6f}, T2e={fit.T2e_us[fit_peak_idx]:.3f} us",
            ),
            (
                "sample 2T1/T2e median",
                f"{np.nanmedian(2.0 * sample.T1_us / sample.T2e_us):.3g}",
            ),
        ],
        columns=["metric", "value"],
    )


def _stage_table(stage: str, table: pd.DataFrame) -> pd.DataFrame:
    stage_table = table.copy()
    stage_table.insert(0, "stage", stage)
    return stage_table


def _save_standard_figures(
    context: T2CurveContext,
    data: T2DephasingAnalysis,
    flux_probe: T2FluxNoiseProbe,
    photon_probe: T2PhotonShotNoiseProbe,
    channel_analysis: T2ChannelAnalysis,
    *,
    save_figures: bool,
    show_figures: bool,
    thermal_n_th_range: tuple[float, float],
    thermal_n_th_count: int,
) -> dict[str, str]:
    figures = {
        "flux_calibration": (
            plot_t2_flux_calibration(data)[0],
            "flux_calibration.png",
            None,
        ),
        "t2e_vs_flux": (plot_t2_dephasing_data(data)[0], "T2e_vs_flux.png", None),
        "flux_noise_probe": (
            plot_flux_noise_probe(flux_probe)[0],
            "Gamma_phi_vs_flux_sensitivity.png",
            None,
        ),
        "photon_shot_noise_probe": (
            plot_photon_shot_noise_probe(
                photon_probe,
                n_th_range=thermal_n_th_range,
                n_th_count=thermal_n_th_count,
            )[0],
            "T2e_thermal_photon_limit.png",
            None,
        ),
        "n_th_vs_flux": (
            plot_nth_limit_vs_flux(photon_probe)[0],
            "n_th_vs_flux.png",
            None,
        ),
        "channel_overlay": (
            plot_t2_channel_analysis(channel_analysis)[0],
            "T2e_flux_noise_fit.png",
            "tight",
        ),
    }
    paths: dict[str, str] = {}
    for name, (fig, filename, bbox_inches) in figures.items():
        path = os.path.join(context.image_dir, filename)
        if save_figures:
            fig.savefig(path, dpi=160, bbox_inches=bbox_inches)
            paths[name] = path
        if show_figures:
            plt.show()
        plt.close(fig)
    return paths


def _emit_outputs(
    result: T2CurveAnalysisResult,
    *,
    display_fn: DisplayFn | None,
    verbose: bool,
    tables: bool,
) -> None:
    if verbose:
        print("params =", result.context.params, "GHz")
        print("flux_half =", result.context.flux_half)
        print("flux_period =", result.context.flux_period)
        print("bare_rf =", result.context.bare_rf, "GHz")
        print("g =", result.context.g, "GHz")
        print(
            "readout kappa/2pi =",
            result.combined_fit.readout_kappa_over_2pi_mhz,
            "MHz",
        )
        print("fit success =", result.combined_fit.fit_result.success)
        print("fit message =", result.combined_fit.fit_result.message)
        print("fixed =", result.combined_fit.fit_result.fixed)
        print("free =", result.combined_fit.fit_result.free)
        print(f"Images saved under: {result.context.image_dir}")

    if tables and display_fn is not None:
        display_fn(result.calibration.scale_report)
        display_fn(result.data.branch_coverage)
        if not result.data.half_preview.empty:
            display_fn(result.data.half_preview)
        display_fn(result.flux_probe.summary_table)
        display_fn(result.photon_probe.summary_table)
        display_fn(result.photon_probe.thermal_limit_table)
        display_fn(result.combined_fit.params_table)
        display_fn(result.summary_table)

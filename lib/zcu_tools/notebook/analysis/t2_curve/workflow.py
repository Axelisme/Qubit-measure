"""Probe, fit and orchestrate the staged T2 curve analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fit_tools import (
    ErrorResolutionResult,
    FluxResidualWeighting,
    MeasurementErrorPolicy,
)

from .base import (
    calculate_t2_channel_curves,
    dispersive_chi01_over_2pi_mhz,
    make_thermal_limit_table,
    predict_domega_dflux,
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
from .models import DEFAULT_THERMAL_PROBE_N_TH
from .models import (
    DisplayFn as DisplayFn,
)
from .models import (
    MechanismName as MechanismName,
)
from .models import (
    MechanismOrParamName as MechanismOrParamName,
)
from .models import (
    ProbeStatistic as ProbeStatistic,
)
from .models import (
    T2ChannelAnalysis as T2ChannelAnalysis,
)
from .models import (
    T2CombinedFit as T2CombinedFit,
)
from .models import (
    T2CurveAnalysisConfig as T2CurveAnalysisConfig,
)
from .models import (
    T2CurveAnalysisResult as T2CurveAnalysisResult,
)
from .models import (
    T2CurveContext as T2CurveContext,
)
from .models import (
    T2CurveData as T2CurveData,
)
from .models import (
    T2CurveThermalEstimate as T2CurveThermalEstimate,
)
from .models import (
    T2DephasingAnalysis as T2DephasingAnalysis,
)
from .models import (
    T2FluxCalibration as T2FluxCalibration,
)
from .models import (
    T2FluxNoiseProbe as T2FluxNoiseProbe,
)
from .models import (
    T2PhotonShotNoiseProbe as T2PhotonShotNoiseProbe,
)
from .models import (
    T2RowDiagnostics as T2RowDiagnostics,
)
from .models import (
    T2WindowData as T2WindowData,
)
from .preparation import (
    calibrate_t2_flux as calibrate_t2_flux,
)
from .preparation import (
    load_t2_curve_context as load_t2_curve_context,
)
from .preparation import (
    prepare_t2_dephasing_data as prepare_t2_dephasing_data,
)
from .presentation import emit_t2_curve_outputs, save_t2_standard_figures
from .presentation import (
    plot_flux_noise_probe as plot_flux_noise_probe,
)
from .presentation import (
    plot_nth_limit_vs_flux as plot_nth_limit_vs_flux,
)
from .presentation import (
    plot_photon_shot_noise_probe as plot_photon_shot_noise_probe,
)
from .presentation import (
    plot_t2_channel_analysis as plot_t2_channel_analysis,
)
from .presentation import (
    plot_t2_dephasing_data as plot_t2_dephasing_data,
)
from .presentation import (
    plot_t2_flux_calibration as plot_t2_flux_calibration,
)
from .presentation import (
    save_t2_curve_figure as save_t2_curve_figure,
)


@dataclass(frozen=True, slots=True)
class _CombinedFitArrays:
    fluxs: NDArray[np.float64]
    T1_us: NDArray[np.float64]
    T1_err_us: NDArray[np.float64]
    T2e_us: NDArray[np.float64]
    T2e_err_us: NDArray[np.float64]


def analyze_flux_noise_limit(  # noqa: PLR0913
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


# tolerated: Notebook documents photon probe controls and shared fit-policy keywords.
def analyze_photon_shot_noise_limit(  # noqa: PLR0913
    data: T2DephasingAnalysis,
    *,
    readout_kappa_over_2pi_mhz: float,
    assumed_A_phi: float = 0.0,
    n_th_init: float | None = None,
    n_th_bounds: tuple[float, float] | None = None,
    statistic: ProbeStatistic = "min",
    thermal_probe_n_th: tuple[float, ...] = DEFAULT_THERMAL_PROBE_N_TH,
    T1_error_policy: MeasurementErrorPolicy | None = None,
    T2_error_policy: MeasurementErrorPolicy | None = None,
    flux_weighting: FluxResidualWeighting | None = None,
    residual_mode: ResidualMode = "gamma",
    loss: str = "linear",
    max_nfev: int | None = 10000,
    progress: bool = False,
) -> T2PhotonShotNoiseProbe | None:
    """Probe photon noise using us lifetimes and MHz readout linewidth.

    Return None when data lacks dispersive inputs, disabling this channel.
    Otherwise infer n_th with the chosen statistic, bounds and residual/error
    policies. Invalid fit inputs or no positive photon estimate raise ValueError.
    """
    if not data.calibration.context.photon_shot_noise_enabled:
        return None
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
    thermal = _thermal_estimate(
        data,
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


# tolerated: Notebook/tests require explicit init/fixed and shared error/weight/solver keywords.
def fit_t2_curve(  # noqa: PLR0913
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
    """Fit echo dephasing on prepared fit-window samples without writing files.

    data comes from prepare_t2_dephasing_data. Only data.fit rows are fitted.
    Its flux, T1_us, T2e_us and error arrays must share a nonempty 1D shape.
    Flux is normalized Phi0, lifetimes/errors are us, context energies are GHz.
    Model sensitivity uses rad/us/Phi0; dispersive chi/2pi uses MHz.
    readout_kappa_over_2pi_mhz is the resonator linewidth kappa/2pi in MHz.

    init enables non-None A_phi (Phi0/sqrt(Hz)) and n_th (mean photon number).
    Active values must be positive finite. bounds maps active names to
    (lower, upper) in those units; defaults are positive and unbounded above.
    fixed holds named active parameters at init values. Without dispersive
    inputs, the photon channel is disabled, n_th is removed from init, fixed
    and bounds, and returned n_th is None. At least one parameter must remain.

    residual_mode='gamma' fits 1/T2e - 1/(2*T1) in 1/us; 'log_t2' fits log T2e.
    T1_error_policy/T2_error_policy resolve effective errors, including NaN.
    flux_weighting controls residual weights over fit fluxs. None selects
    the fit layer's default policies. loss is a SciPy least_squares loss name.
    max_nfev limits evaluations (None uses SciPy's default); progress displays
    residual-evaluation progress.

    Return T2CombinedFit with data, resolved settings, samples, model axes,
    summary/parameter tables and fit_result (parameters, stderr, models,
    residuals, covariance and solver status). Nonconvergence sets success=False.
    Raise ValueError for empty/misaligned/nonfinite inputs, nonpositive
    lifetimes or observed dephasing, invalid errors/policies, parameter names,
    bounds/init or residual/solver options. Physics model errors propagate.
    """
    if not data.calibration.context.photon_shot_noise_enabled:
        init = replace(init, n_th=None)
        fixed = tuple(name for name in fixed if name != "n_th")
        bounds = (
            None
            if bounds is None
            else {name: value for name, value in bounds.items() if name != "n_th"}
        )
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
    domega_dflux, chi_over_2pi_mhz = _fit_model_axes(data, grid_fluxs)
    curves = calculate_t2_channel_curves(
        grid_fluxs,
        params=context.params,
        fit_result=combined_fit.fit_result,
        kappa_over_2pi_mhz=combined_fit.readout_kappa_over_2pi_mhz,
        chi_over_2pi_mhz=chi_over_2pi_mhz,
        domega_dflux=domega_dflux,
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


# tolerated: Notebook explicitly collects each stage and figure paths as separate keywords.
def collect_t2_curve_result(  # noqa: PLR0913
    *,
    context: T2CurveContext,
    calibration: T2FluxCalibration,
    data: T2DephasingAnalysis,
    flux_probe: T2FluxNoiseProbe,
    photon_probe: T2PhotonShotNoiseProbe | None,
    combined_fit: T2CombinedFit,
    channel_analysis: T2ChannelAnalysis,
    figure_paths: Mapping[str, str] | None = None,
) -> T2CurveAnalysisResult:
    summary_table = pd.concat(
        [
            _stage_table("load", calibration.summary_table),
            _stage_table("dephasing", data.summary_table),
            _stage_table("flux_probe", flux_probe.summary_table),
            _stage_table("photon_probe", photon_probe.summary_table)
            if photon_probe is not None
            else _stage_table(
                "photon_probe",
                pd.DataFrame(
                    [("photon shot noise", "disabled: missing dispersive")],
                    columns=["metric", "value"],
                ),
            ),
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


def run_t2_curve_analysis(
    config: T2CurveAnalysisConfig,
    *,
    display_fn: DisplayFn | None = None,
) -> T2CurveAnalysisResult:
    context = load_t2_curve_context(
        result_dir=config.result_dir,
        fluxs=config.fluxs,
        samples_filename=config.samples_filename,
        image_dir=config.image_dir,
    )
    calibration = calibrate_t2_flux(context)
    data = prepare_t2_dephasing_data(
        calibration,
        analysis_flux_range=config.analysis_flux_range,
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
        active_mechanisms=tuple(
            mechanism
            for mechanism in config.active_mechanisms
            if mechanism != "photon_shot_noise" or context.photon_shot_noise_enabled
        ),
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
        figure_paths = save_t2_standard_figures(
            flux_probe,
            photon_probe,
            channel_analysis,
            config=config,
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
    emit_t2_curve_outputs(
        result,
        display_fn=display_fn,
        verbose=config.verbose,
        tables=config.display_tables,
    )
    return result


def _thermal_estimate(
    data: T2DephasingAnalysis,
    *,
    kappa_over_2pi_mhz: float,
) -> T2CurveThermalEstimate:
    sample, fit = data.sample, data.fit
    context = data.calibration.context
    params, bare_rf, g = context.params, context.bare_rf, context.g
    if bare_rf is None or g is None:
        raise ValueError("photon shot noise requires dispersive bare_rf and g")
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
        )
        if context.bare_rf is not None and context.g is not None
        else np.zeros_like(resolved_fluxs),
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


def _stage_table(stage: str, table: pd.DataFrame) -> pd.DataFrame:
    stage_table = table.copy()
    stage_table.insert(0, "stage", stage)
    return stage_table

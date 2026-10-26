"""Combined T1 fitting and end-to-end orchestration with direct stage re-exports."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fit_tools import (
    ErrorResolutionResult,
    FluxResidualWeighting,
    MeasurementErrorPolicy,
)
from zcu_tools.resources.qubit_params import (
    QubitParams,
    T1CurveFit,
    T1CurveFitParams,
    T1CurveFitUncertainty,
)
from zcu_tools.simulate.fluxonium import (
    calculate_eff_t1_vs_flux_fast,
)

from .base import (
    DEFAULT_TEMP_LOWER_BOUND,
    DEFAULT_TEMP_UPPER_BOUND,
    TempBounds,
    format_parameter,
    positive_override,
    resolve_Temp_bounds,
)
from .fit import (
    FitBounds,
    NoiseParameterName,
    ParameterName,
    ResidualMode,
    T1FitParams,
    T1FitResult,
    fit_t1_noise_params,
)
from .models import (
    CURVE_LABELS,
    NOISE_CHANNELS,
    PARAM_TO_MECHANISM,
    MechanismName,
    MechanismOrParamName,
    ProbeStatistic,
    PurcellCorrectionResult,
    PurcellEffectParams,
    T1ChannelAnalysis,
    T1ChannelCurves,
    T1CombinedFit,
    T1CurveAnalysisConfig,
    T1CurveAnalysisResult,
    T1CurveContext,
    T1CurveData,
    T1FluxCalibration,
    T1MechanismOptions,
    T1MechanismProbe,
    T1PreparedData,
    validate_t1_mechanisms,
)
from .preparation import (
    calibrate_t1_flux,
    load_t1_curve_context,
    prepare_t1_curve_data,
)
from .presentation import (
    plot_purcell_Temp_upper_bound,
    plot_t1_channel_analysis,
    plot_t1_curve_data,
    plot_t1_flux_calibration,
    plot_t1_mechanism_dipole,
    plot_t1_mechanism_limit,
    plot_t1_mechanism_probe,
    save_t1_curve_figure,
    t1_parameter_text,
)
from .probes import (
    analyze_t1_capacitive_limit,
    analyze_t1_inductive_limit,
    analyze_t1_mechanism_limit,
    analyze_t1_quasiparticle_limit,
)
from .purcell import (
    calculate_purcell_t1_limit,
    clear_t1_purcell_cache,
    combine_t1_limits,
    estimate_purcell_Temp_upper_bound,
    load_t1_purcell_params,
    relaxation_rates_from_t1,
    subtract_relaxation_limit,
)


# tolerated: Notebook keywords select active channels, probe fallbacks and explicit value overrides.
def make_t1_fit_init(  # noqa: PLR0913
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
    validate_t1_mechanisms(active_mechanisms)
    probe_by_param = {
        "Q_cap": cap_probe,
        "x_qp": qp_probe,
        "Q_ind": ind_probe,
    }
    return T1FitParams(
        Temp=positive_override(None, Temp, "Temp"),
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


# tolerated: Notebook keywords control Q and x_qp limits separately from temperature and span.
def make_t1_fit_bounds(  # noqa: PLR0913
    init: T1FitParams,
    *,
    factor: float = 100.0,
    Temp_bounds: TempBounds = (DEFAULT_TEMP_LOWER_BOUND, DEFAULT_TEMP_UPPER_BOUND),
    Q_lower_floor: float = 1.0,
    x_qp_lower_floor: float = 1e-12,
    Q_upper_cap: float = np.inf,
    x_qp_upper_cap: float = 1.0,
) -> dict[str, tuple[float, float]]:
    if factor <= 1.0:
        raise ValueError("factor must be larger than 1")
    bounds: dict[str, tuple[float, float]] = {"Temp": resolve_Temp_bounds(Temp_bounds)}
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


def _normalize_t1_fit_bounds(
    bounds: FitBounds | None,
) -> dict[str, tuple[float, float]] | None:
    if bounds is None:
        return None
    normalized: dict[str, tuple[float, float]] = {}
    for name, (lower, upper) in bounds.items():
        if lower is None:
            if name != "Temp":
                raise ValueError("only Temp lower bound can be None")
            normalized[name] = resolve_Temp_bounds((None, upper))
        else:
            normalized[name] = (float(lower), float(upper))
    return normalized


# tolerated: Notebook fit keywords keep init/fixed/bounds, error weighting and optimizer controls.
def fit_t1_curve(  # noqa: PLR0913
    data: T1PreparedData,
    *,
    init: T1FitParams,
    purcell: PurcellEffectParams | None = None,
    bounds: FitBounds | None = None,
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
    fit_bounds = _normalize_t1_fit_bounds(bounds)
    fit_result = fit_t1_noise_params(
        data.fit.fluxs,
        data.fit.T1_ns,
        context.params,
        init=init,
        bounds=fit_bounds,
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
        bounds=fit_bounds,
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
        CURVE_LABELS[PARAM_TO_MECHANISM[param_name]]: calculate_eff_t1_vs_flux_fast(
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
        display_T1s = combine_t1_limits(mechanism_T1s, purcell_T1s)

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


# tolerated: Notebook callers explicitly collect stage records and saved paths by keyword.
def collect_t1_curve_result(  # noqa: PLR0913
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


def run_t1_curve_analysis(
    config: T1CurveAnalysisConfig,
) -> T1CurveAnalysisResult:
    context = load_t1_curve_context(
        result_dir=config.result_dir,
        fluxs=config.fluxs,
        samples_filename=config.samples_filename,
        image_dir=config.image_dir,
    )
    calibration = calibrate_t1_flux(context)
    data = prepare_t1_curve_data(
        calibration,
        analysis_flux_range=config.analysis_flux_range,
        max_rel_t1_err=config.max_rel_t1_err,
        use_weighted_points_only=config.use_weighted_points_only,
    )
    cap_probe = (
        analyze_t1_capacitive_limit(
            data,
            Temp=config.Temp,
            purcell=config.purcell,
            Temp_bounds=config.probe_Temp_bounds,
        )
        if "capacitive" in config.active_mechanisms
        else None
    )
    qp_probe = (
        analyze_t1_quasiparticle_limit(
            data,
            Temp=config.Temp,
            purcell=config.purcell,
            Temp_bounds=config.probe_Temp_bounds,
        )
        if "quasiparticle" in config.active_mechanisms
        else None
    )
    ind_probe = (
        analyze_t1_inductive_limit(
            data,
            Temp=config.Temp,
            purcell=config.purcell,
            Temp_bounds=config.probe_Temp_bounds,
        )
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
        return relaxation_rates_from_t1(
            purcell_T1s,
            name="Purcell T1",
        )

    return extra_rate


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
    return positive_override(None, value, name)


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
        channel_name, option_name = NOISE_CHANNELS["Q_cap"]
        values.append(("Q_cap", channel_name, option_name, params.Q_cap))
    if params.x_qp is not None:
        channel_name, option_name = NOISE_CHANNELS["x_qp"]
        values.append(("x_qp", channel_name, option_name, params.x_qp))
    if params.Q_ind is not None:
        channel_name, option_name = NOISE_CHANNELS["Q_ind"]
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
                "display": format_parameter(name, fit_value),
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


def _error_fill_summary(result: ErrorResolutionResult | None) -> str:
    if result is None:
        return "no error column"
    bin_fill_mask = result.bin_fill_mask
    global_fill_mask = result.global_fill_mask
    fallback_fill_mask = result.fallback_fill_mask
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


def _stage_table(stage: str, table: pd.DataFrame) -> pd.DataFrame:
    stage_table = table.copy()
    stage_table.insert(0, "stage", stage)
    return stage_table


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
    "T1MechanismOptions",
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
    "estimate_purcell_Temp_upper_bound",
    "fit_t1_curve",
    "load_t1_curve_context",
    "load_t1_purcell_params",
    "make_t1_fit_bounds",
    "make_t1_fit_init",
    "mechanisms_to_fixed_params",
    "plot_t1_channel_analysis",
    "plot_t1_curve_data",
    "plot_t1_flux_calibration",
    "plot_t1_mechanism_dipole",
    "plot_t1_mechanism_limit",
    "plot_t1_mechanism_probe",
    "plot_purcell_Temp_upper_bound",
    "prepare_t1_curve_data",
    "run_t1_curve_analysis",
    "save_t1_curve_figure",
    "subtract_relaxation_limit",
    "t1_noise_channels_from_params",
    "t1_parameter_text",
    "write_t1_curve_fit",
]

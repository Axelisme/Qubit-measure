"""Shared T1 stage records and mechanism metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fit_tools import (
    FluxResidualWeighting,
    MeasurementErrorPolicy,
)

from .base import (
    DEFAULT_TEMP_LOWER_BOUND,
    DEFAULT_TEMP_UPPER_BOUND,
    TempBounds,
)
from .fit import (
    FitBounds,
    NoiseParameterName,
    ParameterName,
    ResidualMode,
    T1FitParams,
    T1FitResult,
)

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


MECHANISM_TO_PARAM: dict[MechanismName, NoiseParameterName] = {
    "capacitive": "Q_cap",
    "quasiparticle": "x_qp",
    "inductive": "Q_ind",
}


PARAM_TO_MECHANISM: dict[NoiseParameterName, MechanismName] = {
    value: key for key, value in MECHANISM_TO_PARAM.items()
}


NOISE_CHANNELS: dict[NoiseParameterName, tuple[str, str]] = {
    "Q_cap": ("t1_capacitive", "Q_cap"),
    "x_qp": ("t1_quasiparticle_tunneling", "x_qp"),
    "Q_ind": ("t1_inductive", "Q_ind"),
}


CURVE_LABELS: dict[MechanismName, str] = {
    "capacitive": "capacitive",
    "quasiparticle": "quasiparticle",
    "inductive": "inductive",
}


@dataclass(frozen=True, slots=True)
class PurcellEffectParams:
    """Positive, finite GHz inputs for the two-level Purcell equilibrium model.

    kappa_ghz is the resonator linewidth kappa_rate/(2*pi), not a decay rate;
    0.001 means 1 MHz and the kernel uses 2*pi*kappa_ghz in ns^-1.
    bare_rf is the bare resonator frequency; g is its charge-mode coupling.
    The kernel thermally averages each qubit direction over resonator photons
    and returns equilibrium T1=1/(Gamma_eg+Gamma_ge) for 0↔1 only.
    It excludes leakage to higher qubit states.
    """

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
class T1MechanismOptions:
    """Shared controls for pointwise loss extraction and a single-mechanism probe.

    Temp is the evaluation temperature in K, or the initial temperature when
    fit_temperature=True searches for the least dispersed Q values.
    purcell optionally supplies positive GHz resonator parameters; its rate is
    subtracted from measured relaxation before intrinsic Q is extracted.
    omega_range selects the Q-fit window in rad/ns; either None endpoint is open.
    Temp_bounds limits the temperature search in K; a None lower endpoint uses
    the default positive lower bound.
    fit_constant=True fits a constant Q, while False fits a power law in omega.
    statistic selects median, mean, min, max, p10 or p90 of fitted Q values for
    the suggested parameter value.
    parameter_init overrides that suggestion with a positive finite value:
    dimensionless Q_cap/Q_ind, or the quasiparticle fraction x_qp.
    """

    Temp: float
    purcell: PurcellEffectParams | None = None
    omega_range: tuple[float | None, float | None] = (None, None)
    fit_temperature: bool = False
    Temp_bounds: TempBounds = (DEFAULT_TEMP_LOWER_BOUND, DEFAULT_TEMP_UPPER_BOUND)
    fit_constant: bool = True
    statistic: ProbeStatistic = "median"
    parameter_init: float | None = None


@dataclass(frozen=True, slots=True)
class T1CurveAnalysisConfig:
    """Controls for the fixed Notebook T1 workflow.

    result_dir owns params/samples; image_dir defaults to its t1_curve folder.
    samples_filename selects a CSV. fluxs is a finite 1D normalized Phi0 array
    with one value per CSV row, in row order. CSV coordinates are ignored.
    analysis_flux_range selects an integer-aligned normalized-flux window.
    max_rel_t1_err/use_weighted_points_only select fit rows; T1_error_policy
    resolves ns errors and flux_weighting controls residual weighting.
    Temp/probe_Temp_bounds/fit_Temp_override use K; purcell supplies GHz inputs.
    active_mechanisms chooses channels; fixed_mechanisms fixes active values.
    fit_Q_cap_override/fit_x_qp_override/fit_Q_ind_override are dimensionless.
    fit_bounds uses those same units; residual_mode/loss/max_nfev control fitting.
    t_flux_count selects dense curve size; progress controls fit display.
    save_figures enables plotting/saving; show_figures controls display on save.
    """

    result_dir: str
    fluxs: NDArray[np.float64]
    analysis_flux_range: tuple[float, float] = (0.0, 1.0)
    image_dir: str | None = None
    samples_filename: str = "samples.csv"
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
    probe_Temp_bounds: TempBounds = (
        DEFAULT_TEMP_LOWER_BOUND,
        DEFAULT_TEMP_UPPER_BOUND,
    )
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
    fit_bounds: FitBounds | None = None
    residual_mode: ResidualMode = "log"
    loss: str = "linear"
    max_nfev: int = 10000
    t_flux_count: int = 1000
    progress: bool = True
    save_figures: bool = True
    show_figures: bool = True


@dataclass(frozen=True, slots=True)
class T1CurveContext:
    """Loaded sample and fluxonium inputs shared by Notebook stages.

    result_dir/image_dir/samples_filename locate the inputs and figure outputs.
    params is (EJ, EC, EL) in GHz. No device frame is required.
    fluxs is the author's finite normalized Phi0 array, one per source row.
    samples_df contains source rows; params_table is the parameter display;
    samples_preview contains the requested initial rows and available_columns
    lists the source column names in order.
    """

    result_dir: str
    image_dir: str
    samples_filename: str
    params: tuple[float, float, float]
    samples_df: pd.DataFrame
    fluxs: NDArray[np.float64]
    params_table: pd.DataFrame
    samples_preview: pd.DataFrame
    available_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class T1FluxCalibration:
    """Row selections using the author flux in context.

    context owns source rows and the row-aligned normalized flux array.
    t1_df/freq_rows are finite T1/f01 subsets of the source frame.
    summary_table records source and selected row counts for Notebook display.
    """

    context: T1CurveContext
    t1_df: pd.DataFrame
    freq_rows: pd.DataFrame
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class T1CurveData:
    """Aligned sample arrays with a common, sorted row axis.

    fluxs is normalized analysis flux; raw_fluxs is the author's input before
    integer alignment. integer_shifts is the integer branch adjustment.
    f01_mhz is observed/model frequency in MHz; f01_measured is True for observed
    frequencies. These records do not describe the author's preprocessing.
    T1_ns/T1err_ns are relaxation times/errors in ns (errors may be NaN);
    omegas is angular frequency in rad/ns derived from f01_mhz.
    """

    fluxs: NDArray[np.float64]
    raw_fluxs: NDArray[np.float64]
    integer_shifts: NDArray[np.float64]
    f01_mhz: NDArray[np.float64]
    f01_measured: NDArray[np.bool_]
    T1_ns: NDArray[np.float64]
    T1err_ns: NDArray[np.float64]
    omegas: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class T1PreparedData:
    """Sample and fit subsets produced by the preparation stage.

    calibration retains source selections. analysis_flux_range uses normalized
    flux; max_rel_t1_err and
    use_weighted_points_only record error filtering. sample contains retained
    window rows and fit contains its error-filtered subset, both in ns/rad/ns.
    kept_rows/source_rows count window/source T1 rows. summary_table describes
    observed/model f01, alignment and filtering without changing source identities.
    """

    calibration: T1FluxCalibration
    analysis_flux_range: tuple[float, float]
    max_rel_t1_err: float
    use_weighted_points_only: bool
    sample: T1CurveData
    fit: T1CurveData
    kept_rows: int
    source_rows: int
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PurcellCorrectionResult:
    """Rate-domain subtraction results on the observed row axis.

    observed_T1_ns/observed_T1err_ns retain observed ns values and errors.
    intrinsic_T1_ns/intrinsic_T1err_ns hold the subtracted ns values/errors,
    with NaN where no positive intrinsic rate or valid error remains.
    purcell_T1_ns is the ns limit and purcell_rate_per_ns its reciprocal rate.
    valid_mask selects positive finite intrinsic rates, independently of errors.
    """

    observed_T1_ns: NDArray[np.float64]
    observed_T1err_ns: NDArray[np.float64]
    intrinsic_T1_ns: NDArray[np.float64]
    intrinsic_T1err_ns: NDArray[np.float64]
    purcell_T1_ns: NDArray[np.float64]
    purcell_rate_per_ns: NDArray[np.float64]
    valid_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class T1MechanismProbe:
    """Pointwise mechanism analysis and fit suggestions on prepared data.

    data retains row identity; mechanism and parameter_name select the loss.
    temperature is K, omega_range is rad/ns, fit_constant records Q-fit mode.
    q_values/q_errors are dimensionless pointwise Q/errors; fit_omegas and
    fit_q_values are the selected rad/ns fit grid and dimensionless fitted Q.
    dipoles are unit-noise rates in ns^-1. parameter_values, parameter_init,
    parameter_lower and parameter_upper are dimensionless Q_cap/Q_ind or x_qp.
    pointwise_table/summary_table expose diagnostics. purcell records optional
    GHz inputs and purcell_correction records the corresponding subtraction.
    """

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
    """Combined fit inputs, result and diagnostic tables for prepared data.

    data owns ns observations. init contains K temperature and dimensionless
    active noise values; bounds uses matching units or None for defaults.
    fixed identifies fixed active parameters. residual_mode/loss/max_nfev record
    optimizer controls. fit_result owns fitted values, uncertainty and residuals.
    params_table/summary_table present fit diagnostics; purcell records the
    optional GHz resonator inputs used in the fitted total relaxation rate.
    """

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
    """Dense T1 curves with a shared normalized-flux axis.

    fluxs is the dense grid. T1_effective_ns is the intrinsic combined ns curve;
    component_T1s_ns maps ordered component labels to ns limits.
    purcell_T1_ns is the optional separate ns Purcell limit. display_T1_ns is
    the total-rate combined ns curve, or the intrinsic curve without Purcell.
    """

    fluxs: NDArray[np.float64]
    T1_effective_ns: NDArray[np.float64]
    component_T1s_ns: Mapping[str, NDArray[np.float64]]
    purcell_T1_ns: NDArray[np.float64] | None
    display_T1_ns: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class T1ChannelAnalysis:
    """Dense curves and their source fit for Notebook presentation.

    combined_fit retains the input data and fit. curves stores ns T1 arrays;
    flux_range is normalized flux and t_flux_count is the requested grid size.
    summary_table records grid, channel and fixed/free parameter information.
    """

    combined_fit: T1CombinedFit
    curves: T1ChannelCurves
    flux_range: tuple[float, float]
    t_flux_count: int
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class T1CurveAnalysisResult:
    """Collected stage records and saved figure paths from one analysis.

    context, calibration and data retain loading and preparation. cap_probe,
    qp_probe and ind_probe are optional mechanism records for active channels.
    combined_fit and channel_analysis retain fitting and dense ns T1 curves.
    summary_table combines the ordered stage diagnostics; figure_paths maps
    figure names to saved paths and is empty when no figures were saved.
    """

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


def validate_t1_mechanisms(active_mechanisms: tuple[MechanismName, ...]) -> None:
    """Validate a nonempty, duplicate-free sequence of T1 mechanism names.

    Accept capacitive, quasiparticle and inductive in caller order. Return None
    without changing the tuple; raise ValueError for empty, repeated or unknown
    entries. This does not choose noise values or fixed/free fit parameters.
    """
    if not active_mechanisms:
        raise ValueError("at least one T1 mechanism must be active")
    if len(set(active_mechanisms)) != len(active_mechanisms):
        raise ValueError("active_mechanisms contains duplicates")
    known = {"capacitive", "quasiparticle", "inductive"}
    unknown = set(active_mechanisms) - known
    if unknown:
        raise ValueError(f"unknown active mechanism(s): {sorted(unknown)}")

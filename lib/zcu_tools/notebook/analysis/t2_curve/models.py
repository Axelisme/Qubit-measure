"""Shared records and defaults for the staged T2 curve analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fit_tools import (
    FluxResidualWeighting,
    MeasurementErrorPolicy,
)
from zcu_tools.resources.qubit_params import T1CurveFit
from zcu_tools.resources.record_table import (
    DeviceValueUnit,
    RecordFluxFrame,
    RecordFluxResolution,
)

from .base import T2ChannelCurves
from .fit import ParameterName, ResidualMode, T2FitParams, T2FitResult

MechanismName = Literal["flux_noise", "photon_shot_noise"]
MechanismOrParamName = Literal["flux_noise", "photon_shot_noise", "A_phi", "n_th"]
ProbeStatistic = Literal["min", "median", "mean", "p90", "max", "half_flux"]

DisplayFn = Callable[[object], None]

DEFAULT_THERMAL_PROBE_N_TH = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@dataclass(frozen=True, slots=True)
class T2CurveAnalysisConfig:
    """Workflow inputs; device_name selects the recorded Run coordinate device."""

    result_dir: str
    device_name: str
    analysis_flux_range: tuple[float, float] = (0.49, 0.53)
    readout_kappa_over_2pi_mhz: float = 14.75412896809815
    image_dir: str | None = None
    samples_filename: str = "records.jsonl"
    default_bare_rf: float = 5.0
    t_flux_count: int = 1000
    fallback_frame_unit: DeviceValueUnit = "A"
    max_abs_flux_correction: float = 0.03
    correct_flux_from_f01_enabled: bool = True
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
    thermal_probe_n_th: tuple[float, ...] = DEFAULT_THERMAL_PROBE_N_TH
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
    """Calibrated T2 sample selection.

    ``t2e_mask`` / ``t2r_mask`` are positional boolean masks with length
    exactly ``len(context.samples_df)``; they are the authoritative T2e/T2r
    selections (stored as copies), so row identity never depends on DataFrame
    index labels and duplicate labels cannot multiply a selection.
    """

    context: T2CurveContext
    resolution: RecordFluxResolution
    fallback_frame: RecordFluxFrame | None
    t2e_df: pd.DataFrame
    t2r_df: pd.DataFrame
    t2e_mask: NDArray[np.bool_]
    t2r_mask: NDArray[np.bool_]
    freq_rows: pd.DataFrame
    summary_table: pd.DataFrame


@dataclass(frozen=True, slots=True)
class T2CurveData:
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
    fluxs: NDArray[np.float64]
    raw_fluxs: NDArray[np.float64]
    integer_shifts: NDArray[np.float64]
    flux_sources: tuple[str, ...]
    f01_mhz: NDArray[np.float64]
    f01_measured: NDArray[np.bool_]
    f01_correction_applied: NDArray[np.bool_]
    flux_corrections: NDArray[np.float64]
    correction_skipped_reason: tuple[str, ...]
    T2e_us: NDArray[np.float64]
    T2e_err_us: NDArray[np.float64]
    T1_us: NDArray[np.float64]
    T1_err_us: NDArray[np.float64]
    kept_rows: int
    source_rows: int


@dataclass(frozen=True, slots=True)
class T2RowDiagnostics:
    """Per-resolved-T2r-row flux/frequency diagnostics keyed by sample position.

    ``sample_indexes`` holds positional row indexes into ``samples_df`` (not
    DataFrame labels), so duplicate labels cannot be ambiguous. Entries follow
    deterministic ``samples_df`` row order, one entry per resolved T2r row.
    """

    sample_indexes: NDArray[np.int64]
    in_window: NDArray[np.bool_]
    flux_sources: tuple[str, ...]
    f01_observed_mhz: NDArray[np.float64]
    f01_model_mhz: NDArray[np.float64]
    f01_used_mhz: NDArray[np.float64]
    f01_measured: NDArray[np.bool_]
    raw_fluxs: NDArray[np.float64]
    corrected_fluxs: NDArray[np.float64]
    aligned_fluxs: NDArray[np.float64]
    integer_shifts: NDArray[np.float64]
    correction_applied: NDArray[np.bool_]
    correction_skipped_reason: tuple[str, ...]


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
    t2r_diagnostics: T2RowDiagnostics


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

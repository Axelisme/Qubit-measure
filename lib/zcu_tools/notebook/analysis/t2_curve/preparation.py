"""Load, calibrate and prepare T2 samples with positional row diagnostics."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fit_tools import (
    align_flux_to_window,
    correct_flux_from_f01,
    predict_f01_mhz,
)
from zcu_tools.resources.qubit_params import QubitParams
from zcu_tools.resources.record_table import (
    DeviceValueUnit,
    JsonlRecordTable,
    RecordFluxFrame,
    project_device_coordinates,
    resolve_record_flux,
    validate_record_table_v2,
)

from .models import (
    T2CurveContext,
    T2CurveData,
    T2DephasingAnalysis,
    T2FluxCalibration,
    T2RowDiagnostics,
    T2WindowData,
)

_REQUIRED_COLUMNS = [
    "freq_mhz",
    "t1_us",
    "t1_err_us",
    "t2e_us",
    "t2e_err_us",
]


def load_t2_curve_context(
    *,
    result_dir: str,
    device_name: str,
    samples_filename: str = "records.jsonl",
    image_dir: str | None = None,
    default_bare_rf: float = 5.0,
    preview_rows: int = 10,
) -> T2CurveContext:
    """Load fluxdep/dispersive params and latest JSONL rows from result_dir.

    device_name selects devices[device_name].value/value_unit; missing facts
    fail without flat-field fallback. samples_filename is relative to
    result_dir and defaults to records.jsonl;
    rows require freq_mhz, t1_us, t1_err_us, t2e_us and t2e_err_us. Create
    image_dir (default result_dir/t2_curve) and return rows with a preview of
    preview_rows rows. default_bare_rf is in GHz for dispersive input loading.
    Missing params, invalid JSONL or missing required columns fail fast.
    """
    resolved_image_dir = image_dir or os.path.join(result_dir, "t2_curve")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(resolved_image_dir, exist_ok=True)

    params_file = QubitParams(os.path.join(result_dir, "params.json"), readonly=True)
    fit_inputs = params_file.require_dispersive_inputs(default_bare_rf=default_bare_rf)
    dispersive_fit = params_file.get_dispersive_fit()
    if dispersive_fit is None:
        raise RuntimeError("params.json needs a dispersive section with g and bare_rf.")

    samples_df = project_device_coordinates(
        JsonlRecordTable(os.path.join(result_dir, samples_filename)).read(),
        device_name=device_name,
    )
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
    fallback_frame_unit: DeviceValueUnit = "A",
) -> T2FluxCalibration:
    samples_df = context.samples_df
    fallback_frame = RecordFluxFrame(
        fallback_frame_unit, context.flux_int, context.flux_period
    )
    resolution = resolve_record_flux(
        samples_df,
        fallback_frame=fallback_frame,
    )
    t2e_values = _float_column(samples_df, "t2e_us")
    t2e_mask = np.isfinite(t2e_values)
    t2e_df = samples_df.loc[t2e_mask].copy()
    if "t2r_us" in samples_df.columns:
        t2r_values = _float_column(samples_df, "t2r_us")
        t2r_mask = np.isfinite(t2r_values)
        t2r_df = samples_df.loc[t2r_mask].copy()
    else:
        t2r_mask = np.zeros(len(samples_df), dtype=bool)
        t2r_df = samples_df.iloc[0:0].copy()

    freq_mask = np.isfinite(_float_column(samples_df, "freq_mhz"))
    freq_rows = samples_df.loc[freq_mask].copy()
    source_counts = {
        source: sum(1 for item in resolution.sources if item == source)
        for source in ("explicit", "row-frame", "fallback-frame")
    }
    summary_table = pd.DataFrame(
        [
            ("records.jsonl rows", str(len(samples_df))),
            ("explicit flux rows", str(source_counts["explicit"])),
            ("row-frame flux rows", str(source_counts["row-frame"])),
            ("fallback-frame flux rows", str(source_counts["fallback-frame"])),
            ("fallback frame unit", fallback_frame.dev_unit),
            ("finite f01 rows", str(len(freq_rows))),
            ("finite T2e rows", str(len(t2e_df))),
            ("finite T2r rows", str(len(t2r_df))),
        ],
        columns=["metric", "value"],
    )
    return T2FluxCalibration(
        context=context,
        resolution=resolution,
        fallback_frame=fallback_frame,
        t2e_df=t2e_df,
        t2r_df=t2r_df,
        t2e_mask=t2e_mask.copy(),
        t2r_mask=t2r_mask.copy(),
        freq_rows=freq_rows,
        summary_table=summary_table,
    )


def prepare_t2_dephasing_data(
    calibration: T2FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float] = (0.49, 0.53),
    max_abs_flux_correction: float = 0.03,
    correct_flux_from_f01_enabled: bool = True,
    max_rel_t2e_err: float = 0.5,
    use_weighted_points_only: bool = False,
) -> T2DephasingAnalysis:
    window = _prepare_window_data(
        calibration,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    t2r_diagnostics = _t2r_diagnostics(
        calibration,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    branch_coverage = _branch_coverage_table(
        calibration,
        t2r_diagnostics=t2r_diagnostics,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    half_preview = _half_preview_table(
        calibration,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    sample, fit = _derive_dephasing(
        window,
        max_rel_t2e_err=max_rel_t2e_err,
        use_weighted_points_only=use_weighted_points_only,
    )
    if len(fit.T2e_us) == 0:
        raise ValueError("No T2e fit rows remain after filtering")

    summary_table = _dephasing_summary_table(
        window=window,
        sample=sample,
        fit=fit,
        analysis_flux_range=analysis_flux_range,
        max_rel_t2e_err=max_rel_t2e_err,
        use_weighted_points_only=use_weighted_points_only,
        t2r_diagnostics=t2r_diagnostics,
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
        t2r_diagnostics=t2r_diagnostics,
    )


# tolerated: Notebook documents probe controls and shared fit-policy keywords explicitly.
def _validate_required_columns(samples_df: pd.DataFrame) -> None:
    validate_record_table_v2(samples_df, allow_empty=True)
    missing_columns = [
        name for name in _REQUIRED_COLUMNS if name not in samples_df.columns
    ]
    if missing_columns:
        raise KeyError(f"Missing required sample columns: {missing_columns}")


def _float_column(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    return np.asarray(frame[column], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _ResolvedFrame:
    raw_fluxs: NDArray[np.float64]
    corrected_fluxs: NDArray[np.float64]
    aligned_fluxs: NDArray[np.float64]
    shifts: NDArray[np.float64]
    in_window: NDArray[np.bool_]
    f01_measured: NDArray[np.bool_]
    f01_correction_applied: NDArray[np.bool_]
    correction_skipped_reason: tuple[str, ...]
    flux_sources: tuple[str, ...]


def _resolve_analysis_frame(
    calibration: T2FluxCalibration,
    row_mask: NDArray[np.bool_],
    *,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    correct_flux_from_f01_enabled: bool = True,
) -> _ResolvedFrame:
    samples_df = calibration.context.samples_df
    resolution = calibration.resolution
    raw_fluxs = np.asarray(resolution.values[row_mask], dtype=np.float64)
    row_sources = tuple(resolution.sources[index] for index in np.flatnonzero(row_mask))
    f01_observed = _float_column(samples_df, "freq_mhz")[row_mask]
    f01_measured = np.isfinite(f01_observed)
    corrected_fluxs = np.asarray(raw_fluxs, dtype=np.float64).copy()
    applied = np.zeros(len(raw_fluxs), dtype=bool)
    reasons: list[str] = []
    if correct_flux_from_f01_enabled:
        if np.any(f01_measured):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                correction = correct_flux_from_f01(
                    raw_fluxs,
                    1e-3 * f01_observed,
                    calibration.context.params,
                    max_abs_flux_correction=max_abs_flux_correction,
                )
            candidate_fluxs = correction.corrected_fluxs
            accepted = correction.accepted
            for index, source in enumerate(row_sources):
                if source == "explicit":
                    reasons.append("explicit_flux")
                    continue
                if not f01_measured[index]:
                    reasons.append("missing_f01")
                    continue
                if accepted[index]:
                    corrected_fluxs[index] = candidate_fluxs[index]
                    applied[index] = True
                    reasons.append("")
                else:
                    reasons.append("rejected")
        else:
            reasons.extend(
                "explicit_flux" if source == "explicit" else "missing_f01"
                for source in row_sources
            )
    else:
        reasons.extend("disabled" for _ in row_sources)
    aligned_fluxs, shifts, in_window = align_flux_to_window(
        corrected_fluxs, analysis_flux_range
    )
    return _ResolvedFrame(
        raw_fluxs=raw_fluxs,
        corrected_fluxs=corrected_fluxs,
        aligned_fluxs=aligned_fluxs,
        shifts=shifts,
        in_window=in_window,
        f01_measured=f01_measured,
        f01_correction_applied=applied,
        correction_skipped_reason=tuple(reasons),
        flux_sources=row_sources,
    )


def _prepare_window_data(
    calibration: T2FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    correct_flux_from_f01_enabled: bool = True,
) -> T2WindowData:
    samples_df = calibration.context.samples_df
    t2e_values = _float_column(samples_df, "t2e_us")
    valid_t2e = np.isfinite(t2e_values) & (t2e_values > 0.0)
    if not np.any(valid_t2e):
        raise ValueError("no finite positive T2e rows are available for analysis")
    resolved = _resolve_analysis_frame(
        calibration,
        valid_t2e,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    order = np.argsort(resolved.aligned_fluxs[resolved.in_window])

    def take[T: np.generic](values_arr: NDArray[T]) -> NDArray[T]:
        return np.asarray(values_arr)[resolved.in_window][order]

    f01_observed = _float_column(samples_df, "freq_mhz")[valid_t2e]
    measured_taken = take(resolved.f01_measured)
    observed_taken = take(f01_observed)
    f01_used = np.where(
        measured_taken,
        observed_taken,
        predict_f01_mhz(calibration.context.params, take(resolved.aligned_fluxs)),
    )
    return T2WindowData(
        fluxs=take(resolved.aligned_fluxs),
        raw_fluxs=take(resolved.raw_fluxs),
        integer_shifts=take(resolved.shifts),
        flux_sources=tuple(
            take(np.asarray(resolved.flux_sources, dtype=object)).tolist()
        ),
        f01_mhz=f01_used,
        f01_measured=measured_taken,
        f01_correction_applied=take(resolved.f01_correction_applied),
        flux_corrections=take(resolved.corrected_fluxs - resolved.raw_fluxs),
        correction_skipped_reason=tuple(
            take(np.asarray(resolved.correction_skipped_reason, dtype=object)).tolist()
        ),
        T2e_us=take(_float_column(samples_df, "t2e_us")[valid_t2e]),
        T2e_err_us=take(_float_column(samples_df, "t2e_err_us")[valid_t2e]),
        T1_us=take(_float_column(samples_df, "t1_us")[valid_t2e]),
        T1_err_us=take(_float_column(samples_df, "t1_err_us")[valid_t2e]),
        kept_rows=int(np.count_nonzero(resolved.in_window)),
        source_rows=int(np.count_nonzero(valid_t2e)),
    )


def _derive_dephasing(
    window: T2WindowData,
    *,
    max_rel_t2e_err: float,
    use_weighted_points_only: bool,
) -> tuple[T2CurveData, T2CurveData]:
    sample_mask = (
        np.isfinite(window.fluxs)
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


def _t2r_diagnostics(
    calibration: T2FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    correct_flux_from_f01_enabled: bool = True,
) -> T2RowDiagnostics:
    """Build per-row T2r diagnostics from one resolved T2r frame.

    Entries follow deterministic ``samples_df`` positional row order (one entry
    per resolved T2r row); ``sample_indexes`` preserves the stable positional
    identity of each source-table row. Model f01 is predicted at the aligned
    analysis flux, so null-observed explicit/derived rows keep a directly
    inspectable finite model frequency.
    """
    samples_df = calibration.context.samples_df
    row_mask = calibration.t2r_mask
    if not np.any(row_mask):
        return _empty_t2r_diagnostics()
    resolved = _resolve_analysis_frame(
        calibration,
        row_mask,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    f01_observed = _float_column(samples_df, "freq_mhz")[row_mask]
    f01_model = predict_f01_mhz(calibration.context.params, resolved.aligned_fluxs)
    return T2RowDiagnostics(
        sample_indexes=np.flatnonzero(row_mask).astype(np.int64),
        in_window=resolved.in_window,
        flux_sources=resolved.flux_sources,
        f01_observed_mhz=f01_observed,
        f01_model_mhz=f01_model,
        f01_used_mhz=np.where(resolved.f01_measured, f01_observed, f01_model),
        f01_measured=resolved.f01_measured,
        raw_fluxs=resolved.raw_fluxs,
        corrected_fluxs=resolved.corrected_fluxs,
        aligned_fluxs=resolved.aligned_fluxs,
        integer_shifts=resolved.shifts,
        correction_applied=resolved.f01_correction_applied,
        correction_skipped_reason=resolved.correction_skipped_reason,
    )


def _empty_t2r_diagnostics() -> T2RowDiagnostics:
    return T2RowDiagnostics(
        sample_indexes=np.empty(0, dtype=np.int64),
        in_window=np.empty(0, dtype=bool),
        flux_sources=(),
        f01_observed_mhz=np.empty(0, dtype=np.float64),
        f01_model_mhz=np.empty(0, dtype=np.float64),
        f01_used_mhz=np.empty(0, dtype=np.float64),
        f01_measured=np.empty(0, dtype=bool),
        raw_fluxs=np.empty(0, dtype=np.float64),
        corrected_fluxs=np.empty(0, dtype=np.float64),
        aligned_fluxs=np.empty(0, dtype=np.float64),
        integer_shifts=np.empty(0, dtype=np.float64),
        correction_applied=np.empty(0, dtype=bool),
        correction_skipped_reason=(),
    )


def _branch_coverage_table(
    calibration: T2FluxCalibration,
    *,
    t2r_diagnostics: T2RowDiagnostics,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    correct_flux_from_f01_enabled: bool = True,
) -> pd.DataFrame:
    rows = [
        _coverage_row(
            "T2e rows",
            calibration,
            calibration.t2e_mask,
            analysis_flux_range=analysis_flux_range,
            max_abs_flux_correction=max_abs_flux_correction,
            correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
        ),
        _coverage_row_from_diagnostics("T2r rows", t2r_diagnostics),
    ]
    return pd.DataFrame(rows)


def _coverage_row(
    label: str,
    calibration: T2FluxCalibration,
    row_mask: NDArray[np.bool_],
    *,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    correct_flux_from_f01_enabled: bool = True,
) -> dict[str, object]:
    if not np.any(row_mask):
        return _empty_coverage_row(label)
    resolved = _resolve_analysis_frame(
        calibration,
        row_mask,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    return _coverage_columns(
        label,
        resolved,
        model_f01_finite=np.isfinite(
            predict_f01_mhz(calibration.context.params, resolved.aligned_fluxs)
        ),
    )


def _coverage_row_from_diagnostics(
    label: str, diagnostics: T2RowDiagnostics
) -> dict[str, object]:
    if len(diagnostics.raw_fluxs) == 0:
        return _empty_coverage_row(label)
    resolved = _ResolvedFrame(
        raw_fluxs=diagnostics.raw_fluxs,
        corrected_fluxs=diagnostics.corrected_fluxs,
        aligned_fluxs=diagnostics.aligned_fluxs,
        shifts=diagnostics.integer_shifts,
        in_window=diagnostics.in_window,
        f01_measured=diagnostics.f01_measured,
        f01_correction_applied=diagnostics.correction_applied,
        correction_skipped_reason=diagnostics.correction_skipped_reason,
        flux_sources=diagnostics.flux_sources,
    )
    return _coverage_columns(label, resolved, np.isfinite(diagnostics.f01_model_mhz))


def _coverage_columns(
    label: str,
    resolved: _ResolvedFrame,
    model_f01_finite: NDArray[np.bool_],
) -> dict[str, object]:
    raw_fluxs = resolved.raw_fluxs
    corrected_fluxs = resolved.corrected_fluxs
    aligned_fluxs = resolved.aligned_fluxs
    shifts = resolved.shifts
    in_window = resolved.in_window
    f01_measured = resolved.f01_measured
    correction_applied = resolved.f01_correction_applied
    correction_skipped_reason = resolved.correction_skipped_reason
    flux_sources = resolved.flux_sources
    shifts_in_window = shifts[in_window]
    source_counts = {
        source: sum(1 for item in flux_sources if item == source)
        for source in ("explicit", "row-frame", "fallback-frame")
    }
    skip_counts = {
        reason: sum(1 for item in correction_skipped_reason if item == reason)
        for reason in ("explicit_flux", "missing_f01", "rejected", "disabled")
    }
    return {
        "subset": label,
        "n_explicit": source_counts["explicit"],
        "n_row_frame": source_counts["row-frame"],
        "n_fallback_frame": source_counts["fallback-frame"],
        "n_with_f01": int(np.count_nonzero(f01_measured)),
        "n_model_f01": int(np.count_nonzero(~f01_measured & model_f01_finite)),
        "n_in_flux_window": int(np.count_nonzero(in_window)),
        "accepted": int(np.count_nonzero(correction_applied)),
        "skipped": int(
            np.count_nonzero(
                np.asarray([bool(reason) for reason in correction_skipped_reason])
            )
        ),
        "skipped_explicit_flux": skip_counts["explicit_flux"],
        "skipped_missing_f01": skip_counts["missing_f01"],
        "skipped_rejected": skip_counts["rejected"],
        "skipped_disabled": skip_counts["disabled"],
        "raw_min_flux": float(np.nanmin(raw_fluxs)),
        "raw_max_flux": float(np.nanmax(raw_fluxs)),
        "corr_min_flux": (
            np.nan
            if not np.any(in_window)
            else float(np.nanmin(corrected_fluxs[in_window]))
        ),
        "corr_max_flux": (
            np.nan
            if not np.any(in_window)
            else float(np.nanmax(corrected_fluxs[in_window]))
        ),
        "aligned_min_flux": (
            np.nan
            if not np.any(in_window)
            else float(np.nanmin(aligned_fluxs[in_window]))
        ),
        "aligned_max_flux": (
            np.nan
            if not np.any(in_window)
            else float(np.nanmax(aligned_fluxs[in_window]))
        ),
        "window_below_half_flux": int(np.count_nonzero(aligned_fluxs[in_window] < 0.5)),
        "window_at_half_flux": int(
            np.count_nonzero(np.isclose(aligned_fluxs[in_window], 0.5))
        ),
        "window_above_half_flux": int(np.count_nonzero(aligned_fluxs[in_window] > 0.5)),
        "window_shifted_rows": int(np.count_nonzero(shifts_in_window != 0.0)),
        "window_shift_min": (
            np.nan if not np.any(in_window) else float(np.nanmin(shifts_in_window))
        ),
        "window_shift_max": (
            np.nan if not np.any(in_window) else float(np.nanmax(shifts_in_window))
        ),
    }


def _empty_coverage_row(label: str) -> dict[str, object]:
    return {
        "subset": label,
        "n_explicit": 0,
        "n_row_frame": 0,
        "n_fallback_frame": 0,
        "n_with_f01": 0,
        "n_model_f01": 0,
        "n_in_flux_window": 0,
        "accepted": 0,
        "skipped": 0,
        "skipped_explicit_flux": 0,
        "skipped_missing_f01": 0,
        "skipped_rejected": 0,
        "skipped_disabled": 0,
        "raw_min_flux": np.nan,
        "raw_max_flux": np.nan,
        "corr_min_flux": np.nan,
        "corr_max_flux": np.nan,
        "aligned_min_flux": np.nan,
        "aligned_max_flux": np.nan,
        "window_below_half_flux": 0,
        "window_at_half_flux": 0,
        "window_above_half_flux": 0,
        "window_shifted_rows": 0,
        "window_shift_min": np.nan,
        "window_shift_max": np.nan,
    }


def _half_preview_table(
    calibration: T2FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    correct_flux_from_f01_enabled: bool = True,
) -> pd.DataFrame:
    samples_df = calibration.context.samples_df
    row_mask = calibration.t2e_mask
    if not np.any(row_mask):
        return samples_df.iloc[0:0].copy()
    resolved = _resolve_analysis_frame(
        calibration,
        row_mask,
        analysis_flux_range=analysis_flux_range,
        max_abs_flux_correction=max_abs_flux_correction,
        correct_flux_from_f01_enabled=correct_flux_from_f01_enabled,
    )
    frame = calibration.t2e_df
    preview_columns = [
        column
        for column in [
            "freq_mhz",
            "t2r_us",
            "t2e_us",
            "t2e_err_us",
        ]
        if column in frame.columns
    ]
    preview = frame.loc[:, preview_columns].copy()
    preview["raw flux"] = resolved.raw_fluxs
    preview["f01-corrected flux"] = resolved.corrected_fluxs
    preview["aligned flux"] = resolved.aligned_fluxs
    preview["in window"] = resolved.in_window
    preview["integer shift"] = resolved.shifts
    in_half = (
        resolved.in_window
        & (resolved.aligned_fluxs >= analysis_flux_range[0])
        & (resolved.aligned_fluxs <= analysis_flux_range[1])
        & np.isclose(resolved.aligned_fluxs, 0.5)
    )
    return preview.loc[in_half].copy()


def _diagnostic_rows(
    prefix: str,
    data: T2WindowData | T2RowDiagnostics,
) -> list[tuple[str, str]]:
    """Summarize the selected window rows, preserving provenance and shift order."""
    if isinstance(data, T2WindowData):
        sources = data.flux_sources
        f01_measured = data.f01_measured
        correction_skipped_reason = data.correction_skipped_reason
        integer_shifts = data.integer_shifts
        model_f01_finite = np.isfinite(data.f01_mhz) & ~data.f01_measured
    else:
        in_window = data.in_window
        sources = tuple(
            source
            for source, keep in zip(data.flux_sources, in_window, strict=True)
            if keep
        )
        f01_measured = data.f01_measured[in_window]
        correction_skipped_reason = tuple(
            reason
            for reason, keep in zip(
                data.correction_skipped_reason, in_window, strict=True
            )
            if keep
        )
        integer_shifts = data.integer_shifts[in_window]
        model_f01_finite = np.isfinite(data.f01_model_mhz[in_window])
    source_counts = {
        source: sum(1 for item in sources if item == source)
        for source in ("explicit", "row-frame", "fallback-frame")
    }
    observed_count = int(np.count_nonzero(f01_measured))
    model_count = int(np.count_nonzero(~f01_measured & model_f01_finite))
    skip_reasons: dict[str, int] = {}
    for reason in correction_skipped_reason:
        if reason:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    shift_counts: dict[str, int] = {}
    for shift in integer_shifts:
        key = f"{int(round(float(shift)))}"
        shift_counts[key] = shift_counts.get(key, 0) + 1
    shift_summary = (
        " ".join(f"k={k}:{count}" for k, count in sorted(shift_counts.items()))
        if shift_counts
        else "0"
    )
    label = f"{prefix} " if prefix else ""
    return [
        (
            f"{label}flux sources (explicit/row/fallback)",
            f"{source_counts['explicit']}/{source_counts['row-frame']}/"
            f"{source_counts['fallback-frame']}",
        ),
        (f"{label}f01 source (observed/model)", f"{observed_count}/{model_count}"),
        (
            f"{label}f01 correction skipped reasons",
            " ".join(f"{reason}={count}" for reason, count in skip_reasons.items())
            or "none",
        ),
        (f"{label}integer shifts", shift_summary),
    ]


# tolerated: One preparation caller supplies pre-result stages and selection diagnostics.
def _dephasing_summary_table(  # noqa: PLR0913
    *,
    window: T2WindowData,
    sample: T2CurveData,
    fit: T2CurveData,
    analysis_flux_range: tuple[float, float],
    max_rel_t2e_err: float,
    use_weighted_points_only: bool,
    t2r_diagnostics: T2RowDiagnostics,
) -> pd.DataFrame:
    sample_peak_idx = int(np.nanargmax(sample.T2e_us))
    fit_peak_idx = int(np.nanargmax(fit.T2e_us))
    rows: list[tuple[str, str]] = [
        (
            "analysis flux range",
            f"{analysis_flux_range[0]:.3f}..{analysis_flux_range[1]:.3f}",
        ),
        ("max relative T2e err", f"{max_rel_t2e_err:.3g}"),
        ("use weighted points only", str(use_weighted_points_only)),
        ("window T2e rows", f"{window.kept_rows}/{window.source_rows}"),
        ("sample rows", str(len(sample.T2e_us))),
        ("fit rows", str(len(fit.T2e_us))),
    ]
    rows.extend(_diagnostic_rows("", window))
    if len(t2r_diagnostics.raw_fluxs) > 0:
        in_window = t2r_diagnostics.in_window
        rows.append(
            (
                "window T2r rows",
                f"{int(np.count_nonzero(in_window))}/{len(t2r_diagnostics.raw_fluxs)}",
            )
        )
        rows.extend(_diagnostic_rows("T2r", t2r_diagnostics))
    rows.extend(
        [
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
        ]
    )
    return pd.DataFrame(rows, columns=["metric", "value"])

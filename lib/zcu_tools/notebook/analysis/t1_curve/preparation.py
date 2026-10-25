"""Sample loading, flux calibration and row-preserving T1 preparation."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fit_tools import (
    align_flux_to_window,
    correct_flux_from_f01,
    predict_f01_mhz,
)
from zcu_tools.resources.qubit_params import (
    QubitParams,
)
from zcu_tools.resources.record_table import (
    DeviceValueUnit,
    JsonlRecordTable,
    RecordFluxFrame,
    project_device_coordinates,
    resolve_record_flux,
    validate_record_table_v2,
)

from .models import (
    T1CurveContext,
    T1CurveData,
    T1FluxCalibration,
    T1PreparedData,
)
from .utils import freq2omega

_REQUIRED_COLUMNS = ("freq_mhz", "t1_us")


def load_t1_curve_context(
    *,
    result_dir: str,
    device_name: str,
    samples_filename: str = "records.jsonl",
    image_dir: str | None = None,
    preview_rows: int = 10,
) -> T1CurveContext:
    """Load fluxdep params and latest JSONL sample rows from result_dir.

    device_name selects devices[device_name].value/value_unit; missing facts
    fail without flat-field fallback. samples_filename is relative to
    result_dir and defaults to records.jsonl;
    required measurements are freq_mhz and t1_us. Create image_dir (default
    result_dir/t1_curve) and return params, rows and preview_rows preview rows.
    Missing params, invalid JSONL or missing required columns fail fast.
    """
    resolved_image_dir = image_dir or os.path.join(result_dir, "t1_curve")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(resolved_image_dir, exist_ok=True)

    params_file = QubitParams(os.path.join(result_dir, "params.json"), readonly=True)
    fit_inputs = params_file.require_fluxdep_fit()
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
    fallback_frame_unit: DeviceValueUnit = "A",
) -> T1FluxCalibration:
    """Resolve sample flux and record provenance without correcting rows.

    context contains the loaded sample frame and dimensionless flux alignment.
    fallback_frame_unit declares the device-value unit (A, V or native) used
    with that alignment. Return row-aligned resolution, finite measurement
    subsets and a summary. Invalid or unresolved sample coordinates raise the
    sample-table validation error; source row order is retained.
    """
    samples_df = context.samples_df
    fallback_frame = RecordFluxFrame(
        fallback_frame_unit, context.flux_int, context.flux_period
    )
    resolution = resolve_record_flux(
        samples_df,
        fallback_frame=fallback_frame,
    )
    t1_values = _float_column(samples_df, "t1_us")
    t1_mask = np.isfinite(t1_values)
    freq_values = _float_column(samples_df, "freq_mhz")
    freq_mask = np.isfinite(freq_values)
    t1_df = samples_df.loc[t1_mask].copy()
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
            ("finite T1 rows", str(len(t1_df))),
        ],
        columns=["metric", "value"],
    )
    return T1FluxCalibration(
        context=context,
        resolution=resolution,
        fallback_frame=fallback_frame,
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
    """Prepare normalized flux, GHz-derived omega and ns T1/error arrays.

    calibration retains source rows and flux provenance. analysis_flux_range
    selects the dimensionless window after integer alignment; correction uses
    max_abs_flux_correction in normalized flux and only derived measured-f01
    rows when enabled. max_rel_t1_err bounds positive relative errors;
    use_weighted_points_only drops NaN errors instead of retaining them.
    Return sorted sample/fit records and diagnostics without mutating source
    rows. Invalid windows or no remaining positive T1 fit rows raise ValueError.
    """
    sample = _prepare_t1_data(
        calibration,
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


def _validate_required_columns(samples_df: pd.DataFrame) -> None:
    validate_record_table_v2(samples_df, allow_empty=True)
    missing = [
        column for column in _REQUIRED_COLUMNS if column not in samples_df.columns
    ]
    if missing:
        raise ValueError(f"records.jsonl missing required columns: {missing}")


def _float_column(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce")
    return np.asarray(values, dtype=np.float64)


def _prepare_t1_data(
    calibration: T1FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float],
    max_abs_flux_correction: float,
    correct_flux_from_f01_enabled: bool,
) -> T1CurveData:
    context = calibration.context
    samples_df = context.samples_df

    t1_values = _float_column(samples_df, "t1_us")
    eligible = np.isfinite(t1_values) & (t1_values > 0.0)
    if not np.any(eligible):
        raise ValueError("no finite positive T1 rows are available for analysis")

    resolved_values = calibration.resolution.values
    sources = calibration.resolution.sources
    raw_fluxs = np.asarray(resolved_values[eligible], dtype=np.float64)
    row_sources = tuple(sources[index] for index in np.flatnonzero(eligible))
    f01_observed = _float_column(samples_df, "freq_mhz")
    f01_observed = f01_observed[eligible]
    f01_measured = np.isfinite(f01_observed)
    T1_ns = 1e3 * t1_values[eligible]
    T1err_ns = 1e3 * _float_column(samples_df, "t1_err_us")[eligible]

    corrected_fluxs = np.asarray(raw_fluxs, dtype=np.float64).copy()
    applied = np.zeros(len(raw_fluxs), dtype=bool)
    flux_corrections = np.zeros(len(raw_fluxs), dtype=np.float64)
    reasons: list[str] = []
    if correct_flux_from_f01_enabled:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            correction = correct_flux_from_f01(
                raw_fluxs,
                1e-3 * f01_observed,
                context.params,
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
                flux_corrections[index] = candidate_fluxs[index] - raw_fluxs[index]
                applied[index] = True
                reasons.append("")
            else:
                reasons.append("rejected")
    else:
        reasons.extend("disabled" for _ in row_sources)

    aligned_fluxs, shifts, in_window = align_flux_to_window(
        corrected_fluxs, analysis_flux_range
    )
    order = np.argsort(aligned_fluxs[in_window])

    def take[T: np.generic](values_arr: NDArray[T]) -> NDArray[T]:
        return np.asarray(values_arr)[in_window][order]

    aligned_taken = take(aligned_fluxs)
    measured_taken = take(f01_measured)
    observed_taken = take(f01_observed)
    f01_used = np.where(
        measured_taken, observed_taken, predict_f01_mhz(context.params, aligned_taken)
    )
    return T1CurveData(
        fluxs=aligned_taken,
        raw_fluxs=take(raw_fluxs),
        integer_shifts=take(shifts),
        flux_sources=tuple(take(np.asarray(row_sources, dtype=object)).tolist()),
        f01_mhz=f01_used,
        f01_measured=measured_taken,
        f01_correction_applied=take(applied),
        flux_corrections=take(flux_corrections),
        correction_skipped_reason=tuple(
            take(np.asarray(reasons, dtype=object)).tolist()
        ),
        T1_ns=take(T1_ns),
        T1err_ns=take(T1err_ns),
        omegas=np.asarray(freq2omega(1e-3 * f01_used), dtype=np.float64),
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
        fluxs=data.fluxs[mask],
        raw_fluxs=data.raw_fluxs[mask],
        integer_shifts=data.integer_shifts[mask],
        flux_sources=tuple(
            source for source, keep in zip(data.flux_sources, mask, strict=True) if keep
        ),
        f01_mhz=data.f01_mhz[mask],
        f01_measured=data.f01_measured[mask],
        f01_correction_applied=data.f01_correction_applied[mask],
        flux_corrections=data.flux_corrections[mask],
        correction_skipped_reason=tuple(
            reason
            for reason, keep in zip(data.correction_skipped_reason, mask, strict=True)
            if keep
        ),
        T1_ns=data.T1_ns[mask],
        T1err_ns=data.T1err_ns[mask],
        omegas=data.omegas[mask],
    )


# tolerated: One preparation caller supplies stage arrays and filter metadata; no reused option group.
def _prepared_summary_table(  # noqa: PLR0913
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
    source_counts = {
        source: sum(1 for item in sample.flux_sources if item == source)
        for source in ("explicit", "row-frame", "fallback-frame")
    }
    observed_count = int(np.count_nonzero(sample.f01_measured))
    model_count = len(sample.fluxs) - observed_count
    skip_reasons: dict[str, int] = {}
    for reason in sample.correction_skipped_reason:
        if reason:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    shift_counts: dict[str, int] = {}
    for shift in sample.integer_shifts:
        key = f"{int(round(float(shift)))}"
        shift_counts[key] = shift_counts.get(key, 0) + 1
    shift_summary = (
        " ".join(f"k={k}:{count}" for k, count in sorted(shift_counts.items()))
        if shift_counts
        else "0"
    )
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
            (
                "flux sources (explicit/row/fallback)",
                f"{source_counts['explicit']}/{source_counts['row-frame']}/"
                f"{source_counts['fallback-frame']}",
            ),
            ("f01 source (observed/model)", f"{observed_count}/{model_count}"),
            (
                "f01 correction skipped reasons",
                " ".join(f"{reason}={count}" for reason, count in skip_reasons.items())
                or "none",
            ),
            ("integer shifts", shift_summary),
            (
                "sample peak",
                f"flux={sample.fluxs[peak_idx]:.6f}, T1={1e-3 * sample.T1_ns[peak_idx]:.3f} us",
            ),
            (
                "f01 corrections applied",
                f"{int(np.count_nonzero(sample.f01_correction_applied))}/{len(sample.fluxs)}",
            ),
        ],
        columns=["metric", "value"],
    )

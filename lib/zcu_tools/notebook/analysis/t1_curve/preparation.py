"""Sample loading, flux calibration and row-preserving T1 preparation."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from zcu_tools.notebook.analysis.fit_tools import (
    align_flux_to_window,
    predict_f01_mhz,
)
from zcu_tools.resources.qubit_params import (
    QubitParams,
)

from .models import (
    T1CurveContext,
    T1CurveData,
    T1FluxCalibration,
    T1PreparedData,
)
from .utils import freq2omega

_REQUIRED_COLUMNS = ("t1_us",)


def load_t1_curve_context(
    *,
    result_dir: str,
    fluxs: NDArray[np.float64],
    samples_filename: str = "samples.csv",
    image_dir: str | None = None,
    preview_rows: int = 10,
) -> T1CurveContext:
    """Load params.json and CSV samples from result_dir.

    samples_filename is relative to result_dir, or an absolute CSV path.
    fluxs supplies one normalized Phi0 coordinate per CSV row, in row order.
    CSV flux/frame columns are ignored. Require numeric t1_us; missing freq_mhz
    and t1_err_us become NaN. Ignore unused columns. Create image_dir (default
    result_dir/t1_curve) and return inputs with preview_rows initial rows.
    Missing files/parameters, missing required columns or nonnumeric used
    values raise with the file or column name. Non-1D, nonfinite or wrong-length
    fluxs raise ValueError before any analysis.
    """
    resolved_image_dir = image_dir or os.path.join(result_dir, "t1_curve")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(resolved_image_dir, exist_ok=True)

    params_file = QubitParams(os.path.join(result_dir, "params.json"), readonly=True)
    fit_inputs = params_file.require_fluxdep_fit()
    samples_df = pd.read_csv(os.path.join(result_dir, samples_filename))
    _validate_required_columns(samples_df)
    author_fluxs = np.asarray(fluxs, dtype=np.float64).copy()
    if author_fluxs.ndim != 1:
        raise ValueError("fluxs must be a one-dimensional array")
    if len(author_fluxs) != len(samples_df):
        raise ValueError("fluxs must contain one value per samples.csv row")
    if not np.all(np.isfinite(author_fluxs)):
        raise ValueError("fluxs must contain finite values for every row")

    params = fit_inputs.params
    params_table = pd.DataFrame(
        [
            ("EJ (GHz)", params[0]),
            ("EC (GHz)", params[1]),
            ("EL (GHz)", params[2]),
        ],
        columns=["parameter", "value"],
    )

    return T1CurveContext(
        result_dir=result_dir,
        image_dir=resolved_image_dir,
        samples_filename=samples_filename,
        params=params,
        samples_df=samples_df,
        fluxs=author_fluxs,
        params_table=params_table,
        samples_preview=samples_df.head(preview_rows),
        available_columns=tuple(str(column) for column in samples_df.columns),
    )


def calibrate_t1_flux(context: T1CurveContext) -> T1FluxCalibration:
    """Select finite T1/f01 rows using the context's author-supplied flux.

    Return row-aligned flux and a count summary without correcting coordinates
    or mutating context. Missing/nonnumeric used columns raise ValueError.
    Nonfinite flux raises ValueError because no device-frame fallback exists.
    """
    samples_df = context.samples_df.copy()
    _validate_required_columns(samples_df)
    fluxs = context.fluxs
    if not np.all(np.isfinite(fluxs)):
        raise ValueError("author fluxs must be finite")
    t1_values = _float_column(samples_df, "t1_us")
    t1_mask = np.isfinite(t1_values)
    freq_values = _float_column(samples_df, "freq_mhz")
    freq_mask = np.isfinite(freq_values)
    t1_df = samples_df.loc[t1_mask].copy()
    freq_rows = samples_df.loc[freq_mask].copy()
    summary_table = pd.DataFrame(
        [
            ("samples.csv rows", str(len(samples_df))),
            ("finite f01 rows", str(len(freq_rows))),
            ("finite T1 rows", str(len(t1_df))),
        ],
        columns=["metric", "value"],
    )
    return T1FluxCalibration(
        context=context,
        t1_df=t1_df,
        freq_rows=freq_rows,
        summary_table=summary_table,
    )


def prepare_t1_curve_data(
    calibration: T1FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float] = (0.0, 1.0),
    max_rel_t1_err: float = 0.25,
    use_weighted_points_only: bool = False,
) -> T1PreparedData:
    """Prepare normalized flux, GHz-derived omega and ns T1/error arrays.

    calibration retains source rows and author flux. analysis_flux_range
    selects the dimensionless window after integer alignment; flux is not
    corrected. max_rel_t1_err bounds positive relative errors;
    use_weighted_points_only drops NaN errors instead of retaining them.
    Return sorted sample/fit records and diagnostics without mutating source
    rows. Invalid windows or no remaining positive T1 fit rows raise ValueError.
    """
    sample = _prepare_t1_data(
        calibration,
        analysis_flux_range=analysis_flux_range,
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
        max_rel_t1_err=max_rel_t1_err,
        use_weighted_points_only=use_weighted_points_only,
    )
    return T1PreparedData(
        calibration=calibration,
        analysis_flux_range=analysis_flux_range,
        max_rel_t1_err=max_rel_t1_err,
        use_weighted_points_only=use_weighted_points_only,
        sample=sample,
        fit=fit,
        kept_rows=len(sample.fluxs),
        source_rows=len(calibration.t1_df),
        summary_table=summary_table,
    )


def _validate_required_columns(samples_df: pd.DataFrame) -> None:
    missing = [
        column for column in _REQUIRED_COLUMNS if column not in samples_df.columns
    ]
    if missing:
        raise ValueError(f"samples.csv missing required columns: {missing}")
    for column in (*_REQUIRED_COLUMNS, "freq_mhz", "t1_err_us"):
        if column not in samples_df.columns:
            samples_df[column] = np.nan
        try:
            samples_df[column] = pd.to_numeric(samples_df[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"samples.csv column {column} must be numeric") from exc


def _float_column(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=np.float64)
    values = pd.to_numeric(frame[column], errors="coerce")
    return np.asarray(values, dtype=np.float64)


def _prepare_t1_data(
    calibration: T1FluxCalibration,
    *,
    analysis_flux_range: tuple[float, float],
) -> T1CurveData:
    context = calibration.context
    samples_df = context.samples_df

    t1_values = _float_column(samples_df, "t1_us")
    eligible = np.isfinite(t1_values) & (t1_values > 0.0)
    if not np.any(eligible):
        raise ValueError("no finite positive T1 rows are available for analysis")

    raw_fluxs = np.asarray(context.fluxs[eligible], dtype=np.float64)
    f01_observed = _float_column(samples_df, "freq_mhz")
    f01_observed = f01_observed[eligible]
    f01_measured = np.isfinite(f01_observed)
    T1_ns = 1e3 * t1_values[eligible]
    T1err_ns = 1e3 * _float_column(samples_df, "t1_err_us")[eligible]

    aligned_fluxs, shifts, in_window = align_flux_to_window(
        raw_fluxs, analysis_flux_range
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
        f01_mhz=f01_used,
        f01_measured=measured_taken,
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
        f01_mhz=data.f01_mhz[mask],
        f01_measured=data.f01_measured[mask],
        T1_ns=data.T1_ns[mask],
        T1err_ns=data.T1err_ns[mask],
        omegas=data.omegas[mask],
    )


def _prepared_summary_table(
    *,
    calibration: T1FluxCalibration,
    sample: T1CurveData,
    fit: T1CurveData,
    analysis_flux_range: tuple[float, float],
    max_rel_t1_err: float,
    use_weighted_points_only: bool,
) -> pd.DataFrame:
    peak_idx = int(np.nanargmax(sample.T1_ns))
    observed_count = int(np.count_nonzero(sample.f01_measured))
    model_count = len(sample.fluxs) - observed_count
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
            ("max relative T1err", f"{max_rel_t1_err:.3g}"),
            ("use weighted points only", str(use_weighted_points_only)),
            ("window T1 rows", f"{len(sample.fluxs)}/{len(calibration.t1_df)}"),
            ("fit rows", str(len(fit.fluxs))),
            ("f01 source (observed/model)", f"{observed_count}/{model_count}"),
            ("integer shifts", shift_summary),
            (
                "sample peak",
                f"flux={sample.fluxs[peak_idx]:.6f}, T1={1e-3 * sample.T1_ns[peak_idx]:.3f} us",
            ),
        ],
        columns=["metric", "value"],
    )

"""Flux-first SampleMerge: merge v2 sample CSVs into one authoritative target frame.

Each source is a flat v2 CSV (``validate_sample_table_v2``). Per-row flux is
resolved through the shared provenance SSOT ``resolve_sample_flux`` (explicit
``flux`` -> row ``flux_int``/``flux_period`` frame -> caller-declared
``fallback_frame``), shifted by the caller's explicit integer branch offset,
optionally corrected by one small batch offset fitted against the target f01
model, and finally mapped to the target frame's device values. The merged
output is a complete flat v2 coordinate: adjusted ``flux`` plus target
``dev_value``/``dev_unit``/``flux_int``/``flux_period``, followed by the
caller-owned measurement columns. Units and integer branches are never
inferred; source CSVs are never overwritten.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from zcu_tools.meta_tool import (
    DEV_UNIT_COLUMN,
    DEV_VALUE_COLUMN,
    DeviceValueUnit,
    FLUX_COLUMN,
    FLUX_INT_COLUMN,
    FLUX_PERIOD_COLUMN,
    QubitParams,
    SAMPLE_COORDINATE_COLUMNS,
    SampleFluxFrame,
    SampleFluxResolution,
    resolve_sample_flux,
    validate_sample_table_v2,
)

from .flux import predict_f01_mhz

BatchFluxOffsetObjective = Literal["soft_l1", "median_abs", "mean_abs", "rms"]

#: Exact caller-owned measurement column holding the observed f01 used by the
#: optional batch flux offset fit and diagnostics. v2 CSVs carry no aliases;
#: a missing column simply disables the batch fit (no inference).
F01_FREQUENCY_COLUMN = "Freq (MHz)"

_FLUX_SOURCE_LABELS = ("explicit", "row-frame", "fallback-frame")


@dataclass(frozen=True, slots=True)
class FluxFrame:
    """Analysis affine flux frame: f01 model params plus the v2 device frame.

    Construction fast-fails on unsupported A/V ``dev_unit``, non-finite
    ``flux_int`` and non-positive/non-finite ``flux_period``; the f01 model
    ``params`` must be three finite floats. ``flux_from_dev_value`` /
    ``dev_value_from_flux`` provide the scalar/array round trip
    ``(dev_value - flux_int) / flux_period``.
    """

    params: tuple[float, float, float]
    dev_unit: DeviceValueUnit
    flux_int: float
    flux_period: float
    label: str
    _sample_frame: SampleFluxFrame = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.params, tuple)
            or len(self.params) != 3
            or not all(
                isinstance(value, (int, float)) and np.isfinite(value)
                for value in self.params
            )
        ):
            raise ValueError(
                f"params must be a tuple of three finite floats, got {self.params!r}"
            )
        object.__setattr__(
            self,
            "_sample_frame",
            SampleFluxFrame(self.dev_unit, self.flux_int, self.flux_period),
        )

    @classmethod
    def from_result_dir(
        cls,
        result_dir: str | Path,
        *,
        dev_unit: DeviceValueUnit,
        label: str | None = None,
    ) -> Self:
        """Build a frame from a result dir's ``fluxdep_fit``.

        ``dev_unit`` is required because the persisted params carry no unit
        metadata; the caller declares the device's A/V unit explicitly.
        """
        path = Path(result_dir)
        fit = QubitParams.for_result_dir(path, readonly=True).require_fluxdep_fit()
        return cls(
            params=fit.params,
            dev_unit=dev_unit,
            flux_int=fit.flux_int,
            flux_period=fit.flux_period,
            label=label or str(path),
        )

    @overload
    def flux_from_dev_value(self, value: float) -> float: ...

    @overload
    def flux_from_dev_value(
        self, value: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def flux_from_dev_value(
        self, value: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        return self._sample_frame.flux_from_dev_value(value)

    @overload
    def dev_value_from_flux(self, flux: float) -> float: ...

    @overload
    def dev_value_from_flux(self, flux: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def dev_value_from_flux(
        self, flux: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        return self._sample_frame.dev_value_from_flux(flux)


@dataclass(frozen=True, slots=True)
class SampleSource:
    """One v2 sample CSV feeding the merge.

    ``fallback_frame`` resolves migrated rows that carry neither explicit
    ``flux`` nor a row frame (its unit must match the row ``dev_unit``).
    ``integer_flux_offset`` is the caller-declared integer branch shift applied
    to the resolved flux; the merge never infers an integer branch. The
    optional batch offset fields fit one small constant flux offset against the
    target f01 model.
    """

    path: str | Path
    label: str | None = None
    fallback_frame: FluxFrame | None = None
    integer_flux_offset: int = 0
    fit_batch_flux_offset: bool = False
    batch_flux_offset_objective: BatchFluxOffsetObjective = "soft_l1"
    batch_flux_offset_range: tuple[float, float] | None = None
    max_abs_batch_flux_offset: float = 0.03
    f01_fit_scale_mhz: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.integer_flux_offset, int) or isinstance(
            self.integer_flux_offset, bool
        ):
            raise ValueError("integer_flux_offset must be an int")
        if self.batch_flux_offset_objective not in (
            "soft_l1",
            "median_abs",
            "mean_abs",
            "rms",
        ):
            raise ValueError(
                "batch_flux_offset_objective must be soft_l1, median_abs, "
                "mean_abs, or rms"
            )
        if self.batch_flux_offset_range is not None:
            lower, upper = self.batch_flux_offset_range
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(
                    "batch_flux_offset_range must be an increasing finite pair"
                )
        if (
            not np.isfinite(self.max_abs_batch_flux_offset)
            or self.max_abs_batch_flux_offset < 0.0
        ):
            raise ValueError(
                "max_abs_batch_flux_offset must be finite and non-negative"
            )
        if not np.isfinite(self.f01_fit_scale_mhz) or self.f01_fit_scale_mhz <= 0.0:
            raise ValueError("f01_fit_scale_mhz must be positive and finite")


@dataclass(frozen=True, slots=True)
class BatchFluxOffsetResult:
    fitted_flux_offset: float
    objective: BatchFluxOffsetObjective
    finite_f01_rows: int
    success: bool
    cost: float


@dataclass(frozen=True, slots=True)
class SampleMergeResult:
    merged: pd.DataFrame
    diagnostics: pd.DataFrame
    summary_table: pd.DataFrame
    target_frame: FluxFrame


def merge_sample_sources(
    *,
    target_frame: FluxFrame,
    sources: Iterable[SampleSource],
) -> SampleMergeResult:
    """Merge v2 sample tables into one authoritative ``target_frame``.

    The pipeline per source is: resolve flux with shared provenance
    (``resolve_sample_flux``) -> apply the caller-declared
    ``integer_flux_offset`` -> optionally fit one small batch offset against
    the target f01 model -> map to the target frame's ``dev_value``. Units and
    integer branches are never inferred; unresolved rows fail fast with their
    indexes. Source CSVs are never modified.

    The returned ``merged`` table is a complete flat v2 coordinate
    (``flux``, ``dev_value``, ``dev_unit``, ``flux_int``, ``flux_period``)
    followed by the caller-owned measurement columns. Per-row provenance and
    offsets are kept in ``diagnostics`` instead of being written into the
    analysis-facing table.
    """

    merged_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for source in tuple(sources):
        source_path = Path(source.path)
        source_label = source.label or source_path.stem
        raw = pd.read_csv(source_path, encoding="utf-8-sig")
        validate_sample_table_v2(raw, allow_empty=True)
        resolution = resolve_sample_flux(
            raw,
            fallback_frame=_as_sample_frame(source.fallback_frame),
        )
        flux_resolved = resolution.values
        flux_shifted = flux_resolved + float(source.integer_flux_offset)
        f01_mhz = (
            _float_column(raw, F01_FREQUENCY_COLUMN)
            if F01_FREQUENCY_COLUMN in raw.columns
            else None
        )
        batch = _fit_batch_offset(
            flux_shifted,
            f01_mhz,
            target_frame=target_frame,
            source=source,
        )
        flux_final = flux_shifted + batch.fitted_flux_offset
        dev_values = np.asarray(
            target_frame.dev_value_from_flux(flux_final), dtype=np.float64
        )

        merged = _make_output_table(
            raw,
            dev_values=dev_values,
            flux_final=flux_final,
            target_frame=target_frame,
        )
        diagnostics = _make_diagnostics_table(
            raw,
            source=source,
            source_label=source_label,
            source_path=source_path,
            target_frame=target_frame,
            resolution=resolution,
            flux_resolved=flux_resolved,
            flux_final=flux_final,
            dev_values=dev_values,
            batch=batch,
        )
        merged_parts.append(merged)
        diagnostic_parts.append(diagnostics)
        summary_rows.append(
            _summary_row(
                source_label=source_label,
                source=source,
                source_path=source_path,
                target_frame=target_frame,
                raw=raw,
                resolution=resolution,
                diagnostics=diagnostics,
                batch=batch,
            )
        )

    if not merged_parts:
        raise ValueError("At least one SampleSource is required")

    merged_df = pd.concat(merged_parts, ignore_index=True)
    diagnostics_df = pd.concat(diagnostic_parts, ignore_index=True)
    summary_table = pd.DataFrame(summary_rows)
    return SampleMergeResult(
        merged=merged_df,
        diagnostics=diagnostics_df,
        summary_table=summary_table,
        target_frame=target_frame,
    )


def write_merged_samples(
    result: SampleMergeResult,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Write the merged v2 table to a caller-owned path.

    Refuses to overwrite any source CSV that fed the merge.
    """
    output_path = Path(path)
    _require_distinct_from_sources(result, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.merged.to_csv(output_path, index=index)
    return output_path


def write_sample_merge_report(
    result: SampleMergeResult,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    """Write the merge diagnostics report to a caller-owned path.

    Refuses to overwrite any source CSV that fed the merge.
    """
    output_path = Path(path)
    _require_distinct_from_sources(result, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.diagnostics.to_csv(output_path, index=index)
    return output_path


def plot_sample_merge_f01_diagnostics(
    result: SampleMergeResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    merged = result.merged
    if F01_FREQUENCY_COLUMN not in merged.columns:
        raise ValueError(
            f"merged table has no {F01_FREQUENCY_COLUMN!r} column for diagnostics"
        )
    finite = np.isfinite(_float_column(merged, FLUX_COLUMN)) & np.isfinite(
        _float_column(merged, F01_FREQUENCY_COLUMN)
    )
    if not np.any(finite):
        raise ValueError("No finite flux/Freq rows are available for plotting")

    fluxs = _float_column(merged, FLUX_COLUMN)
    f01_mhz = _float_column(merged, F01_FREQUENCY_COLUMN)
    labels = result.diagnostics["source_label"].astype(str)
    target_model = predict_f01_mhz(result.target_frame.params, fluxs)
    residual = f01_mhz - target_model
    flux_min = float(np.nanmin(fluxs[finite]))
    flux_max = float(np.nanmax(fluxs[finite]))
    t_fluxs = np.linspace(flux_min - 0.02, flux_max + 0.02, 700)

    fig, (ax_curve, ax_residual) = plt.subplots(
        2,
        1,
        figsize=(8.0, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax_curve.plot(
        t_fluxs,
        predict_f01_mhz(result.target_frame.params, t_fluxs),
        color="black",
        linewidth=1.5,
        label="target model f01",
    )
    for label in dict.fromkeys(labels[finite]):
        mask = finite & (labels == label).to_numpy()
        ax_curve.scatter(fluxs[mask], f01_mhz[mask], s=24, label=label)
        ax_residual.scatter(fluxs[mask], residual[mask], s=24, label=label)

    ax_curve.set_ylabel("f01 (MHz)")
    ax_curve.set_title("Merged samples in target flux frame")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend(loc="best")

    ax_residual.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_residual.set_xlabel(r"Flux quanta ($\Phi_\mathrm{ext}/\Phi_0$)")
    ax_residual.set_ylabel("measured - model (MHz)")
    ax_residual.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig, (ax_curve, ax_residual)


def _as_sample_frame(frame: FluxFrame | None) -> SampleFluxFrame | None:
    if frame is None:
        return None
    return SampleFluxFrame(frame.dev_unit, frame.flux_int, frame.flux_period)


def _fit_batch_offset(
    flux_shifted: NDArray[np.float64],
    f01_mhz: NDArray[np.float64] | None,
    *,
    target_frame: FluxFrame,
    source: SampleSource,
) -> BatchFluxOffsetResult:
    f01 = (
        f01_mhz
        if f01_mhz is not None
        else np.full(flux_shifted.shape, np.nan, dtype=np.float64)
    )
    finite = np.isfinite(flux_shifted) & np.isfinite(f01)
    if source.batch_flux_offset_range is not None:
        lower, upper = source.batch_flux_offset_range
        fit_axis = flux_shifted
        finite &= (fit_axis >= lower) & (fit_axis <= upper)
    finite_count = int(np.count_nonzero(finite))
    if not source.fit_batch_flux_offset or finite_count == 0:
        return BatchFluxOffsetResult(
            fitted_flux_offset=0.0,
            objective=source.batch_flux_offset_objective,
            finite_f01_rows=finite_count,
            success=True,
            cost=np.nan,
        )

    fit_fluxs = flux_shifted[finite]
    fit_f01 = f01[finite]
    bound = float(source.max_abs_batch_flux_offset)
    f_scale = float(source.f01_fit_scale_mhz)

    def objective(delta: float) -> float:
        residual = predict_f01_mhz(target_frame.params, fit_fluxs + delta)
        residual = residual - fit_f01
        return _batch_flux_offset_cost(
            residual,
            objective=source.batch_flux_offset_objective,
            f01_fit_scale_mhz=f_scale,
        )

    if bound == 0.0:
        fitted = 0.0
        cost = objective(0.0)
        success = True
    else:
        result = minimize_scalar(objective, bounds=(-bound, bound), method="bounded")
        fitted = float(result.x)
        cost = float(result.fun)
        success = bool(result.success)

    return BatchFluxOffsetResult(
        fitted_flux_offset=fitted,
        objective=source.batch_flux_offset_objective,
        finite_f01_rows=finite_count,
        success=success,
        cost=cost,
    )


def _batch_flux_offset_cost(
    residual_mhz: NDArray[np.float64],
    *,
    objective: BatchFluxOffsetObjective,
    f01_fit_scale_mhz: float,
) -> float:
    if residual_mhz.size == 0:
        return np.nan
    scaled = residual_mhz / f01_fit_scale_mhz
    if objective == "soft_l1":
        return float(np.sum(2.0 * (np.sqrt(1.0 + scaled**2) - 1.0)))
    if objective == "median_abs":
        return float(np.nanmedian(np.abs(residual_mhz)))
    if objective == "mean_abs":
        return float(np.nanmean(np.abs(residual_mhz)))
    if objective == "rms":
        return float(np.sqrt(np.nanmean(residual_mhz**2)))
    raise AssertionError(f"unhandled batch_flux_offset_objective {objective!r}")


def _make_output_table(
    raw: pd.DataFrame,
    *,
    dev_values: NDArray[np.float64],
    flux_final: NDArray[np.float64],
    target_frame: FluxFrame,
) -> pd.DataFrame:
    row_count = len(raw)
    output = pd.DataFrame(
        {
            FLUX_COLUMN: np.asarray(flux_final, dtype=np.float64),
            DEV_VALUE_COLUMN: np.asarray(dev_values, dtype=np.float64),
            DEV_UNIT_COLUMN: np.full(row_count, target_frame.dev_unit, dtype=object),
            FLUX_INT_COLUMN: np.full(
                row_count, target_frame.flux_int, dtype=np.float64
            ),
            FLUX_PERIOD_COLUMN: np.full(
                row_count, target_frame.flux_period, dtype=np.float64
            ),
        }
    )
    measurement_columns = [
        column for column in raw.columns if column not in SAMPLE_COORDINATE_COLUMNS
    ]
    for column in measurement_columns:
        output[column] = raw[column].to_numpy()
    return output


def _make_diagnostics_table(
    raw: pd.DataFrame,
    *,
    source: SampleSource,
    source_label: str,
    source_path: Path,
    target_frame: FluxFrame,
    resolution: SampleFluxResolution,
    flux_resolved: NDArray[np.float64],
    flux_final: NDArray[np.float64],
    dev_values: NDArray[np.float64],
    batch: BatchFluxOffsetResult,
) -> pd.DataFrame:
    f01_mhz = (
        _float_column(raw, F01_FREQUENCY_COLUMN)
        if F01_FREQUENCY_COLUMN in raw.columns
        else np.full(len(raw), np.nan, dtype=np.float64)
    )
    before_batch = flux_resolved + float(source.integer_flux_offset)
    residual_before = _predict_residual_or_nan(
        target_frame.params, before_batch, f01_mhz
    )
    residual_after = _predict_residual_or_nan(target_frame.params, flux_final, f01_mhz)
    return pd.DataFrame(
        {
            "source_label": source_label,
            "source_path": str(source_path.resolve()),
            "flux_source": list(resolution.sources),
            "integer_flux_offset": source.integer_flux_offset,
            "batch_flux_offset_objective": batch.objective,
            "fitted_flux_offset": batch.fitted_flux_offset,
            "batch_flux_offset": batch.fitted_flux_offset,
            "row_index": np.arange(len(raw), dtype=np.int64),
            "dev_value": _float_column(raw, DEV_VALUE_COLUMN),
            "dev_unit": raw[DEV_UNIT_COLUMN].astype(str).to_numpy(),
            "flux_resolved": flux_resolved,
            "flux": flux_final,
            "dev_value_target": dev_values,
            "residual_before_offset_MHz": residual_before,
            "residual_after_merge_MHz": residual_after,
        }
    )


def _summary_row(
    *,
    source_label: str,
    source: SampleSource,
    source_path: Path,
    target_frame: FluxFrame,
    raw: pd.DataFrame,
    resolution: SampleFluxResolution,
    diagnostics: pd.DataFrame,
    batch: BatchFluxOffsetResult,
) -> dict[str, object]:
    source_counts = {
        name: sum(1 for item in resolution.sources if item == name)
        for name in _FLUX_SOURCE_LABELS
    }
    flux_resolved = _float_column(diagnostics, "flux_resolved")
    flux_final = _float_column(diagnostics, "flux")
    residual_before = _float_column(diagnostics, "residual_before_offset_MHz")
    residual_after = _float_column(diagnostics, "residual_after_merge_MHz")
    return {
        "source": source_label,
        "path": str(source_path.resolve()),
        "rows": len(raw),
        "explicit_flux_rows": source_counts["explicit"],
        "row_frame_flux_rows": source_counts["row-frame"],
        "fallback_frame_flux_rows": source_counts["fallback-frame"],
        "integer_flux_offset": source.integer_flux_offset,
        "fit_batch_flux_offset": source.fit_batch_flux_offset,
        "batch_flux_offset_objective": batch.objective,
        "batch_flux_offset_range": source.batch_flux_offset_range,
        "fitted_flux_offset": batch.fitted_flux_offset,
        "batch_fit_success": batch.success,
        "batch_fit_cost": batch.cost,
        "finite_f01_rows": batch.finite_f01_rows,
        "target_frame": target_frame.label,
        "flux_resolved_min": _nan_stat(flux_resolved, np.nanmin),
        "flux_resolved_max": _nan_stat(flux_resolved, np.nanmax),
        "merged_flux_min": _nan_stat(flux_final, np.nanmin),
        "merged_flux_max": _nan_stat(flux_final, np.nanmax),
        "residual_median_abs_before_offset_MHz": _finite_abs_median(residual_before),
        "residual_median_abs_after_merge_MHz": _finite_abs_median(residual_after),
    }


def _require_distinct_from_sources(
    result: SampleMergeResult, output_path: Path
) -> None:
    """Raise before writing ``output_path`` if it equals any source CSV path.

    Source paths come from the per-source ``summary_table`` so that header-only
    (zero-row) sources — which have no row diagnostics — are protected too.
    Row-level diagnostic paths are unioned in as defense in depth. Paths are
    resolved so exact/relative/symlink-equivalent spellings cannot bypass the
    check to the extent ``Path.resolve`` already supports.
    """
    resolved_output = output_path.resolve()
    source_paths = {
        Path(value).resolve()
        for value in result.summary_table["path"].astype(str).unique()
    }
    if "source_path" in result.diagnostics.columns:
        source_paths.update(
            Path(value).resolve()
            for value in result.diagnostics["source_path"].astype(str).unique()
        )
    if resolved_output in source_paths:
        raise ValueError(f"refusing to overwrite source CSV {output_path}")


def _predict_residual_or_nan(
    params: tuple[float, float, float],
    fluxs: NDArray[np.float64],
    f01_mhz: NDArray[np.float64],
) -> NDArray[np.float64]:
    residual = np.full_like(f01_mhz, np.nan, dtype=np.float64)
    finite = np.isfinite(fluxs) & np.isfinite(f01_mhz)
    if np.any(finite):
        residual[finite] = f01_mhz[finite] - predict_f01_mhz(params, fluxs[finite])
    return residual


def _float_column(frame: pd.DataFrame, column: str) -> NDArray[np.float64]:
    numeric = cast(pd.Series, pd.to_numeric(_series(frame, column), errors="coerce"))
    return cast(
        NDArray[np.float64],
        numeric.to_numpy(dtype=np.float64),
    )


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, frame[column])


def _nan_stat(
    values: NDArray[np.float64],
    fn: Callable[[NDArray[np.float64]], np.float64],
) -> float:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return np.nan
    return float(fn(values))


def _finite_abs_median(values: NDArray[np.float64]) -> float:
    finite = np.isfinite(values)
    return _nan_stat(np.abs(values[finite]), np.nanmedian)

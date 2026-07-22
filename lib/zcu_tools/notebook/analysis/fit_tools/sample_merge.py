from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from zcu_tools.meta_tool import QubitParams
from zcu_tools.simulate import flux2value, value2flux

from .flux import predict_f01_mhz

SampleCurrentUnit = Literal["mA", "A"]
BatchFluxOffsetReference = Literal["source", "target"]
BatchFluxOffsetObjective = Literal["soft_l1", "median_abs", "mean_abs", "rms"]

_INPUT_COLUMNS = (
    "calibrated mA",
    "Freq (MHz)",
    "T1 (us)",
    "T1err (us)",
    "T2r (us)",
    "T2r err (us)",
    "T2e (us)",
    "T2e err (us)",
    "date",
)
_OUTPUT_COLUMNS = (
    "calibrated mA",
    "Flux",
    "Freq (MHz)",
    "T1 (us)",
    "T1err (us)",
    "T2r (us)",
    "T2r err (us)",
    "T2e (us)",
    "T2e err (us)",
    "date",
)
_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "calibrated mA": ("calibrated mA",),
    "Freq (MHz)": ("Freq (MHz)", "Freq", "f01 (MHz)", "f01"),
    "T1 (us)": ("T1 (us)", "T1"),
    "T1err (us)": ("T1err (us)", "T1 err (us)", "T1err", "T1 err"),
    "T2r (us)": ("T2r (us)", "T2R", "T2r", "T2R (us)"),
    "T2r err (us)": ("T2r err (us)", "T2R err", "T2r err", "T2Rerr"),
    "T2e (us)": ("T2e (us)", "T2E", "T2e", "T2E (us)"),
    "T2e err (us)": ("T2e err (us)", "T2E err", "T2e err", "T2Eerr"),
    "date": ("date", "Date"),
}


@dataclass(frozen=True, slots=True)
class FluxFrame:
    params: tuple[float, float, float]
    flux_half: float
    flux_period: float
    label: str

    @classmethod
    def from_result_dir(
        cls, result_dir: str | Path, *, label: str | None = None
    ) -> Self:
        path = Path(result_dir)
        fit = QubitParams.for_result_dir(path, readonly=True).require_fluxdep_fit()
        return cls(
            params=fit.params,
            flux_half=fit.flux_half,
            flux_period=fit.flux_period,
            label=label or str(path),
        )


@dataclass(frozen=True, slots=True)
class SampleSource:
    path: str | Path
    unit: SampleCurrentUnit = "mA"
    label: str | None = None
    source_result_dir: str | Path | None = None
    source_frame: FluxFrame | None = None
    current_scale_to_source_frame: float = 1.0
    fit_batch_flux_offset: bool = False
    batch_flux_offset_reference: BatchFluxOffsetReference = "source"
    batch_flux_offset_objective: BatchFluxOffsetObjective = "soft_l1"
    batch_flux_offset_range: tuple[float, float] | None = None
    manual_flux_offset: float = 0.0
    max_abs_batch_flux_offset: float = 0.03
    f01_fit_scale_mhz: float = 20.0

    def __post_init__(self) -> None:
        if self.unit not in ("mA", "A"):
            raise ValueError("unit must be 'mA' or 'A'")
        if self.source_result_dir is not None and self.source_frame is not None:
            raise ValueError("Use either source_result_dir or source_frame, not both")
        if self.batch_flux_offset_reference not in ("source", "target"):
            raise ValueError("batch_flux_offset_reference must be 'source' or 'target'")
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
            not np.isfinite(self.current_scale_to_source_frame)
            or self.current_scale_to_source_frame <= 0.0
        ):
            raise ValueError(
                "current_scale_to_source_frame must be positive and finite"
            )
        if not np.isfinite(self.manual_flux_offset):
            raise ValueError("manual_flux_offset must be finite")
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
    manual_flux_offset: float
    total_flux_offset: float
    reference: BatchFluxOffsetReference
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
    target_result_dir: str | Path,
    sources: Iterable[SampleSource],
    target_frame: FluxFrame | None = None,
) -> SampleMergeResult:
    """Merge raw sample tables into one canonical target-frame ``samples.csv``.

    The returned ``merged`` table is intentionally small and compatible with the
    T1/T2 curve notebooks. Source provenance and batch diagnostics are kept in
    ``diagnostics`` instead of being written into the analysis-facing sample table.
    """

    target_path = Path(target_result_dir)
    resolved_target_frame = target_frame or FluxFrame.from_result_dir(
        target_path, label=str(target_path)
    )
    merged_parts: list[pd.DataFrame] = []
    diagnostic_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for source in tuple(sources):
        source_frame = _resolve_source_frame(source, resolved_target_frame)
        source_path = _resolve_source_path(source.path, target_path)
        source_label = source.label or source_path.stem
        raw = pd.read_csv(source_path, encoding="utf-8-sig")
        normalized = _canonicalize_sample_columns(raw, source_label=source_label)
        source_values = (
            _float_column(normalized, "calibrated mA")
            * source.current_scale_to_source_frame
        )
        source_flux_raw = np.asarray(
            value2flux(source_values, source_frame.flux_half, source_frame.flux_period),
            dtype=np.float64,
        )
        f01_mhz = _float_column(normalized, "Freq (MHz)")
        batch = _fit_batch_offset(
            source_flux_raw,
            f01_mhz,
            source_frame=source_frame,
            target_frame=resolved_target_frame,
            source=source,
        )
        source_flux = source_flux_raw + batch.total_flux_offset
        canonical_values = np.asarray(
            flux2value(
                source_flux,
                resolved_target_frame.flux_half,
                resolved_target_frame.flux_period,
            ),
            dtype=np.float64,
        )

        merged = _make_output_table(normalized, canonical_values, source_flux)
        diagnostics = _make_diagnostics_table(
            normalized,
            source=source,
            source_label=source_label,
            source_path=source_path,
            source_frame=source_frame,
            target_frame=resolved_target_frame,
            source_values=source_values,
            source_flux_raw=source_flux_raw,
            source_flux=source_flux,
            canonical_values=canonical_values,
            batch=batch,
        )
        merged_parts.append(merged)
        diagnostic_parts.append(diagnostics)
        summary_rows.append(
            _summary_row(
                source_label=source_label,
                source=source,
                source_path=source_path,
                source_frame=source_frame,
                normalized=normalized,
                source_flux_raw=source_flux_raw,
                source_flux=source_flux,
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
        target_frame=resolved_target_frame,
    )


def write_merged_samples(
    result: SampleMergeResult,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.merged.to_csv(output_path, index=index)
    return output_path


def write_sample_merge_report(
    result: SampleMergeResult,
    path: str | Path,
    *,
    index: bool = False,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.diagnostics.to_csv(output_path, index=index)
    return output_path


def plot_sample_merge_f01_diagnostics(
    result: SampleMergeResult,
) -> tuple[Figure, tuple[Axes, Axes]]:
    finite = np.isfinite(_float_column(result.merged, "Flux")) & np.isfinite(
        _float_column(result.merged, "Freq (MHz)")
    )
    if not np.any(finite):
        raise ValueError("No finite Flux/Freq rows are available for plotting")

    fluxs = _float_column(result.merged, "Flux")
    f01_mhz = _float_column(result.merged, "Freq (MHz)")
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


def _resolve_source_frame(source: SampleSource, target_frame: FluxFrame) -> FluxFrame:
    if source.source_frame is not None:
        return source.source_frame
    if source.source_result_dir is not None:
        return FluxFrame.from_result_dir(source.source_result_dir)
    return target_frame


def _resolve_source_path(path: str | Path, target_result_dir: Path) -> Path:
    source_path = Path(path)
    if source_path.is_absolute():
        return source_path
    candidate = target_result_dir / source_path
    return candidate if candidate.exists() else source_path


def _fit_batch_offset(
    source_flux_raw: NDArray[np.float64],
    f01_mhz: NDArray[np.float64],
    *,
    source_frame: FluxFrame,
    target_frame: FluxFrame,
    source: SampleSource,
) -> BatchFluxOffsetResult:
    finite = np.isfinite(source_flux_raw) & np.isfinite(f01_mhz)
    if source.batch_flux_offset_range is not None:
        lower, upper = source.batch_flux_offset_range
        fit_axis = source_flux_raw + float(source.manual_flux_offset)
        finite &= (fit_axis >= lower) & (fit_axis <= upper)
    finite_count = int(np.count_nonzero(finite))
    manual = float(source.manual_flux_offset)
    if not source.fit_batch_flux_offset or finite_count == 0:
        return BatchFluxOffsetResult(
            fitted_flux_offset=0.0,
            manual_flux_offset=manual,
            total_flux_offset=manual,
            reference=source.batch_flux_offset_reference,
            objective=source.batch_flux_offset_objective,
            finite_f01_rows=finite_count,
            success=True,
            cost=np.nan,
        )

    fit_fluxs = source_flux_raw[finite]
    fit_f01 = f01_mhz[finite]
    bound = float(source.max_abs_batch_flux_offset)
    f_scale = float(source.f01_fit_scale_mhz)
    fit_frame = (
        target_frame if source.batch_flux_offset_reference == "target" else source_frame
    )

    def objective(delta: float) -> float:
        residual = predict_f01_mhz(fit_frame.params, fit_fluxs + manual + delta)
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
        manual_flux_offset=manual,
        total_flux_offset=manual + fitted,
        reference=source.batch_flux_offset_reference,
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


def _canonicalize_sample_columns(
    raw: pd.DataFrame, *, source_label: str
) -> pd.DataFrame:
    normalized_names = {
        _normalize_column_name(column): str(column) for column in raw.columns
    }
    columns: dict[str, pd.Series] = {}
    for canonical in _INPUT_COLUMNS:
        matches: list[str] = [
            normalized_names[name]
            for alias in _COLUMN_ALIASES[canonical]
            if (name := _normalize_column_name(alias)) in normalized_names
        ]
        if canonical == "date":
            columns[canonical] = _coalesce_object_columns(raw, matches)
        else:
            columns[canonical] = _coalesce_numeric_columns(
                raw,
                matches,
                canonical=canonical,
                source_label=source_label,
            )
    if not np.any(np.isfinite(columns["calibrated mA"].to_numpy(dtype=np.float64))):
        raise ValueError(f"{source_label}: missing finite 'calibrated mA' rows")
    return pd.DataFrame(columns)


def _normalize_column_name(column: object) -> str:
    return (
        str(column)
        .replace("\ufeff", "")
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _coalesce_numeric_columns(
    raw: pd.DataFrame,
    matches: list[str],
    *,
    canonical: str,
    source_label: str,
) -> pd.Series:
    if not matches:
        return pd.Series(np.nan, index=raw.index, dtype=np.float64)

    result = cast(
        pd.Series, pd.to_numeric(_series(raw, matches[0]), errors="coerce")
    ).astype(float)
    for column in matches[1:]:
        values = cast(
            pd.Series, pd.to_numeric(_series(raw, column), errors="coerce")
        ).astype(float)
        overlap = result.notna() & values.notna()
        conflict = overlap & ~np.isclose(result, values, rtol=0.0, atol=1e-12)
        if bool(conflict.any()):
            raise ValueError(
                f"{source_label}: conflicting aliases for {canonical!r}: "
                f"{matches[0]!r} and {column!r}"
            )
        result = result.combine_first(values)
    return result


def _coalesce_object_columns(raw: pd.DataFrame, matches: list[str]) -> pd.Series:
    if not matches:
        return pd.Series([pd.NA] * len(raw), index=raw.index, dtype="object")
    result = _series(raw, matches[0]).astype("object")
    for column in matches[1:]:
        values = _series(raw, column).astype("object")
        result = result.where(result.notna(), values)
    return result


def _make_output_table(
    normalized: pd.DataFrame,
    canonical_values: NDArray[np.float64],
    canonical_fluxs: NDArray[np.float64],
) -> pd.DataFrame:
    output = normalized.copy()
    output["calibrated mA"] = canonical_values
    output["Flux"] = canonical_fluxs
    return output.loc[:, _OUTPUT_COLUMNS]


def _make_diagnostics_table(
    normalized: pd.DataFrame,
    *,
    source: SampleSource,
    source_label: str,
    source_path: Path,
    source_frame: FluxFrame,
    target_frame: FluxFrame,
    source_values: NDArray[np.float64],
    source_flux_raw: NDArray[np.float64],
    source_flux: NDArray[np.float64],
    canonical_values: NDArray[np.float64],
    batch: BatchFluxOffsetResult,
) -> pd.DataFrame:
    f01_mhz = _float_column(normalized, "Freq (MHz)")
    source_before = _predict_residual_or_nan(
        source_frame.params,
        source_flux_raw + batch.manual_flux_offset,
        f01_mhz,
    )
    source_after = _predict_residual_or_nan(source_frame.params, source_flux, f01_mhz)
    target_before = _predict_residual_or_nan(
        target_frame.params,
        source_flux_raw + batch.manual_flux_offset,
        f01_mhz,
    )
    target_after = _predict_residual_or_nan(target_frame.params, source_flux, f01_mhz)
    return pd.DataFrame(
        {
            "source_label": source_label,
            "source_path": str(source_path),
            "source_unit": source.unit,
            "source_frame": source_frame.label,
            "batch_flux_offset_reference": batch.reference,
            "batch_flux_offset_objective": batch.objective,
            "row_index": np.arange(len(normalized), dtype=np.int64),
            "source_value": source_values,
            "source_flux_raw": source_flux_raw,
            "batch_flux_offset": batch.total_flux_offset,
            "Flux": source_flux,
            "calibrated mA": canonical_values,
            "Freq (MHz)": f01_mhz,
            "source_residual_before_offset_MHz": source_before,
            "source_residual_after_offset_MHz": source_after,
            "target_residual_before_offset_MHz": target_before,
            "target_residual_after_merge_MHz": target_after,
        }
    )


def _summary_row(
    *,
    source_label: str,
    source: SampleSource,
    source_path: Path,
    source_frame: FluxFrame,
    normalized: pd.DataFrame,
    source_flux_raw: NDArray[np.float64],
    source_flux: NDArray[np.float64],
    diagnostics: pd.DataFrame,
    batch: BatchFluxOffsetResult,
) -> dict[str, object]:
    finite_flux = np.isfinite(source_flux)
    source_residual_before = _float_column(
        diagnostics, "source_residual_before_offset_MHz"
    )
    source_residual_after = _float_column(
        diagnostics, "source_residual_after_offset_MHz"
    )
    target_residual_before = _float_column(
        diagnostics, "target_residual_before_offset_MHz"
    )
    target_residual_after = _float_column(
        diagnostics, "target_residual_after_merge_MHz"
    )
    return {
        "source": source_label,
        "path": str(source_path),
        "unit": source.unit,
        "source_frame": source_frame.label,
        "rows": len(normalized),
        "finite_f01_rows": batch.finite_f01_rows,
        "fit_batch_flux_offset": source.fit_batch_flux_offset,
        "batch_flux_offset_reference": batch.reference,
        "batch_flux_offset_objective": batch.objective,
        "batch_flux_offset_range": source.batch_flux_offset_range,
        "manual_flux_offset": batch.manual_flux_offset,
        "fitted_flux_offset": batch.fitted_flux_offset,
        "total_flux_offset": batch.total_flux_offset,
        "batch_fit_success": batch.success,
        "batch_fit_cost": batch.cost,
        "raw_flux_min": _nan_stat(source_flux_raw, np.nanmin),
        "raw_flux_max": _nan_stat(source_flux_raw, np.nanmax),
        "merged_flux_min": _nan_stat(source_flux[finite_flux], np.nanmin),
        "merged_flux_max": _nan_stat(source_flux[finite_flux], np.nanmax),
        "source_residual_median_abs_before_offset_MHz": _finite_abs_median(
            source_residual_before
        ),
        "source_residual_median_abs_after_offset_MHz": _finite_abs_median(
            source_residual_after
        ),
        "target_residual_median_abs_before_offset_MHz": _finite_abs_median(
            target_residual_before
        ),
        "target_residual_median_abs_MHz": _finite_abs_median(target_residual_after),
    }


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

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

ErrorNanPolicy = Literal["unweighted", "global_median", "bin_median"]
FluxWeightingMode = Literal["sample", "equal_flux_bin"]


@dataclass(frozen=True, slots=True)
class MeasurementErrorPolicy:
    nan_policy: ErrorNanPolicy = "unweighted"
    absolute_floor: float = 0.0
    relative_floor: float = 0.0
    fallback_error: float | None = None


@dataclass(frozen=True, slots=True)
class ErrorResolutionResult:
    raw_errors: NDArray[np.float64]
    effective_errors: NDArray[np.float64]
    nan_mask: NDArray[np.bool_]
    bin_fill_mask: NDArray[np.bool_]
    global_fill_mask: NDArray[np.bool_]
    fallback_fill_mask: NDArray[np.bool_]
    floor_mask: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FluxResidualWeighting:
    mode: FluxWeightingMode = "sample"
    bin_width: float | None = None
    bin_count: int | None = None
    origin: float | None = None


@dataclass(frozen=True, slots=True)
class FluxResidualWeights:
    residual_weights: NDArray[np.float64]
    bin_indices: NDArray[np.int64]
    bin_counts: NDArray[np.int64]
    effective_observation_count: float
    mode: FluxWeightingMode
    bin_width: float | None
    bin_count: int | None


def build_flux_residual_weights(
    fluxs: NDArray[np.float64] | None,
    policy: FluxResidualWeighting | None = None,
    *,
    sample_count: int | None = None,
) -> FluxResidualWeights:
    resolved_policy = policy or FluxResidualWeighting()
    _validate_flux_weighting(resolved_policy)

    if fluxs is None:
        if sample_count is None:
            raise ValueError("sample_count is required when fluxs is None")
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if resolved_policy.mode != "sample":
            raise ValueError("fluxs are required for equal_flux_bin weighting")
        bin_indices = np.arange(sample_count, dtype=np.int64)
        bin_counts = np.ones(sample_count, dtype=np.int64)
        return FluxResidualWeights(
            residual_weights=np.ones(sample_count, dtype=np.float64),
            bin_indices=bin_indices,
            bin_counts=bin_counts,
            effective_observation_count=float(sample_count),
            mode=resolved_policy.mode,
            bin_width=resolved_policy.bin_width,
            bin_count=resolved_policy.bin_count,
        )

    flux_arr = np.asarray(fluxs, dtype=np.float64)
    if flux_arr.ndim != 1:
        raise ValueError("fluxs must be a 1D array")
    if flux_arr.size == 0:
        raise ValueError("at least one flux sample is required")
    if not np.all(np.isfinite(flux_arr)):
        raise ValueError("fluxs must be finite")

    if resolved_policy.mode == "sample":
        bin_indices = np.arange(flux_arr.size, dtype=np.int64)
        bin_counts = np.ones(flux_arr.size, dtype=np.int64)
        weights = np.ones(flux_arr.size, dtype=np.float64)
        effective_n = float(flux_arr.size)
    else:
        bin_indices = _equal_flux_bin_indices(flux_arr, resolved_policy)
        unique_bins, counts = np.unique(bin_indices, return_counts=True)
        count_map = dict(zip(unique_bins.tolist(), counts.tolist(), strict=True))
        bin_counts = np.asarray(
            [count_map[int(idx)] for idx in bin_indices], dtype=np.int64
        )
        weights = 1.0 / np.sqrt(bin_counts.astype(np.float64))
        effective_n = float(len(unique_bins))

    return FluxResidualWeights(
        residual_weights=weights,
        bin_indices=bin_indices,
        bin_counts=bin_counts,
        effective_observation_count=effective_n,
        mode=resolved_policy.mode,
        bin_width=resolved_policy.bin_width,
        bin_count=resolved_policy.bin_count,
    )


def resolve_measurement_errors(
    values: NDArray[np.float64],
    errors: NDArray[np.float64],
    *,
    policy: MeasurementErrorPolicy | None = None,
    flux_weights: FluxResidualWeights | None = None,
    name: str = "errors",
) -> ErrorResolutionResult:
    resolved_policy = policy or MeasurementErrorPolicy()
    _validate_error_policy(resolved_policy)

    values_arr = np.asarray(values, dtype=np.float64)
    errors_arr = np.asarray(errors, dtype=np.float64)
    if values_arr.shape != errors_arr.shape:
        raise ValueError(f"{name} must have the same shape as values")
    if values_arr.ndim != 1:
        raise ValueError("values and errors must be 1D arrays")
    if not np.all(np.isfinite(values_arr)):
        raise ValueError("values must be finite")

    valid = np.isnan(errors_arr) | (np.isfinite(errors_arr) & (errors_arr > 0.0))
    if not np.all(valid):
        raise ValueError(f"{name} must be positive finite values or NaN")

    floor = np.maximum(
        float(resolved_policy.absolute_floor),
        float(resolved_policy.relative_floor) * np.abs(values_arr),
    )
    effective = errors_arr.copy()
    finite_mask = np.isfinite(effective)
    effective[finite_mask] = np.maximum(effective[finite_mask], floor[finite_mask])

    nan_mask = np.isnan(effective)
    bin_fill_mask = np.zeros_like(nan_mask, dtype=bool)
    global_fill_mask = np.zeros_like(nan_mask, dtype=bool)
    fallback_fill_mask = np.zeros_like(nan_mask, dtype=bool)

    if np.any(nan_mask) and resolved_policy.nan_policy != "unweighted":
        global_value, global_used_fallback = _global_fill_value(
            effective,
            floor,
            resolved_policy,
            name,
        )
        if resolved_policy.nan_policy == "global_median":
            effective[nan_mask] = np.maximum(global_value, floor[nan_mask])
            if global_used_fallback:
                fallback_fill_mask[nan_mask] = True
            else:
                global_fill_mask[nan_mask] = True
        elif resolved_policy.nan_policy == "bin_median":
            if flux_weights is None:
                raise ValueError("flux_weights are required for bin_median error fill")
            for bin_index in np.unique(flux_weights.bin_indices[nan_mask]):
                in_bin = flux_weights.bin_indices == bin_index
                known = effective[in_bin & np.isfinite(effective)]
                fill_value = float(np.nanmedian(known)) if known.size else global_value
                fill_mask = nan_mask & in_bin
                effective[fill_mask] = np.maximum(fill_value, floor[fill_mask])
                if known.size:
                    bin_fill_mask[fill_mask] = True
                elif global_used_fallback:
                    fallback_fill_mask[fill_mask] = True
                else:
                    global_fill_mask[fill_mask] = True
        else:
            raise ValueError(f"unknown error nan policy: {resolved_policy.nan_policy}")

    floor_mask = np.isfinite(effective) & (effective <= floor) & (floor > 0.0)
    return ErrorResolutionResult(
        raw_errors=errors_arr,
        effective_errors=effective,
        nan_mask=nan_mask,
        bin_fill_mask=bin_fill_mask,
        global_fill_mask=global_fill_mask,
        fallback_fill_mask=fallback_fill_mask,
        floor_mask=floor_mask,
    )


def _validate_error_policy(policy: MeasurementErrorPolicy) -> None:
    if policy.nan_policy not in ("unweighted", "global_median", "bin_median"):
        raise ValueError(f"unknown error nan policy: {policy.nan_policy}")
    if not np.isfinite(policy.absolute_floor) or policy.absolute_floor < 0.0:
        raise ValueError("absolute_error_floor must be finite and non-negative")
    if not np.isfinite(policy.relative_floor) or policy.relative_floor < 0.0:
        raise ValueError("relative_error_floor must be finite and non-negative")
    if policy.fallback_error is not None and (
        not np.isfinite(policy.fallback_error) or policy.fallback_error <= 0.0
    ):
        raise ValueError("fallback_error must be positive and finite")


def _validate_flux_weighting(policy: FluxResidualWeighting) -> None:
    if policy.mode not in ("sample", "equal_flux_bin"):
        raise ValueError(f"unknown flux weighting mode: {policy.mode}")
    if policy.bin_width is not None and (
        not np.isfinite(policy.bin_width) or policy.bin_width <= 0.0
    ):
        raise ValueError("flux bin_width must be positive and finite")
    if policy.bin_count is not None and policy.bin_count <= 0:
        raise ValueError("flux bin_count must be positive")
    if policy.bin_width is not None and policy.bin_count is not None:
        raise ValueError("set only one of bin_width or bin_count")
    if policy.origin is not None and not np.isfinite(policy.origin):
        raise ValueError("flux bin origin must be finite")
    if policy.mode == "sample" and (
        policy.bin_width is not None or policy.bin_count is not None
    ):
        raise ValueError("sample flux weighting does not use bins")


def _equal_flux_bin_indices(
    fluxs: NDArray[np.float64],
    policy: FluxResidualWeighting,
) -> NDArray[np.int64]:
    if policy.bin_width is not None:
        origin = float(np.min(fluxs) if policy.origin is None else policy.origin)
        return np.floor((fluxs - origin) / policy.bin_width).astype(np.int64)

    if policy.bin_count is None:
        raise ValueError("equal_flux_bin weighting requires bin_width or bin_count")

    lower = float(np.min(fluxs) if policy.origin is None else policy.origin)
    upper = float(np.max(fluxs))
    if upper <= lower:
        return np.zeros_like(fluxs, dtype=np.int64)
    width = (upper - lower) / float(policy.bin_count)
    indices = np.floor((fluxs - lower) / width).astype(np.int64)
    return np.clip(indices, 0, policy.bin_count - 1).astype(np.int64)


def _global_fill_value(
    effective_errors: NDArray[np.float64],
    floor: NDArray[np.float64],
    policy: MeasurementErrorPolicy,
    name: str,
) -> tuple[float, bool]:
    known = effective_errors[np.isfinite(effective_errors)]
    if known.size:
        return float(np.nanmedian(known)), False
    if policy.fallback_error is not None:
        return float(policy.fallback_error), True
    positive_floor = floor[np.isfinite(floor) & (floor > 0.0)]
    if positive_floor.size:
        return float(np.nanmedian(positive_floor)), True
    raise ValueError(
        f"{name} contains NaN values but no finite errors are available for filling"
    )

"""Derive the measured and full-work denominators from an experiment Result."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from pydantic import JsonValue

from tools._experiment_wall_time_inputs import (
    EvaluationMetric,
    Extrapolation,
    ResultAxis,
    SpeedMetric,
)
from tools._experiment_wall_time_report import ExtrapolationFactors, Row


def _result_array(result: object, field: str) -> np.ndarray:
    values: object = getattr(result, field, None)
    if not isinstance(values, np.ndarray):
        raise ValueError(f"Result field {field!r} is not a NumPy array")
    return values


def _axis_length(result: object, axis: ResultAxis) -> int:
    values = _result_array(result, axis.field)
    if axis.dimension >= values.ndim:
        raise ValueError(
            f"Result axis {axis.field!r} has no dimension {axis.dimension}"
        )
    length = values.shape[axis.dimension]
    if length == 0:
        raise ValueError(f"Result axis {axis.field!r} is empty")
    return length


def _record_single_acquire_evaluations(row: Row, candidates: np.ndarray) -> None:
    if candidates.ndim != 2 or not candidates.shape[1]:
        raise ValueError("Result candidates must be rows of parameters")
    unreached = int(np.count_nonzero(np.all(np.isnan(candidates), axis=1)))
    entered = len(candidates) - unreached
    row["rejected_evaluations"] = 0
    row["unreached_evaluations"] = unreached
    observed = row["acquires"]
    if observed == entered:
        row["point_count"] = observed
    else:
        row["workload_issue"] = (
            f"Entered evaluations ({entered}) differ from observed acquires ({observed})"
        )


def _record_evaluations(row: Row, result: object, metric: EvaluationMetric) -> None:
    if (candidate_field := metric.single_acquire_candidate_field) is not None:
        _record_single_acquire_evaluations(row, _result_array(result, candidate_field))
        return
    if metric.peak_field is None or metric.counts_field is None:
        raise ValueError(
            "Peak-gated evaluation requires the Result peak and counts fields"
        )
    peak = _result_array(result, metric.peak_field)
    counts = _result_array(result, metric.counts_field)
    if peak.ndim != 1 or not counts.ndim or counts.shape[0] != len(peak):
        raise ValueError("Result peaks and counts must share a candidate axis")
    # In an on-time successful CZOptimizeResult, a recorded peak without
    # seed counts means the experiment skipped that candidate at its peak check.
    acquired = np.any(~np.isnan(counts), axis=tuple(range(1, counts.ndim)))
    reached = ~np.isnan(peak)
    row["point_count"] = int(np.count_nonzero(acquired))
    row["rejected_evaluations"] = int(np.count_nonzero(reached & ~acquired))
    row["unreached_evaluations"] = int(np.count_nonzero(~reached))


def _extrapolation_factors(
    projection: Extrapolation,
    measured_axes: list[int],
    points: int | None,
    overrides: Mapping[tuple[str | int, ...], JsonValue],
) -> ExtrapolationFactors:
    axis_scale = rounds_scale = 1.0
    if (outer := projection.outer_axis) is not None:
        measured_outer = measured_axes[0]
        if outer.full_points <= measured_outer:
            raise ValueError("full_points must exceed the measured outer axis length")
        axis_scale = outer.full_points / measured_outer
        points = outer.full_points * math.prod(measured_axes[1:])
    if (rounds := projection.rounds) is not None:
        rounds_scale = rounds.scale(overrides[rounds.path])
    return {
        "axis_scale": axis_scale,
        "rounds_scale": rounds_scale,
        "point_count": points,
    }


def record_workload(
    row: Row,
    result: object,
    metric: SpeedMetric,
    projection: Extrapolation | None,
    overrides: Mapping[tuple[str | int, ...], JsonValue],
) -> None:
    """Attach workload facts to an on-time successful Run with core facts filled.

    Use the published experiment Result for scans and evaluations. A shots Run
    may pass None as its Result. Use the validated scene's Form writes for round
    scaling. Raise for invalid Result axes or inconsistent full counts. Attach
    extrapolation factors only with complete core seconds and any needed points.
    """
    axes: list[int] = []
    if metric.kind == "scan":
        axes = [_axis_length(result, axis) for axis in metric.axes]
        row["point_count"] = math.prod(axes)
    elif metric.kind == "evaluation":
        _record_evaluations(row, result, metric)
    if (
        projection is not None
        and row["core_s"] is not None
        and (metric.kind == "shots" or row["point_count"])
    ):
        row["extrapolation"] = _extrapolation_factors(
            projection, axes, row["point_count"], overrides
        )

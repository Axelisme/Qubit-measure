"""Timing primitives shared by autofluxdep experiment nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def pop_sweep_range(
    raw_cfg: dict[str, Any], key: str, *, node_name: str
) -> tuple[float, float]:
    return pop_sweep_ranges(raw_cfg, (key,), node_name=node_name)[key]


def pop_sweep_ranges(
    raw_cfg: dict[str, Any], keys: tuple[str, ...], *, node_name: str
) -> dict[str, tuple[float, float]]:
    sweep = raw_cfg.pop("sweep", None)
    if not isinstance(sweep, dict):
        raise RuntimeError(f"{node_name} raw cfg has no sweep section")
    ranges: dict[str, tuple[float, float]] = {}
    for key in keys:
        if key not in sweep:
            raise RuntimeError(f"{node_name} raw cfg has no sweep.{key}")
        ranges[key] = _raw_range_tuple(sweep[key])
    return ranges


def _raw_range_tuple(value: Any) -> tuple[float, float]:
    if hasattr(value, "start") and hasattr(value, "stop"):
        return (float(value.start), float(value.stop))
    lo, hi = value
    return (float(lo), float(hi))


def times_to_cycles_and_axis(
    soccfg: Any, times: NDArray[np.float64]
) -> tuple[list[int], NDArray[np.float64]]:
    """Quantize a requested delay axis to hardware cycles and return actual times."""
    cycles = [int(soccfg.us2cycles(float(time))) for time in times]
    if any(right <= left for left, right in zip(cycles, cycles[1:], strict=False)):
        raise ValueError(
            "delay sweep collapsed after cycle quantization; "
            "reduce expts or widen the delay sweep"
        )
    actual_times = np.asarray(
        [soccfg.cycles2us(int(cycle)) for cycle in cycles], dtype=np.float64
    )
    return cycles, actual_times


__all__ = [
    "pop_sweep_range",
    "pop_sweep_ranges",
    "times_to_cycles_and_axis",
]

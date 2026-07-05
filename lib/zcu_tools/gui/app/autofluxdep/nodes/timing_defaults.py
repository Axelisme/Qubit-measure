"""Timing seed and range helpers shared by autofluxdep nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zcu_tools.gui.app.autofluxdep.nodes.defaults import ctx_md_float


def seed_md_float(ctx: Any | None, key: str, fallback: float) -> float:
    return ctx_md_float(ctx, key) or fallback


def snapshot_float(snapshot: Mapping[str, Any], key: str, fallback: float) -> float:
    value = snapshot.get(key)
    if value is None:
        return fallback
    return float(value)


def fixed_sweep_range(sweep: Any) -> tuple[float, float]:
    return (float(sweep.start), float(sweep.stop))


def auto_relax_delay_from_t1(
    t1: float, *, factor: float, minimum: float | None
) -> float:
    value = float(factor) * float(t1)
    if minimum is None:
        return value
    return max(float(minimum), value)


def auto_stop_sweep_range(
    seed: float, *, start: float, stop_factor: float, stop_min: float | None
) -> tuple[float, float]:
    stop = float(stop_factor) * float(seed)
    if stop_min is not None:
        stop = max(float(stop_min), stop)
    return (float(start), stop)

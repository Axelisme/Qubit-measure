from __future__ import annotations

import math

from .adapter.types import EvalValue, SweepValue


class SweepEditor:
    """Pure canonical transformation rules for a single sweep axis."""

    @staticmethod
    def canonicalize(value: SweepValue) -> SweepValue:
        bounds = SweepEditor._numeric_bounds(value)
        if bounds is None:
            return value
        start, stop = bounds
        return SweepValue(
            start=value.start,
            stop=value.stop,
            expts=value.expts,
            step=SweepEditor._step_from_expts(start, stop, value.expts),
        )

    @staticmethod
    def update_start(value: SweepValue, start: float | EvalValue) -> SweepValue:
        return SweepEditor.canonicalize(
            SweepValue(start=start, stop=value.stop, expts=value.expts, step=value.step)
        )

    @staticmethod
    def update_stop(value: SweepValue, stop: float | EvalValue) -> SweepValue:
        return SweepEditor.canonicalize(
            SweepValue(start=value.start, stop=stop, expts=value.expts, step=value.step)
        )

    @staticmethod
    def update_expts(value: SweepValue, expts: int) -> SweepValue:
        return SweepEditor.canonicalize(
            SweepValue(start=value.start, stop=value.stop, expts=expts, step=value.step)
        )

    @staticmethod
    def update_step(value: SweepValue, step: float) -> SweepValue:
        if not math.isfinite(step):
            raise ValueError("Sweep step must be finite")
        bounds = SweepEditor._numeric_bounds(value)
        if bounds is None:
            return value
        start, stop = bounds
        expts = 1 if step == 0.0 else max(1, round((stop - start) / step + 1))
        return SweepEditor.canonicalize(
            SweepValue(start=value.start, stop=value.stop, expts=expts, step=step)
        )

    @staticmethod
    def _numeric_bounds(value: SweepValue) -> tuple[float, float] | None:
        start = SweepEditor._resolved_edge(value.start)
        stop = SweepEditor._resolved_edge(value.stop)
        if start is None or stop is None:
            return None
        return start, stop

    @staticmethod
    def _resolved_edge(value: float | EvalValue) -> float | None:
        resolved = value.resolved if isinstance(value, EvalValue) else value
        if resolved is None:
            return None
        numeric = float(resolved)
        if not math.isfinite(numeric):
            raise ValueError("Sweep bounds must be finite")
        return numeric

    @staticmethod
    def _step_from_expts(start: float, stop: float, expts: int) -> float:
        return 0.0 if expts == 1 else (stop - start) / (expts - 1)

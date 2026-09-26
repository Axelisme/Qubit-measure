from __future__ import annotations

import math
from dataclasses import replace

from ..model import (
    CenteredSweepValue,
    DirectValue,
    EvalValue,
    ScalarValue,
    SweepValue,
    resolved_direct_number,
)


def _canonical_step_input(
    current: float | DirectValue, derived: float | None
) -> float | DirectValue:
    if (
        isinstance(current, DirectValue)
        and current.raw is not None
        and (current.value is None or derived is None or current.value == derived)
    ):
        return current
    return DirectValue(None) if derived is None else derived


def _resolved_step_input(
    source: float | DirectValue, resolved: float
) -> float | DirectValue:
    if isinstance(source, DirectValue):
        return DirectValue(resolved, raw=source.raw)
    return resolved


class SweepEditor:
    """Pure canonical transformation rules for a single sweep axis."""

    @staticmethod
    def canonicalize(value: SweepValue) -> SweepValue:
        bounds = SweepEditor._numeric_bounds(value)
        points = resolved_direct_number(value.expts)
        derived = None
        if bounds is not None and points is not None:
            derived = SweepEditor._step_from_expts(*bounds, int(points))
        return replace(
            value, step=_canonical_step_input(value.step, derived), auto_norm=False
        )

    @staticmethod
    def update_start(value: SweepValue, start: float | ScalarValue) -> SweepValue:
        return SweepEditor.canonicalize(replace(value, start=start, auto_norm=False))

    @staticmethod
    def update_stop(value: SweepValue, stop: float | ScalarValue) -> SweepValue:
        return SweepEditor.canonicalize(replace(value, stop=stop, auto_norm=False))

    @staticmethod
    def update_expts(value: SweepValue, expts: int | DirectValue) -> SweepValue:
        return SweepEditor.canonicalize(
            replace(value, expts=expts, step=0.0, auto_norm=False)
        )

    @staticmethod
    def update_step(value: SweepValue, step: float | DirectValue) -> SweepValue:
        candidate = replace(value, step=step, auto_norm=False)
        requested = resolved_direct_number(candidate.step)
        if requested is None:
            return candidate
        if not math.isfinite(requested):
            raise ValueError("Sweep step must be finite")
        bounds = SweepEditor._numeric_bounds(candidate)
        if bounds is None:
            # A typed edit retains the existing unresolved-axis contract. Text
            # input still needs a snapshot of the new raw, even before resolution.
            return candidate if isinstance(step, DirectValue) else value
        start, stop = bounds
        expts = 1 if requested == 0.0 else max(1, round((stop - start) / requested + 1))
        resolved = SweepEditor._step_from_expts(start, stop, expts)
        return replace(
            candidate,
            expts=expts,
            step=_resolved_step_input(candidate.step, resolved),
            auto_norm=False,
        )

    @staticmethod
    def _numeric_bounds(value: SweepValue) -> tuple[float, float] | None:
        start = SweepEditor._resolved_edge(value.start)
        stop = SweepEditor._resolved_edge(value.stop)
        if start is None or stop is None:
            return None
        return start, stop

    @staticmethod
    def _resolved_edge(value: float | ScalarValue) -> float | None:
        resolved = (
            value.resolved
            if isinstance(value, EvalValue)
            else resolved_direct_number(value)
        )
        if resolved is None:
            return None
        numeric = float(resolved)
        if not math.isfinite(numeric):
            raise ValueError("Sweep bounds must be finite")
        return numeric

    @staticmethod
    def _step_from_expts(start: float, stop: float, expts: int) -> float:
        return 0.0 if expts == 1 else (stop - start) / (expts - 1)


class CenteredSweepEditor:
    """Pure canonical transformation rules for a center/span sweep axis."""

    @staticmethod
    def canonicalize(value: CenteredSweepValue) -> CenteredSweepValue:
        span = resolved_direct_number(value.span)
        points = resolved_direct_number(value.expts)
        derived = None
        if span is not None and points is not None:
            derived = CenteredSweepEditor._step_from_expts(float(span), int(points))
        return replace(
            value, step=_canonical_step_input(value.step, derived), auto_norm=False
        )

    @staticmethod
    def update_center(
        value: CenteredSweepValue, center: float | ScalarValue
    ) -> CenteredSweepValue:
        return CenteredSweepEditor.canonicalize(
            replace(value, center=center, auto_norm=False)
        )

    @staticmethod
    def update_span(
        value: CenteredSweepValue, span: float | DirectValue
    ) -> CenteredSweepValue:
        return CenteredSweepEditor.canonicalize(
            replace(value, span=span, step=0.0, auto_norm=False)
        )

    @staticmethod
    def update_expts(
        value: CenteredSweepValue, expts: int | DirectValue
    ) -> CenteredSweepValue:
        return CenteredSweepEditor.canonicalize(
            replace(value, expts=expts, step=0.0, auto_norm=False)
        )

    @staticmethod
    def update_step(
        value: CenteredSweepValue, step: float | DirectValue
    ) -> CenteredSweepValue:
        candidate = replace(value, step=step, auto_norm=False)
        requested = resolved_direct_number(candidate.step)
        if requested is None:
            return candidate
        if not math.isfinite(requested) or requested < 0.0:
            raise ValueError("Centered sweep step must be finite and >= 0")
        span = resolved_direct_number(candidate.span)
        if span is None:
            return candidate
        expts = 1 if requested == 0.0 else max(1, round(span / requested + 1))
        resolved = CenteredSweepEditor._step_from_expts(float(span), expts)
        return replace(
            candidate,
            expts=expts,
            step=_resolved_step_input(candidate.step, resolved),
            auto_norm=False,
        )

    @staticmethod
    def _step_from_expts(span: float, expts: int) -> float:
        return 0.0 if expts == 1 else span / (expts - 1)

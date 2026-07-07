from __future__ import annotations

import math

from .adapter.types import CenteredSweepValue, EvalValue, SweepValue


class SweepEditor:
    """Pure canonical transformation rules for a single sweep axis."""

    @staticmethod
    def canonicalize(value: SweepValue) -> SweepValue:
        bounds = SweepEditor._numeric_bounds(value)
        if bounds is None:
            return value
        start, stop = bounds
        # auto_norm=False: this IS the canonicalisation authority — step is
        # already derived here, don't let SweepValue re-derive it.
        return SweepValue(
            start=value.start,
            stop=value.stop,
            expts=value.expts,
            step=SweepEditor._step_from_expts(start, stop, value.expts),
            auto_norm=False,
        )

    @staticmethod
    def update_start(value: SweepValue, start: float | EvalValue) -> SweepValue:
        return SweepEditor.canonicalize(
            SweepValue(
                start=start,
                stop=value.stop,
                expts=value.expts,
                step=value.step,
                auto_norm=False,
            )
        )

    @staticmethod
    def update_stop(value: SweepValue, stop: float | EvalValue) -> SweepValue:
        return SweepEditor.canonicalize(
            SweepValue(
                start=value.start,
                stop=stop,
                expts=value.expts,
                step=value.step,
                auto_norm=False,
            )
        )

    @staticmethod
    def update_expts(value: SweepValue, expts: int) -> SweepValue:
        return SweepEditor.canonicalize(
            SweepValue(
                start=value.start,
                stop=value.stop,
                expts=expts,
                step=value.step,
                auto_norm=False,
            )
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
        # step is the user's input → expts is reverse-derived; auto_norm=False so
        # the supplied step is preserved (not overwritten by the forward rule).
        return SweepEditor.canonicalize(
            SweepValue(
                start=value.start,
                stop=value.stop,
                expts=expts,
                step=step,
                auto_norm=False,
            )
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


class CenteredSweepEditor:
    """Pure canonical transformation rules for a center/span sweep axis."""

    @staticmethod
    def canonicalize(value: CenteredSweepValue) -> CenteredSweepValue:
        return CenteredSweepValue(
            center=value.center,
            span=value.span,
            expts=value.expts,
            step=CenteredSweepEditor._step_from_expts(value.span, value.expts),
            auto_norm=False,
        )

    @staticmethod
    def update_center(
        value: CenteredSweepValue, center: float | EvalValue
    ) -> CenteredSweepValue:
        return CenteredSweepEditor.canonicalize(
            CenteredSweepValue(
                center=center,
                span=value.span,
                expts=value.expts,
                step=value.step,
                auto_norm=False,
            )
        )

    @staticmethod
    def update_span(value: CenteredSweepValue, span: float) -> CenteredSweepValue:
        return CenteredSweepEditor.canonicalize(
            CenteredSweepValue(
                center=value.center,
                span=span,
                expts=value.expts,
                step=value.step,
                auto_norm=False,
            )
        )

    @staticmethod
    def update_expts(value: CenteredSweepValue, expts: int) -> CenteredSweepValue:
        return CenteredSweepEditor.canonicalize(
            CenteredSweepValue(
                center=value.center,
                span=value.span,
                expts=expts,
                step=value.step,
                auto_norm=False,
            )
        )

    @staticmethod
    def update_step(value: CenteredSweepValue, step: float) -> CenteredSweepValue:
        if not math.isfinite(step) or step < 0.0:
            raise ValueError("Centered sweep step must be finite and >= 0")
        expts = 1 if step == 0.0 else max(1, round(value.span / step + 1))
        return CenteredSweepEditor.canonicalize(
            CenteredSweepValue(
                center=value.center,
                span=value.span,
                expts=expts,
                step=step,
                auto_norm=False,
            )
        )

    @staticmethod
    def _step_from_expts(span: float, expts: int) -> float:
        return 0.0 if expts == 1 else span / (expts - 1)

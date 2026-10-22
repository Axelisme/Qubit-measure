"""Signed sweep coordinates, sampling and atomic representation changes."""

import math
from functools import cached_property
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, JsonValue, model_validator

from zcu_tools.gui.form import (
    Display,
    FormCommandContext,
    FormCommandError,
    FormCommandResult,
    FormCommandSnapshot,
    FormModel,
    InputCommand,
    InputDefault,
    field_error,
)


def _data(snapshot: FormCommandSnapshot) -> dict[str, JsonValue]:
    data = snapshot.copy_input()
    if not isinstance(data, dict):
        raise FormCommandError("Cannot convert coordinates: expected a sweep object")
    return data


def _number(context: FormCommandContext, name: str) -> float:
    try:
        value = context.evaluate((name,))
    except FormCommandError as exc:
        raise FormCommandError(f"Cannot convert coordinates: {exc}") from exc
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FormCommandError(f"Cannot convert coordinates: {name} is not numeric")
    return float(value)


def _converted(data: dict[str, JsonValue]) -> FormCommandResult:
    kinds = {
        model.model_fields["kind"].default: model
        for model in (LinRange, CenterSpan, StepRange, StepCenter)
    }
    kind = data["kind"]
    if not isinstance(kind, str):
        raise FormCommandError("Invalid sweep kind")
    return FormCommandResult({"kind": kind}, kinds[kind], data)


def _to_center(
    snapshot: FormCommandSnapshot, context: FormCommandContext
) -> FormCommandResult:
    data = _data(snapshot)
    start, stop = _number(context, "start"), _number(context, "stop")
    sampling = "step" if data.get("kind") == "linrange_step" else "n"
    result: dict[str, JsonValue] = {
        "kind": "center_step" if sampling == "step" else "center",
        "center": start / 2 + stop / 2,
        "span": stop - start,
    }
    if sampling in data:
        result[sampling] = data[sampling]
    return _converted(result)


def _to_range(
    snapshot: FormCommandSnapshot, context: FormCommandContext
) -> FormCommandResult:
    data = _data(snapshot)
    center, span = _number(context, "center"), _number(context, "span")
    sampling = "step" if data.get("kind") == "center_step" else "n"
    result: dict[str, JsonValue] = {
        "kind": "linrange_step" if sampling == "step" else "linrange",
        "start": center - span / 2,
        "stop": center + span / 2,
    }
    if sampling in data:
        result[sampling] = data[sampling]
    return _converted(result)


def _step_count(span: float, step: float) -> int:
    if span == 0:
        if step != 0:
            raise ValueError("A zero span requires step=0 for a single point")
        return 1
    if step == 0 or (step > 0) != (span > 0):
        raise ValueError("Step must be nonzero and have the same direction as span")
    quotient = span / step
    if not math.isfinite(quotient):
        raise ValueError("Span / step must be finite")
    nearest = round(quotient)
    # Relative tolerance only repairs almost-integral quotients, not ordinary
    # nondivisible spans. This matches the native sweep consistency tolerance.
    if math.isclose(quotient, nearest, rel_tol=1e-9, abs_tol=0):
        quotient = nearest
    if quotient < 1:
        raise ValueError("Step must not exceed a nonzero span")
    return math.floor(quotient) + 1


def _expts_step(span: float, n: int) -> float:
    if n == 1:
        if span != 0:
            raise ValueError("A single point requires zero span")
        return 0
    if span == 0:
        raise ValueError("Zero span requires a single point")
    return span / (n - 1)


def _sampling_change(
    snapshot: FormCommandSnapshot, context: FormCommandContext, *, to_step: bool
) -> dict[str, JsonValue]:
    data = _data(snapshot)
    center_mode = data.get("kind") in ("center", "center_step")
    if center_mode:
        center, span = _number(context, "center"), _number(context, "span")
        start, stop = center - span / 2, center + span / 2
    else:
        start, stop = _number(context, "start"), _number(context, "stop")
    try:
        if to_step:
            count = context.evaluate(("n",))
            if not isinstance(count, int) or isinstance(count, bool):
                raise FormCommandError("Cannot convert sampling: n is not an integer")
            step = _expts_step(stop - start, count)
            sampling: dict[str, JsonValue] = {"step": step}
        else:
            step = _number(context, "step")
            count = _step_count(stop - start, step)
            stop = start + step * (count - 1)
            sampling = {"n": count}
    except ValueError as exc:
        raise FormCommandError(f"Cannot convert sampling: {exc}") from exc
    if center_mode:
        return {
            "kind": "center_step" if to_step else "center",
            "center": start / 2 + stop / 2,
            "span": stop - start,
            **sampling,
        }
    return {
        "kind": "linrange_step" if to_step else "linrange",
        "start": start,
        "stop": stop,
        **sampling,
    }


def _to_step(
    snapshot: FormCommandSnapshot, context: FormCommandContext
) -> FormCommandResult:
    return _converted(_sampling_change(snapshot, context, to_step=True))


def _to_expts(
    snapshot: FormCommandSnapshot, context: FormCommandContext
) -> FormCommandResult:
    return _converted(_sampling_change(snapshot, context, to_step=False))


_CENTER = InputCommand(
    "to_center",
    "Switch to Center / Span",
    (("start",), ("stop",)),
    _to_center,
    result_tag=("kind", "center"),
)
_STEP_CENTER = InputCommand(
    "to_center",
    "Switch to Center / Span",
    (("start",), ("stop",)),
    _to_center,
    result_tag=("kind", "center_step"),
)
_RANGE = InputCommand(
    "to_range",
    "Switch to Range",
    (("center",), ("span",)),
    _to_range,
    result_tag=("kind", "linrange"),
)
_STEP_RANGE = InputCommand(
    "to_range",
    "Switch to Range",
    (("center",), ("span",)),
    _to_range,
    result_tag=("kind", "linrange_step"),
)
_RANGE_STEP = InputCommand(
    "to_step",
    "Switch to Step",
    (("start",), ("stop",), ("n",)),
    _to_step,
    result_tag=("kind", "linrange_step"),
)
_CENTER_STEP = InputCommand(
    "to_step",
    "Switch to Step",
    (("center",), ("span",), ("n",)),
    _to_step,
    result_tag=("kind", "center_step"),
)
_RANGE_EXPTS = InputCommand(
    "to_expts",
    "Switch to Expts",
    (("start",), ("stop",), ("step",)),
    _to_expts,
    result_tag=("kind", "linrange"),
)
_CENTER_EXPTS = InputCommand(
    "to_expts",
    "Switch to Expts",
    (("center",), ("span",), ("step",)),
    _to_expts,
    result_tag=("kind", "center"),
)


class LinRange(FormModel):
    """Range + Expts in caller units, including descending sweeps.

    kind selects the branch. start/stop are finite endpoints; n is a positive
    count. Native consistency (including single points) belongs to cfg lowering.
    Commands require authored participants and replace them with direct values.
    """

    input_commands: ClassVar[tuple[InputCommand, ...]] = (_CENTER, _RANGE_STEP)
    kind: Annotated[Literal["linrange"], Display(hidden=True)] = "linrange"
    start: float
    stop: float
    n: Annotated[int, Display(label="Expts")] = Field(ge=1)


class CenterSpan(FormModel):
    """Center + Expts; span is signed full width, never halfwidth.

    kind selects the branch; center is the finite midpoint in caller units,
    span is stop minus start; n is a positive count. No old-mode cache exists.
    """

    input_commands: ClassVar[tuple[InputCommand, ...]] = (_RANGE, _CENTER_STEP)
    kind: Annotated[Literal["center"], Display(hidden=True)] = "center"
    center: float
    span: float
    n: Annotated[int, Display(label="Expts")] = Field(ge=1)

    @property
    def start(self) -> float:
        """Return the effective first endpoint in caller units."""
        return self.center - self.span / 2

    @property
    def stop(self) -> float:
        """Return the effective final endpoint in caller units."""
        return self.center + self.span / 2


class StepRange(FormModel):
    """Range + Step; validation projects stop, without changing authored input.

    kind selects the branch. start/stop are finite caller-unit endpoints; step is
    signed. A nonzero interval needs a same-direction step no larger than itself.
    Zero span requires step=0. Effective stop preserves start and the given step;
    n uses floor(span/step)+1, repairing near integers with relative 1e-9 tolerance.
    """

    input_commands: ClassVar[tuple[InputCommand, ...]] = (_STEP_CENTER, _RANGE_EXPTS)
    kind: Annotated[Literal["linrange_step"], Display(hidden=True)] = "linrange_step"
    start: float
    stop: float
    step: float

    @model_validator(mode="after")
    def _effective_stop(self) -> Self:
        try:
            count = self.n
        except ValueError as exc:
            raise field_error(("step",), str(exc)) from exc
        self.stop = self.start + self.step * (count - 1)
        return self

    @cached_property
    def n(self) -> int:
        """Return the validated count, retained before endpoint projection."""
        # Subtracting projected endpoints at a large offset can lose a step.
        return _step_count(self.stop - self.start, self.step)


class StepCenter(FormModel):
    """Center + Step; validation projects the effective midpoint and signed span.

    kind selects the branch. center/span are finite caller-unit coordinates;
    span is full width; step is signed. StepRange's sampling rules apply.
    Projection preserves the authored first endpoint, never centers a shortened
    interval on the authored midpoint. FormInput still holds the authored fields.
    """

    input_commands: ClassVar[tuple[InputCommand, ...]] = (_STEP_RANGE, _CENTER_EXPTS)
    kind: Annotated[Literal["center_step"], Display(hidden=True)] = "center_step"
    center: float
    span: float
    step: float

    @model_validator(mode="after")
    def _effective_coordinates(self) -> Self:
        try:
            count = self.n
        except ValueError as exc:
            raise field_error(("step",), str(exc)) from exc
        start = self.start
        self.span = self.step * (count - 1)
        self.center = start + self.span / 2
        return self

    @property
    def start(self) -> float:
        """Return the effective first endpoint in caller units."""
        return self.center - self.span / 2

    @property
    def stop(self) -> float:
        """Return the effective final endpoint in caller units."""
        return self.center + self.span / 2

    @cached_property
    def n(self) -> int:
        """Return the validated count, retained before coordinate projection."""
        return _step_count(self.span, self.step)


def _fresh_sweep() -> JsonValue:
    return {"kind": "linrange"}


type SweepInput = Annotated[
    LinRange | CenterSpan | StepRange | StepCenter,
    Field(discriminator="kind"),
    Display(selector=False),
    InputDefault(_fresh_sweep),
]
"""Four sweep branches; fresh handwritten fields start with Range + Expts defaults."""

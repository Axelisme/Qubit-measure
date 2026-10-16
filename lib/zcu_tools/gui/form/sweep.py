"""Signed sweep coordinates, sampling and atomic representation changes."""

import math
from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from .model import (
    Display,
    FormCommandError,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    InputCommand,
    field_error,
)


class _RangeCoordinates(FormModel):
    start: float
    stop: float


class _CenterCoordinates(FormModel):
    center: float
    span: float


class _Count(FormModel):
    n: int = Field(ge=1)


class _Step(FormModel):
    step: float


def _evaluated[F: FormModel](
    model: type[F], data: Mapping[str, JsonValue], env: FormEnv, scope: str
) -> F:
    # Command participants must be authored, never supplied by branch defaults.
    result = FormState(
        FormInput(model, {key: data[key] for key in model.model_fields if key in data})
    ).result(env)
    if isinstance(result, FormErrors):
        raise FormCommandError(f"Cannot convert {scope}: {result.messages}")
    return result


def _to_center(data: Mapping[str, JsonValue], env: FormEnv) -> dict[str, JsonValue]:
    coords = _evaluated(_RangeCoordinates, data, env, "coordinates")
    sampling = "step" if data.get("kind") == "linrange_step" else "n"
    result: dict[str, JsonValue] = {
        "kind": "center_step" if sampling == "step" else "center",
        "center": coords.start / 2 + coords.stop / 2,
        "span": coords.stop - coords.start,
    }
    if sampling in data:
        result[sampling] = data[sampling]
    return result


def _to_range(data: Mapping[str, JsonValue], env: FormEnv) -> dict[str, JsonValue]:
    coords = _evaluated(_CenterCoordinates, data, env, "coordinates")
    sampling = "step" if data.get("kind") == "center_step" else "n"
    result: dict[str, JsonValue] = {
        "kind": "linrange_step" if sampling == "step" else "linrange",
        "start": coords.center - coords.span / 2,
        "stop": coords.center + coords.span / 2,
    }
    if sampling in data:
        result[sampling] = data[sampling]
    return result


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
    data: Mapping[str, JsonValue], env: FormEnv, *, to_step: bool
) -> dict[str, JsonValue]:
    center_mode = data.get("kind") in ("center", "center_step")
    if center_mode:
        coords = _evaluated(_CenterCoordinates, data, env, "sampling")
        start, stop = coords.center - coords.span / 2, coords.center + coords.span / 2
    else:
        coords = _evaluated(_RangeCoordinates, data, env, "sampling")
        start, stop = coords.start, coords.stop
    try:
        if to_step:
            count = _evaluated(_Count, data, env, "sampling").n
            step = _expts_step(stop - start, count)
            sampling: dict[str, JsonValue] = {"step": step}
        else:
            step = _evaluated(_Step, data, env, "sampling").step
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


def _to_step(data: Mapping[str, JsonValue], env: FormEnv) -> dict[str, JsonValue]:
    return _sampling_change(data, env, to_step=True)


def _to_expts(data: Mapping[str, JsonValue], env: FormEnv) -> dict[str, JsonValue]:
    return _sampling_change(data, env, to_step=False)


_CENTER = InputCommand(
    "to_center", "Switch to Center / Span", ("start", "stop"), _to_center
)
_RANGE = InputCommand("to_range", "Switch to Range", ("center", "span"), _to_range)
_RANGE_STEP = InputCommand(
    "to_step", "Switch to Step", ("start", "stop", "n"), _to_step
)
_CENTER_STEP = InputCommand(
    "to_step", "Switch to Step", ("center", "span", "n"), _to_step
)
_RANGE_EXPTS = InputCommand(
    "to_expts", "Switch to Expts", ("start", "stop", "step"), _to_expts
)
_CENTER_EXPTS = InputCommand(
    "to_expts", "Switch to Expts", ("center", "span", "step"), _to_expts
)


class LinRange(FormModel):
    """Range + Expts in caller units, including descending sweeps.

    kind selects the branch. start/stop are finite endpoints; n is a positive
    count. Native consistency (including single points) belongs to cfg lowering.
    Commands require authored participants and replace them with direct values.
    """

    kind: Literal["linrange"] = "linrange"
    start: Annotated[float, _CENTER] = 0.0
    stop: Annotated[float, _CENTER] = 1.0
    n: Annotated[int, Display(label="Expts"), _RANGE_STEP] = Field(default=3, ge=1)


class CenterSpan(FormModel):
    """Center + Expts; span is signed full width, never halfwidth.

    kind selects the branch; center is the finite midpoint in caller units,
    span is stop minus start; n is a positive count. No old-mode cache exists.
    """

    kind: Literal["center"] = "center"
    center: Annotated[float, _RANGE]
    span: Annotated[float, _RANGE]
    n: Annotated[int, Display(label="Expts"), _CENTER_STEP] = Field(default=3, ge=1)

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

    kind: Literal["linrange_step"] = "linrange_step"
    start: Annotated[float, _CENTER]
    stop: Annotated[float, _CENTER]
    step: Annotated[float, _RANGE_EXPTS]

    @model_validator(mode="after")
    def _effective_stop(self) -> Self:
        try:
            count = _step_count(self.stop - self.start, self.step)
        except ValueError as exc:
            raise field_error(("step",), str(exc)) from exc
        self.stop = self.start + self.step * (count - 1)
        return self

    @property
    def n(self) -> int:
        """Return the effective count, including the first point."""
        return _step_count(self.stop - self.start, self.step)


class StepCenter(FormModel):
    """Center + Step; validation projects the effective midpoint and signed span.

    kind selects the branch. center/span are finite caller-unit coordinates;
    span is full width; step is signed. StepRange's sampling rules apply.
    Projection preserves the authored first endpoint, never centers a shortened
    interval on the authored midpoint. FormInput still holds the authored fields.
    """

    kind: Literal["center_step"] = "center_step"
    center: Annotated[float, _RANGE]
    span: Annotated[float, _RANGE]
    step: Annotated[float, _CENTER_EXPTS]

    @model_validator(mode="after")
    def _effective_coordinates(self) -> Self:
        try:
            count = _step_count(self.span, self.step)
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

    @property
    def n(self) -> int:
        """Return the effective count, including the first point."""
        return _step_count(self.span, self.step)


type SweepInput = Annotated[
    LinRange | CenterSpan | StepRange | StepCenter,
    Field(discriminator="kind"),
    Display(selector=False),
]
"""Four active sweep branches shared by generated and hand-authored forms."""

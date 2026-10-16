"""Signed sweep coordinates and explicit, field-local representation changes."""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue

from .model import (
    Display,
    FormCommandError,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    InputCommand,
)


class _RangeCoordinates(FormModel):
    start: float
    stop: float


class _CenterCoordinates(FormModel):
    center: float
    span: float


def _to_center(data: Mapping[str, JsonValue], env: FormEnv) -> dict[str, JsonValue]:
    coords = FormState(
        FormInput(
            _RangeCoordinates,
            {key: data[key] for key in ("start", "stop") if key in data},
        )
    ).result(env)
    if isinstance(coords, FormErrors):
        raise FormCommandError(f"Cannot convert coordinates: {coords.messages}")
    result: dict[str, JsonValue] = {
        "kind": "center",
        "center": coords.start / 2 + coords.stop / 2,
        "span": coords.stop - coords.start,
    }
    if "n" in data:
        result["n"] = data["n"]
    return result


def _to_range(data: Mapping[str, JsonValue], env: FormEnv) -> dict[str, JsonValue]:
    coords = FormState(
        FormInput(
            _CenterCoordinates,
            {key: data[key] for key in ("center", "span") if key in data},
        )
    ).result(env)
    if isinstance(coords, FormErrors):
        raise FormCommandError(f"Cannot convert coordinates: {coords.messages}")
    result: dict[str, JsonValue] = {
        "kind": "linrange",
        "start": coords.center - coords.span / 2,
        "stop": coords.center + coords.span / 2,
    }
    if "n" in data:
        result["n"] = data["n"]
    return result


_CENTER = InputCommand(
    "to_center", "Switch to Center / Span", ("start", "stop"), _to_center
)
_RANGE = InputCommand("to_range", "Switch to Range", ("center", "span"), _to_range)


class LinRange(FormModel):
    """Range + Expts input in caller units, including descending sweeps.

    kind is the branch tag. start/stop are finite endpoints, n is a positive
    point count. Native consistency (including single points) belongs to cfg
    lowering. Coordinate commands require both authored endpoints, not defaults.
    """

    kind: Literal["linrange"] = "linrange"
    start: Annotated[float, _CENTER] = 0.0
    stop: Annotated[float, _CENTER] = 1.0
    n: int = Field(default=3, ge=1)


class CenterSpan(FormModel):
    """Center + Expts input; span is signed full width, never halfwidth.

    kind is the branch tag, center is the finite midpoint in caller units,
    span is stop minus start, n is a positive point count. No old-mode cache.
    """

    kind: Literal["center"] = "center"
    center: Annotated[float, _RANGE]
    span: Annotated[float, _RANGE]
    n: int = Field(default=3, ge=1)

    @property
    def start(self) -> float:
        """Return the effective first endpoint in caller units."""
        return self.center - self.span / 2

    @property
    def stop(self) -> float:
        """Return the effective final endpoint in caller units."""
        return self.center + self.span / 2


type SweepInput = Annotated[
    LinRange | CenterSpan, Field(discriminator="kind"), Display(selector=False)
]
"""Shared tagged coordinates for hand-authored forms, including workflows."""

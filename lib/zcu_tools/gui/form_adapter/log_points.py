"""Opt-in Log/List integer inputs; native cfgs receive only list[int]."""

from collections.abc import Mapping
from functools import cached_property
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, JsonValue, ValidationError, model_validator

from zcu_tools.gui.form import (
    Display,
    FormCommandContext,
    FormCommandError,
    FormCommandSnapshot,
    FormConversionError,
    FormConversionResult,
    FormModel,
    InputConverter,
)
from zcu_tools.gui.form.commands import ReadonlyFormInput


class LogIntegerPoints(FormModel):
    """Log input with a nonnegative max and requested total points including 0.

    kind selects Log. max is the inclusive endpoint. points must be at least
    two, including when max=0. Validation projects points
    to the actual unique count; raw input retains the request. values caches
    the original sampling before this projection, so lowering never resamples.
    """

    kind: Annotated[Literal["log"], Display(hidden=True)] = "log"
    max: Annotated[int, Display(label="Max")] = Field(ge=0, strict=True)
    points: Annotated[int, Display(label="Points")] = Field(ge=2, strict=True)

    @cached_property
    def values(self) -> list[int]:
        """Return rounded unique logarithmic points, always including 0 and max.

        Sample geomspace(1, max, points-1), fixing its last value to max before
        rounding and deduplication. Two requested points yield [0, max]. max=0
        yields [0]. Rounding uses NumPy nearest-even for half-integers.
        """
        if self.max == 0:
            return [0]
        samples = np.geomspace(1, self.max, self.points - 1)
        samples[-1] = self.max
        return [0, *np.unique(np.rint(samples)).astype(int).tolist()]

    @model_validator(mode="after")
    def project_count(self) -> Self:
        """Retain original samples and expose their actual count to preview."""
        self.points = len(self.values)
        return self


class PositiveLogIntegerPoints(LogIntegerPoints):
    """Positive Log input with both 1 and max retained after rounding.

    max is at least 1. points counts only positive values; max>1 requires at
    least two requested points. max=1 accepts one and produces [1]. Preview
    projects points to the unique count without changing the authored request.
    """

    max: Annotated[int, Display(label="Max")] = Field(ge=1, strict=True)
    points: Annotated[int, Display(label="Points")] = Field(ge=1, strict=True)

    @cached_property
    def values(self) -> list[int]:
        """Return positive rounded unique geomspace values, retaining 1 and max.

        Use the shared zero-inclusive engine with one extra requested point,
        then remove zero. This does not alter the zero-inclusive policy.
        """
        return [
            value
            for value in LogIntegerPoints(max=self.max, points=self.points + 1).values
            if value > 0
        ]

    @model_validator(mode="after")
    def project_count(self) -> Self:
        """Reject impossible endpoints, then expose the original unique count."""
        if self.max > 1 and self.points < 2:
            raise ValueError("At least two points are required to include 1 and max")
        self.points = len(self.values)
        return self


class ExplicitIntegerPoints(FormModel):
    """List input; values is a nonempty ordered list of nonnegative integers.

    kind selects List. No sorting, deduplication or resampling occurs. Native
    cfg validation may impose further experiment-specific restrictions.
    """

    kind: Annotated[Literal["list"], Display(hidden=True)] = "list"
    values: list[Annotated[int, Field(ge=0, strict=True)]] = Field(min_length=1)


class PositiveExplicitIntegerPoints(ExplicitIntegerPoints):
    """List input with nonempty positive integer values, preserving order."""

    values: list[Annotated[int, Field(ge=1, strict=True)]] = Field(min_length=1)


def _convert(
    old: FormCommandSnapshot,
    selection: ReadonlyFormInput,
    context: FormCommandContext,
    *,
    include_zero: bool = True,
) -> FormConversionResult:
    if not isinstance(selection, Mapping):
        raise FormConversionError("Integer points selection must be Log or List")
    raw = old.copy_input()
    if not isinstance(raw, dict):
        raise FormConversionError("Integer points input must be a tagged object")
    log_model = LogIntegerPoints if include_zero else PositiveLogIntegerPoints
    list_model = (
        ExplicitIntegerPoints if include_zero else PositiveExplicitIntegerPoints
    )
    try:
        if raw.get("kind") == "log":
            source = log_model.model_validate(
                {
                    "max": context.evaluate(("max",)),
                    "points": context.evaluate(("points",)),
                }
            )
        else:
            source = list_model.model_validate(
                {
                    "values": context.evaluate(("values",)),
                }
            )
    except (ValidationError, FormCommandError) as exc:
        raise FormConversionError(str(exc)) from exc
    if selection.get("kind") == "list":
        data: dict[str, JsonValue] = {"kind": "list", "values": [*source.values]}
        return FormConversionResult(list_model, data)
    if selection.get("kind") == "log":
        return FormConversionResult(
            log_model,
            {
                "kind": "log",
                "max": max(source.values),
                "points": len(source.values),
            },
        )
    raise FormConversionError("Integer points selection must be Log or List")


def _convert_positive(
    old: FormCommandSnapshot,
    selection: ReadonlyFormInput,
    context: FormCommandContext,
) -> FormConversionResult:
    return _convert(old, selection, context, include_zero=False)


PositiveIntegerPointsInput = Annotated[
    PositiveLogIntegerPoints | PositiveExplicitIntegerPoints,
    Field(discriminator="kind"),
    InputConverter(_convert_positive),
]
"""Positive Log/List annotation for LogPoints(include_zero=False)."""


IntegerPointsInput = Annotated[
    LogIntegerPoints | ExplicitIntegerPoints,
    Field(discriminator="kind"),
    InputConverter(_convert),
]
"""Log/List branch annotation used only by LogPoints-authored integer lists."""

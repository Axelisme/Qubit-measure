"""Front-end-neutral scalar nodes for core-derived form declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .template_nodes import Expr, Hidden, Template, TemplateSeed


@dataclass(frozen=True, slots=True)
class Range:
    """A linear sweep declaration in the core field's units.

    start/stop are literal endpoints, Expr nodes, or None; expts is the point
    count. None leaves that endpoint visibly blank and required in the Form.
    It is not 0/1 or an executable default: the operator must enter a number
    before submission, even when the endpoint's input key is absent.
    Conversion to a core sweep validates the endpoints and count. The helpers
    treat string arguments as expression bodies, unlike ordinary tree strings.
    """

    start: float | Expr | None
    stop: float | Expr | None
    expts: int

    @staticmethod
    def around(center: str | float, halfwidth: str | float, expts: int) -> Range:
        """Create endpoints center ± halfwidth; strings denote expressions.

        No evaluation occurs. expts is passed unchanged to core sweep validation.
        """
        return Range(
            Expr(f"({center}) - ({halfwidth})"),
            Expr(f"({center}) + ({halfwidth})"),
            expts,
        )

    @staticmethod
    def between(
        start: str | float | None, stop: str | float | None, expts: int
    ) -> Range:
        """Create endpoints; strings denote expressions, None means blank required.

        expts is passed unchanged to core sweep validation. No evaluation occurs.
        """
        return Range(
            Expr(start) if isinstance(start, str) else start,
            Expr(stop) if isinstance(stop, str) else stop,
            expts,
        )


@dataclass(frozen=True, slots=True)
class LogPoints:
    """Opt a list[int] field into Log/List input with these fresh Log values.

    max is the inclusive integer endpoint. include_zero=True keeps a zero
    point and counts it in points. False requests positive points starting at
    1; max>1 needs at least two points, while max=1 accepts one. The frontend
    owns generation, rounding, deduplication and validation; this declaration
    performs no computation.
    """

    max: int = 128
    points: int = 9
    include_zero: bool = True


@dataclass(frozen=True, slots=True)
class AliasName:
    """Resolve an alias once at Reset to a direct name, not an expression.

    alias is a bare alias name or a $-prefixed path. choices_source='devices'
    limits acceptable names to captured connected devices; None disables that
    restriction. Missing/unbound aliases leave the input absent and Invalid
    when the core field is required. Optional core defaults remain absent.
    """

    alias: str
    choices_source: Literal["devices"] | None = "devices"


type FormNode = (
    TemplateSeed
    | Template
    | Range
    | LogPoints
    | AliasName
    | Hidden
    | Mapping[str, FormNode]
    | list[FormNode]
)

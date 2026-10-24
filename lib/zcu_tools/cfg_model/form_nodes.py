"""Front-end-neutral scalar nodes for core-derived form declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .template_nodes import Expr, Hidden, Template, TemplateSeed


@dataclass(frozen=True, slots=True, init=False)
class Range:
    """A linear sweep declaration in the core field's units.

    start/stop are literal endpoints, expression strings, Expr nodes, or None.
    expts is the point count. None leaves that endpoint visibly blank and required in the Form.
    It is not 0/1 or an executable default: the operator must enter a number
    before submission, even when the endpoint's input key is absent.
    Conversion to a core sweep validates the endpoints and count. String
    arguments denote expression bodies, unlike ordinary tree strings.
    """

    start: float | Expr | None
    stop: float | Expr | None
    expts: int

    def __init__(
        self,
        start: str | float | Expr | None,
        stop: str | float | Expr | None,
        expts: int,
    ) -> None:
        """Declare endpoints without evaluation; strings become Expr nodes."""
        object.__setattr__(
            self, "start", Expr(start) if isinstance(start, str) else start
        )
        object.__setattr__(self, "stop", Expr(stop) if isinstance(stop, str) else stop)
        object.__setattr__(self, "expts", expts)

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


type FormNode = (
    TemplateSeed
    | Template
    | Range
    | LogPoints
    | Hidden
    | Mapping[str, FormNode]
    | list[FormNode]
)

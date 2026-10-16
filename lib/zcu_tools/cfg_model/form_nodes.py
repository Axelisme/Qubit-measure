"""Front-end-neutral scalar nodes for core-derived form declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .template_nodes import Expr, Template, TemplateSeed


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
class AliasName:
    """Resolve an alias once at Reset to a direct name, not an expression.

    alias is a bare alias name or a $-prefixed path. choices_source='devices'
    limits acceptable names to captured connected devices; None disables that
    restriction. Missing/unbound aliases leave the input absent and Invalid
    when the core field is required. Optional core defaults remain absent.
    """

    alias: str
    choices_source: Literal["devices"] | None = "devices"


@dataclass(frozen=True, slots=True)
class Locked:
    """A scalar value validated as a single-value Literal in the generated model.

    value is an ordinary scalar, never an expression. It remains visible in
    input/schema; changing it causes validation failure, not silent overriding.
    """

    value: str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class Hidden:
    """Omit a core field from input; cfg validation restores its declared default.

    The field must have a core default. A required hidden field fails during
    form construction because no value could be restored during conversion.
    """


type FormNode = (
    TemplateSeed
    | Template
    | Range
    | AliasName
    | Locked
    | Hidden
    | Mapping[str, FormNode]
)

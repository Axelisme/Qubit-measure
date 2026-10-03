"""MCP-only projections of captured execution facts; never read the live GUI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from recipes import RecipeDefinition

ReplyDetail = Literal["summary", "full"]


@dataclass(frozen=True)
class SummaryParameter:
    name: str
    field: str
    unit: str | None = None


@dataclass(frozen=True)
class SummaryEstimate:
    name: str
    value_key: str
    error_key: str | None = None
    unit: str | None = None


def project_execution(
    snapshot: dict[str, Any],
    *,
    definition: RecipeDefinition | None = None,
    detail: ReplyDetail = "summary",
) -> dict[str, Any]:
    """Project one captured snapshot without refreshing observations or guards."""
    del definition, detail
    return deepcopy(snapshot)

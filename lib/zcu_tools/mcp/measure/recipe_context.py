"""Recipe progress and fixed GUI binding, separate from GUI-owned operation state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.tool_context import MeasureToolContext

RecipeStatus = Literal[
    "running", "needs_parameters", "interactive", "finished", "failed", "cancelled"
]
RecipePhase = Literal[
    "preparing", "run", "raw_save", "analysis", "writeback_read", "terminal"
]


@dataclass(frozen=True)
class MissingParameter:
    parameter: str
    reason: str


@dataclass(frozen=True)
class RecipeError:
    phase: RecipePhase
    reason: str
    message: str
    code: str | None = None


@dataclass(frozen=True)
class RawSave:
    status: Literal["not_started", "saving", "saved", "failed", "unknown"] = "not_started"
    reserved_path: str | None = None
    path: str | None = None
    operation_outcome: dict[str, Any] | None = None


@dataclass
class RecipeSnapshot:
    execution: str
    recipe: str
    tab: str | None = None
    status: RecipeStatus = "running"
    phase: RecipePhase = "preparing"
    cancel_requested: bool = False
    finish_early_requested: bool = False
    run_op: int | None = None
    op: int | None = None
    actual: dict[str, Any] | None = None
    missing: list[MissingParameter] = field(default_factory=list)
    run_outcome: dict[str, Any] | None = None
    result_state: dict[str, Any] | None = None
    raw_save: RawSave = field(default_factory=RawSave)
    analysis: dict[str, Any] | None = None
    writeback: dict[str, Any] | None = None
    error: RecipeError | None = None


class RecipeContext:
    """One recipe's binding and progress; helpers never reconnect or retry."""

    def __init__(self, tools: MeasureToolContext, recipe: str) -> None:
        self.tools = tools.bound()
        self.progress = RecipeSnapshot(f"recipe-{uuid4().hex}", recipe)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self.progress)

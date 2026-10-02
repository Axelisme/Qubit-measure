"""Recipe progress and fixed GUI binding, separate from GUI-owned operation state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from zcu_tools.mcp.measure.session import GuiRpcError

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
    status: Literal["not_started", "saving", "saved", "failed", "unknown"] = (
        "not_started"
    )
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

    def rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        return self.tools.send_gui_rpc(method, params)

    def prepare_tab(self, experiment: str, reuse_tab_id: str | None) -> dict[str, Any]:
        if reuse_tab_id is None:
            created = self.rpc("tab.new", {"adapter_name": experiment})
            tab = created["tab_id"]
            if not isinstance(tab, str) or not tab:
                raise GuiRpcError(
                    "Invalid tab creation reply", reason="incompatible_wire"
                )
            self.progress.tab = tab
            return self.rpc("tab.get_cfg", {"tab_id": tab})
        self.progress.tab = reuse_tab_id
        observed = self.rpc("tab.snapshot", {"tab_id": reuse_tab_id})["tabs"]
        if len(observed) != 1 or observed[0]["tab_id"] != reuse_tab_id:
            raise GuiRpcError("Requested tab was not found", reason="unknown_tab")
        snapshot = observed[0]
        if snapshot["adapter_name"] != experiment:
            raise GuiRpcError(
                "Tab has a different experiment", reason="wrong_experiment"
            )
        interaction = snapshot["interaction"]
        if any(
            interaction[key] for key in ("is_running", "is_analyzing", "is_saving_data")
        ):
            raise GuiRpcError("Tab is busy", reason="tab_busy")
        cfg = self.rpc("tab.get_cfg", {"tab_id": reuse_tab_id})
        return self.rpc(
            "tab.reset_cfg", {"tab_id": reuse_tab_id, "expected": cfg["cfg_ref"]}
        )

    def needs_parameters(self, missing: list[MissingParameter]) -> None:
        self.progress.missing = missing
        self.progress.status = "needs_parameters"
        self.progress.phase = "terminal"

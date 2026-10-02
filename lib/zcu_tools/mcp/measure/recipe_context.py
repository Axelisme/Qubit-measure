"""Recipe progress and fixed GUI binding, separate from GUI-owned operation state."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from zcu_tools.mcp.core.reply import PngImage
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
        self.images: tuple[PngImage, ...] = ()

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

    def edit_cfg(
        self, publication: dict[str, Any], edits: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return self.rpc(
            "tab.edit_cfg",
            {
                "tab_id": self.progress.tab,
                "expected": publication["cfg_ref"],
                "edits": edits,
            },
        )

    def run_once(self, publication: dict[str, Any], fields: dict[str, Any]) -> None:
        """Run one observed publication; preserve the original receipts throughout."""
        tab = self.progress.tab
        if tab is None:
            raise RuntimeError("Prepare a tab before running")
        self.progress.actual = deepcopy(
            {
                "cfg_ref": publication["cfg_ref"],
                "fields": fields,
                "source_basis": publication["source_basis"],
            }
        )
        self.rpc("tab.snapshot", {"tab_id": tab})
        self.rpc("soc.info", {"include_cfg": True})
        devices = self.rpc("device.list", {})["devices"]
        for device in devices:
            self.rpc("device.snapshot", {"name": device["name"]})
        self.progress.phase = "run"
        started = self.rpc(
            "tab.run_start",
            {
                "tab_id": tab,
                "expected": publication["cfg_ref"],
            },
        )
        run_op = started["handle"]
        self.progress.run_op = self.progress.op = run_op
        outcome = self._await_operation(run_op)
        self.progress.run_outcome = outcome
        if outcome["status"] == "failed":
            raise GuiRpcError(
                str(outcome.get("error", "Run failed")), reason="run_failed"
            )
        if outcome["status"] == "cancelled":
            self.progress.status = "cancelled"
            self.progress.phase = "terminal"
            return
        observed = self.rpc("tab.snapshot", {"tab_id": tab})["tabs"][0]
        self.progress.result_state = deepcopy(observed["result_state"])
        self._save_raw(tab, run_op)
        self.progress.phase = "analysis"
        started = self.tools.gui.send_gui_rpc(
            "tab.analyze",
            {"tab_id": tab, "updates": {}},
            run_operation_handle=run_op,
        )
        self.progress.op = started["handle"]
        execution = self.tools.session.executions.start(
            self.tools.gui, tab, "primary", started
        )
        while True:
            reply = execution.wait(0.25)
            self.progress.analysis = reply.data
            self.images = reply.images
            if reply.data["status"] != "running":
                break
        if reply.data["status"] != "finished":
            self.progress.status = reply.data["status"]
            self.progress.phase = (
                "analysis" if self.progress.status == "interactive" else "terminal"
            )
            return
        self.progress.phase = "writeback_read"
        self.progress.writeback = self.tools.gui.send_gui_rpc(
            "tab.writeback_preview",
            {"tab_id": tab, "subtab_id": "analysis"},
            operation_handle=started["handle"],
        )
        self.progress.status = "finished"
        self.progress.phase = "terminal"

    def _await_operation(self, op: int) -> dict[str, Any]:
        while True:
            began = time.monotonic()
            outcome = self.tools.gui.send_gui_rpc(
                "operation.await",
                {"timeout": 0.25},
                2.25,
                operation_handle=op,
            )
            if outcome.get("reason") == "completed":
                if outcome.get("status") not in ("finished", "failed", "cancelled"):
                    raise GuiRpcError(
                        "Invalid operation outcome", reason="incompatible_wire"
                    )
                return outcome
            if outcome.get("reason") not in ("timeout", "user_feedback"):
                raise GuiRpcError(
                    "Invalid operation wait reply", reason="incompatible_wire"
                )
            time.sleep(max(0.0, 0.25 - (time.monotonic() - began)))

    def _save_raw(self, tab: str, run_op: int) -> None:
        self.progress.phase = "raw_save"
        try:
            started = self.tools.gui.send_gui_rpc(
                "tab.save_data",
                {"tab_id": tab},
                run_operation_handle=run_op,
            )
            path = started["data_path"]
            self.progress.op = started["handle"]
            self.progress.raw_save = RawSave("saving", reserved_path=path)
            outcome = self._await_operation(started["handle"])
            if outcome["status"] != "finished":
                self.progress.raw_save = replace(
                    self.progress.raw_save,
                    status="failed",
                    operation_outcome=outcome,
                )
                raise GuiRpcError(
                    str(outcome.get("error", "Raw save failed")),
                    reason="raw_save_failed",
                )
            self.progress.raw_save = RawSave("saved", path, path, outcome)
        except Exception as error:
            if self.progress.raw_save.status != "failed":
                reason = getattr(error, "reason", None)
                unknown = reason in (
                    "gui_transport_timeout",
                    "connection_lost",
                    "message_too_large",
                )
                self.progress.raw_save = replace(
                    self.progress.raw_save,
                    status="unknown" if unknown else "failed",
                )
            raise

    def needs_parameters(self, missing: list[MissingParameter]) -> None:
        self.progress.missing = missing
        self.progress.status = "needs_parameters"
        self.progress.phase = "terminal"

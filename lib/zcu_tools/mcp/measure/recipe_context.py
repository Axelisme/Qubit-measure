"""Recipe progress and fixed GUI binding, separate from GUI-owned operation state."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from threading import Condition, Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.session import GuiRpcError

if TYPE_CHECKING:
    from zcu_tools.mcp.measure.tool_context import MeasureToolContext

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
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

    def __init__(self, tools: MeasureToolContext, recipe: str, closed: Event) -> None:
        self.tools = tools.bound()
        self.progress = RecipeSnapshot(f"recipe-{uuid4().hex}", recipe)
        self.images: tuple[PngImage, ...] = ()
        self._closed = closed
        self._condition = Condition()
        self._thread: Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return asdict(self.progress)

    def wait(self, timeout: float) -> ToolReply:
        """Wait locally; a timeout ends only this wait, never the worker."""
        began = time.monotonic()
        with self._condition:
            self._condition.wait_for(
                lambda: self.progress.status != "running" or self._closed.is_set(),
                timeout,
            )
            return ToolReply(
                {**asdict(self.progress), "elapsed_s": time.monotonic() - began},
                self.images,
                is_error=self.progress.status == "failed",
            )

    def start(
        self,
        run: Callable[[RecipeContext, dict[str, Any]], None],
        arguments: dict[str, Any],
    ) -> None:
        """Start under the registry lock so close cannot miss this worker."""
        self._thread = Thread(
            target=self._run,
            args=(run, deepcopy(arguments)),
            name=self.progress.execution,
        )
        self._thread.start()

    def _run(
        self,
        run: Callable[[RecipeContext, dict[str, Any]], None],
        arguments: dict[str, Any],
    ) -> None:
        try:
            run(self, arguments)
        except Exception as error:  # Worker boundary retains partial progress.
            logger.exception("Recipe %s failed", self.progress.recipe)
            self._publish(
                error=RecipeError(
                    self.progress.phase,
                    str(getattr(error, "reason", None) or "recipe_failed"),
                    str(error),
                    getattr(error, "code", None),
                ),
                status="failed",
                phase="terminal",
            )

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _publish(self, **changes: Any) -> None:
        with self._condition:
            self.progress = replace(self.progress, **changes)
            self._condition.notify_all()

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
            self._publish(tab=tab)
            return self.rpc("tab.get_cfg", {"tab_id": tab})
        self._publish(tab=reuse_tab_id)
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
        self._publish(
            actual=deepcopy(
                {
                    "cfg_ref": publication["cfg_ref"],
                    "fields": fields,
                    "source_basis": publication["source_basis"],
                }
            )
        )
        self.rpc("tab.snapshot", {"tab_id": tab})
        self.rpc("soc.info", {"include_cfg": True})
        devices = self.rpc("device.list", {})["devices"]
        for device in devices:
            self.rpc("device.snapshot", {"name": device["name"]})
        self._publish(phase="run")
        started = self.rpc(
            "tab.run_start",
            {
                "tab_id": tab,
                "expected": publication["cfg_ref"],
            },
        )
        run_op = started["handle"]
        self._publish(run_op=run_op, op=run_op)
        outcome = self._await_operation(run_op)
        self._publish(run_outcome=outcome)
        if outcome["status"] == "failed":
            raise GuiRpcError(
                str(outcome.get("error", "Run failed")), reason="run_failed"
            )
        if outcome["status"] == "cancelled":
            self._publish(status="cancelled", phase="terminal")
            return
        observed = self.rpc("tab.snapshot", {"tab_id": tab})["tabs"][0]
        self._publish(result_state=deepcopy(observed["result_state"]))
        self._save_raw(tab, run_op)
        self._publish(phase="analysis")
        started = self.tools.gui.send_gui_rpc(
            "tab.analyze",
            {"tab_id": tab, "updates": {}},
            run_operation_handle=run_op,
        )
        self._publish(op=started["handle"])
        execution = self.tools.session.executions.start(
            self.tools.gui, tab, "primary", started
        )
        while True:
            reply = execution.wait(0.25)
            with self._condition:
                self.images = reply.images
                self._publish(
                    analysis=reply.data,
                    status="interactive"
                    if reply.data["status"] == "interactive"
                    else "running",
                )
            if reply.data["status"] == "interactive":
                # Interactive waits return immediately. Pace local observation while
                # the analysis owner continues to track the original operation.
                if self._closed.wait(0.25):
                    raise GuiRpcError("MCP session is closed", reason="session_closed")
                continue
            if reply.data["status"] != "running":
                break
        if reply.data["status"] == "failed":
            error = reply.data["error"]
            raise GuiRpcError(
                error["message"], reason=error["reason"], code=error["code"]
            )
        if reply.data["status"] != "finished":
            self._publish(
                status=reply.data["status"],
                phase="analysis"
                if reply.data["status"] == "interactive"
                else "terminal",
            )
            return
        self._publish(phase="writeback_read")
        writeback = self.tools.gui.send_gui_rpc(
            "tab.writeback_preview",
            {"tab_id": tab, "subtab_id": "analysis"},
            operation_handle=started["handle"],
        )
        self._publish(writeback=writeback, status="finished", phase="terminal")

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
        self._publish(phase="raw_save")
        try:
            started = self.tools.gui.send_gui_rpc(
                "tab.save_data",
                {"tab_id": tab},
                run_operation_handle=run_op,
            )
            path = started["data_path"]
            self._publish(
                op=started["handle"], raw_save=RawSave("saving", reserved_path=path)
            )
            outcome = self._await_operation(started["handle"])
            if outcome["status"] != "finished":
                self._publish(
                    raw_save=replace(
                        self.progress.raw_save,
                        status="failed",
                        operation_outcome=outcome,
                    )
                )
                raise GuiRpcError(
                    str(outcome.get("error", "Raw save failed")),
                    reason="raw_save_failed",
                )
            self._publish(raw_save=RawSave("saved", path, path, outcome))
        except Exception as error:
            if self.progress.raw_save.status != "failed":
                reason = getattr(error, "reason", None)
                unknown = reason in (
                    "gui_transport_timeout",
                    "connection_lost",
                    "message_too_large",
                )
                self._publish(
                    raw_save=replace(
                        self.progress.raw_save,
                        status="unknown" if unknown else "failed",
                    )
                )
            raise

    def needs_parameters(self, missing: list[MissingParameter]) -> None:
        self._publish(missing=missing, status="needs_parameters", phase="terminal")


class RecipeExecutions:
    """Session lifetime owner for fixed-binding recipe workers, not a scheduler."""

    def __init__(self, closed: Event) -> None:
        self._closed = closed
        self._lock = Lock()
        self._executions: dict[str, RecipeContext] = {}

    def start(
        self,
        tools: MeasureToolContext,
        recipe: str,
        run: Callable[[RecipeContext, dict[str, Any]], None],
        arguments: dict[str, Any],
    ) -> RecipeContext:
        context = RecipeContext(tools, recipe, self._closed)
        with self._lock:
            if self._closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            self._executions[context.progress.execution] = context
            context.start(run, arguments)
        return context

    def get(self, execution: str) -> RecipeContext:
        with self._lock:
            found = self._executions.get(execution)
        if found is None:
            raise GuiRpcError(
                f"unknown execution: {execution!r}", reason="unknown_execution"
            )
        return found

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            executions = list(self._executions.values())
        return [execution.snapshot() for execution in executions]

    def stop_admission(self) -> None:
        self._closed.set()
        with self._lock:
            for execution in self._executions.values():
                execution.wake()

    def join(self) -> None:
        with self._lock:
            executions = list(self._executions.values())
        for execution in executions:
            execution.join()

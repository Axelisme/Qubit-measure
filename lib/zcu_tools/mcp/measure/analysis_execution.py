"""Session-owned completion of one analysis operation, without replay or recovery."""

from __future__ import annotations

import base64
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from threading import Condition, Event, Lock, Thread
from typing import Any, Literal

from zcu_tools.mcp.core.reply import PngImage, ToolReply
from zcu_tools.mcp.measure.images import validated_png
from zcu_tools.mcp.measure.session import GuiConnection, GuiRpcError, MeasureMcpSession

AnalysisStage = Literal["primary", "post"]
ExecutionStatus = Literal["running", "interactive", "finished", "failed", "cancelled"]
ExecutionPhase = Literal[
    "operation", "result_read", "image_save", "figure_read", "terminal"
]
SaveStatus = Literal[
    "not_started", "not_available", "saving", "saved", "incomplete", "unknown"
]


@dataclass(frozen=True)
class AnalysisResult:
    summary: Any
    params: dict[str, Any]
    operation_state: dict[str, Any]


@dataclass(frozen=True)
class SavedImage:
    figure_name: str
    image_path: str


@dataclass(frozen=True)
class ExecutionError:
    phase: ExecutionPhase
    reason: str
    message: str
    code: str | None = None


@dataclass(frozen=True)
class ExecutionSnapshot:
    execution: str
    tab: str
    stage: AnalysisStage
    op: int
    status: ExecutionStatus = "running"
    phase: ExecutionPhase = "operation"
    cancel_requested: bool = False
    operation_outcome: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    invalidated: list[str] | None = None
    result: AnalysisResult | None = None
    interaction: dict[str, Any] | None = None
    save_status: SaveStatus = "not_started"
    saved_images: list[SavedImage] = field(default_factory=list)
    remaining_images: list[str] | None = None
    unconfirmed_image: str | None = None
    figure: str | None = None
    error: ExecutionError | None = None


def _analysis_result(
    reply: dict[str, Any], stage: AnalysisStage
) -> tuple[AnalysisResult, list[str]]:
    state = reply.get("operation_state")
    params = reply.get("params")
    if (
        "summary" not in reply
        or not isinstance(params, dict)
        or not isinstance(state, dict)
    ):
        raise GuiRpcError("invalid analysis result reply", reason="incompatible_wire")
    pane = state.get("analysis_state" if stage == "primary" else "post_analysis_state")
    names = pane.get("figure_names") if isinstance(pane, dict) else None
    if (
        not isinstance(names, list)
        or any(not isinstance(name, str) or not name for name in names)
        or len(names) != len(set(names))
    ):
        raise GuiRpcError("invalid analysis figure names", reason="incompatible_wire")
    return AnalysisResult(
        deepcopy(reply["summary"]), deepcopy(params), deepcopy(state)
    ), list(names)


class _RpcFailure(RuntimeError):
    """Keep the attempted phase distinct from the last admitted phase."""

    def __init__(self, phase: ExecutionPhase, cause: Exception) -> None:
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause


class AnalysisExecution:
    """One fixed GUI binding and detached observations of its completion."""

    def __init__(
        self,
        snapshot: ExecutionSnapshot,
        connection: GuiConnection,
        session: MeasureMcpSession,
        closed: Event,
        invalidated: list[str],
    ) -> None:
        self._snapshot = snapshot
        self._connection = connection
        self._session = session
        self._closed = closed
        self._invalidated = invalidated
        self._condition = Condition()
        self._images: tuple[PngImage, ...] = ()
        self._thread: Thread | None = None

    def snapshot(self) -> ExecutionSnapshot:
        with self._condition:
            return deepcopy(self._snapshot)

    def wait(self, timeout: float) -> ToolReply:
        """Wait locally for completion or an interactive handoff, never cancel."""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._snapshot.status
                    in ("interactive", "finished", "failed", "cancelled")
                ),
                timeout=timeout,
            )
            snapshot = deepcopy(self._snapshot)
            return ToolReply(
                asdict(snapshot), self._images, is_error=snapshot.status == "failed"
            )

    def start(self) -> None:
        """Called under the registry lock, so close cannot miss an admitted worker."""
        if self._closed.is_set():
            self._fail(GuiRpcError("MCP session is closed", reason="session_closed"))
            return
        thread = Thread(target=self._run, name=self._snapshot.execution)
        try:
            thread.start()
        except RuntimeError as exc:
            self._fail(GuiRpcError(str(exc), reason="worker_start_failed"))
        else:
            self._thread = thread

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()

    def _publish(self, **changes: Any) -> None:
        with self._condition:
            self._snapshot = replace(self._snapshot, **changes)
            self._condition.notify_all()

    def _admit(self, phase: ExecutionPhase, image: str | None = None) -> None:
        with self._condition:
            if self._closed.is_set():
                raise GuiRpcError("MCP session is closed", reason="session_closed")
            self._snapshot = replace(
                self._snapshot,
                phase=phase,
                unconfirmed_image=image,
                save_status="saving"
                if image is not None
                else self._snapshot.save_status,
            )

    def _rpc(
        self,
        phase: ExecutionPhase,
        method: str,
        params: dict[str, Any],
        *,
        image: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        try:
            return self._connection.send_gui_rpc(
                method,
                params,
                timeout_seconds=timeout,
                operation_handle=self._snapshot.op,
                before_send=lambda: self._admit(phase, image),
            )
        except Exception as exc:
            raise _RpcFailure(phase, exc) from exc

    def _fail(self, exc: Exception, phase: ExecutionPhase | None = None) -> None:
        with self._condition:
            snapshot = self._snapshot
            phase = snapshot.phase if phase is None else phase
            reason = exc.reason if isinstance(exc, GuiRpcError) else None
            code = exc.code if isinstance(exc, GuiRpcError) else None
            save_status = snapshot.save_status
            unconfirmed = snapshot.unconfirmed_image
            if phase == "image_save" and save_status != "saved":
                save_status = "incomplete"
            if unconfirmed is not None:
                known_rejection = code is not None and reason != "gui_transport_timeout"
                save_status = "incomplete" if known_rejection else "unknown"
                if known_rejection:
                    unconfirmed = None
            self._snapshot = replace(
                snapshot,
                status="failed",
                phase="terminal",
                save_status=save_status,
                unconfirmed_image=unconfirmed,
                error=ExecutionError(
                    phase, reason or "execution_failed", str(exc), code
                ),
            )
            self._condition.notify_all()

    def _run(self) -> None:
        # This is the worker isolation boundary: retain partial facts for every failure.
        try:
            if not self._await_operation():
                return
            self._complete_analysis()
        except _RpcFailure as failure:
            self._fail(failure.cause, failure.phase)
        except Exception as exc:  # noqa: BLE001 - worker boundary must publish every failure
            self._fail(exc)

    def _await_operation(self) -> bool:
        while True:
            began = time.monotonic()
            reply = self._rpc(
                "operation", "operation.await", {"timeout": 0.25}, timeout=2.25
            )
            reason = reply.get("reason")
            if reason == "completed":
                status = reply.get("status")
                if status not in ("finished", "failed", "cancelled"):
                    raise GuiRpcError(
                        "invalid operation outcome", reason="incompatible_wire"
                    )
                self._publish(operation_outcome=deepcopy(reply))
                if status == "failed":
                    raise GuiRpcError(
                        str(reply.get("error", "analysis failed")),
                        reason="analysis_failed",
                    )
                if status == "cancelled":
                    self._publish(status="cancelled", phase="terminal")
                    return False
                self._publish(invalidated=deepcopy(self._invalidated))
                return True
            if reason not in ("timeout", "user_feedback"):
                raise GuiRpcError(
                    "invalid operation await reply", reason="incompatible_wire"
                )
            # A GUI can answer immediately with user feedback. Pace observation
            # without holding the RPC lock, and let close wake the worker.
            with self._condition:
                self._condition.wait_for(
                    self._closed.is_set, max(0.0, 0.25 - (time.monotonic() - began))
                )

    def _complete_analysis(self) -> None:
        snapshot = self.snapshot()
        pane = "analysis" if snapshot.stage == "primary" else "post_analysis"
        method = (
            "tab.get_analyze_result"
            if snapshot.stage == "primary"
            else "tab.get_post_analyze_result"
        )
        reply = self._rpc("result_read", method, {"tab_id": snapshot.tab})
        result, names = _analysis_result(reply, snapshot.stage)
        self._publish(
            result=result, params=deepcopy(result.params), remaining_images=names
        )
        if not names:
            self._publish(
                save_status="not_available", status="finished", phase="terminal"
            )
            return
        saved: list[SavedImage] = []
        for index, name in enumerate(names):
            reply = self._rpc(
                "image_save",
                "tab.save_image",
                {
                    "tab_id": snapshot.tab,
                    "subtab_id": pane,
                    "figure_name": name,
                    "image_path": None,
                },
                image=name,
            )
            path = reply.get("image_path")
            if not isinstance(path, str) or not path:
                raise GuiRpcError(
                    "invalid saved image path", reason="incompatible_wire"
                )
            saved.append(SavedImage(name, path))
            self._publish(
                saved_images=list(saved),
                remaining_images=names[index + 1 :],
                unconfirmed_image=None,
            )
        self._publish(save_status="saved")
        reply = self._rpc(
            "figure_read", "tab.get_figure", {"tab_id": snapshot.tab, "subtab_id": pane}
        )
        encoded = reply.get("png_b64")
        if not isinstance(encoded, str):
            raise GuiRpcError(
                "invalid analysis preview reply", reason="incompatible_wire"
            )
        image = validated_png(base64.b64decode(encoded, validate=True))
        path = self._session.write_png(image.data)
        with self._condition:
            self._images = (image,)
            self._snapshot = replace(
                self._snapshot, figure=str(path), status="finished", phase="terminal"
            )
            self._condition.notify_all()


class AnalysisExecutions:
    """Session lifetime owner; each opaque operation gets one completion owner."""

    def __init__(self, session: MeasureMcpSession, closed: Event) -> None:
        self._session = session
        self._closed = closed
        self._lock = Lock()
        self._next_id = 1
        self._by_op: dict[int, AnalysisExecution] = {}

    def start(
        self,
        connection: GuiConnection,
        tab: str,
        stage: AnalysisStage,
        started: dict[str, Any],
    ) -> AnalysisExecution:
        """Retain the delivered start receipt, even when close wins admission."""
        with self._lock:
            op = started["handle"]
            if op in self._by_op:
                return self._by_op[op]
            execution = AnalysisExecution(
                ExecutionSnapshot(
                    execution=f"analysis-{self._next_id}",
                    tab=tab,
                    stage=stage,
                    op=op,
                    params=deepcopy(started["params"]),
                ),
                connection,
                self._session,
                self._closed,
                deepcopy(started["invalidated_on_success"]),
            )
            self._next_id += 1
            self._by_op[op] = execution
            execution.start()
            return execution

    def stop_admission(self) -> None:
        """Permanently reject new workers and wake existing ones."""
        self._closed.set()
        with self._lock:
            for execution in self._by_op.values():
                execution.wake()

    def join(self) -> None:
        """Join admitted workers after transport disconnect, before PNG cleanup."""
        with self._lock:
            executions = list(self._by_op.values())
        for execution in executions:
            execution.join()

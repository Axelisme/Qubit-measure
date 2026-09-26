"""App-facing generic operation control facet for driving adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from zcu_tools.gui.expected_error import FailedPreconditionError

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.services.run_analyze_control import (
        RunAnalyzeControlPort,
    )
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.operation_handles import AwaitResult, OperationOutcome
    from zcu_tools.gui.session.pbar_host import ProgressBarModel


class OperationAwaitPort(Protocol):
    """Thread-safe await surface consumed by operation control."""

    def await_known_outcome(
        self, operation_id: int, timeout: float, /
    ) -> AwaitResult: ...
    def known_outcome(self, operation_id: int, /) -> OperationOutcome | None: ...
    def has_cancel_hook(self, operation_id: int, /) -> bool: ...


class OperationProgressPort(Protocol):
    """Operation-scoped progress read surface consumed by operation control."""

    def bars_for_operation(
        self, operation_id: int, /
    ) -> tuple[tuple[int, ProgressBarModel], ...]: ...


@dataclass(frozen=True, slots=True)
class ActiveOperation:
    op: int
    tab: str | None
    kind: Literal["run", "analyze", "device", "save"]


class OperationControlPort(Protocol):
    """App-facing op-agnostic operation handle/progress surface."""

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult:
        """Block on a known async operation handle from an off-main RPC worker."""
        ...

    def get_operation_progress(
        self, operation_id: int
    ) -> tuple[tuple[int, ProgressBarModel], ...]:
        """Return live progress bars for any operation id."""
        ...

    def active_operations(self) -> tuple[ActiveOperation, ...]:
        """Owner-thread projection of every live operation, regardless of origin."""
        ...

    def cancel_operation(
        self, operation_id: int
    ) -> Literal["cancelling", "cancelled", "finished"]:
        """Request cancellation, or report an already failed outcome as an error."""
        ...


class OperationControlFacet:
    """Composite adapter over operation handles and operation-scoped progress."""

    def __init__(
        self,
        *,
        handles: OperationAwaitPort,
        progress: OperationProgressPort,
        run_analyze: RunAnalyzeControlPort,
        device: DeviceControlPort,
    ) -> None:
        self._handles = handles
        self._progress = progress
        self._run_analyze = run_analyze
        self._device = device

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult:
        return self._handles.await_known_outcome(operation_id, timeout)

    def get_operation_progress(
        self, operation_id: int
    ) -> tuple[tuple[int, ProgressBarModel], ...]:
        return self._progress.bars_for_operation(operation_id)

    def cancel_operation(
        self, operation_id: int
    ) -> Literal["cancelling", "cancelled", "finished"]:
        """Address the existing domain hook by its live handle on the owner thread."""
        outcome = self._handles.known_outcome(operation_id)
        if outcome is not None:
            if outcome.status == "failed":
                raise FailedPreconditionError(
                    f"operation {operation_id} failed: {outcome.error or 'unknown failure'}",
                    reason_code="operation_failed",
                )
            return "cancelled" if outcome.status == "cancelled" else "finished"
        if not self._handles.has_cancel_hook(operation_id):
            raise FailedPreconditionError(
                f"operation {operation_id} has no cancellation point",
                reason_code="not_cancellable",
            )
        for op in self._run_analyze.active_tab_operations():
            if op.op != operation_id:
                continue
            if op.kind == "run":
                if not self._run_analyze.cancel_run():
                    raise RuntimeError("run lost its active operation")
            elif not self._run_analyze.cancel_analyze(op.tab):
                raise RuntimeError("interactive analyze lost its active operation")
            return "cancelling"
        for op in self._device.get_active_device_operations():
            if op.token == operation_id:
                self._device.cancel_device_operation(op.device_name)
                return "cancelling"
        raise RuntimeError(f"live operation {operation_id} has no domain owner")

    def active_operations(self) -> tuple[ActiveOperation, ...]:
        """Merge domain owners' live handles, not an MCP-side history."""
        tab_ops = (
            ActiveOperation(op.op, op.tab, op.kind)
            for op in self._run_analyze.active_tab_operations()
        )
        device_ops = (
            ActiveOperation(op.token, None, "device")
            for op in self._device.get_active_device_operations()
        )
        return tuple(sorted((*tab_ops, *device_ops), key=lambda op: op.op))

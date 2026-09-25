"""App-facing generic operation control facet for driving adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.services.run_analyze_control import RunAnalyzeControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.operation_handles import AwaitResult


class OperationAwaitPort(Protocol):
    """Thread-safe await surface consumed by operation control."""

    def await_outcome(
        self, operation_id: int, timeout: float, /
    ) -> AwaitResult | None: ...


class OperationProgressPort(Protocol):
    """Operation-scoped progress read surface consumed by operation control."""

    def bars_for_operation(self, operation_id: int, /) -> tuple: ...


@dataclass(frozen=True, slots=True)
class ActiveOperation:
    op: int
    tab: str | None
    kind: Literal["run", "analyze", "device", "save"]


class OperationControlPort(Protocol):
    """App-facing op-agnostic operation handle/progress surface."""

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        """Block on any async operation handle from an off-main RPC worker."""
        ...

    def get_operation_progress(self, operation_id: int) -> tuple:
        """Return live progress bars for any operation id."""
        ...

    def active_operations(self) -> tuple[ActiveOperation, ...]:
        """Owner-thread projection of every live operation, regardless of origin."""
        ...

    def cancel_operation(self, operation_id: int) -> Literal["cancelling", "cancelled", "finished"]:
        """Request cancellation by op through its owning domain's existing hook."""
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

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        return self._handles.await_outcome(operation_id, timeout)

    def get_operation_progress(self, operation_id: int) -> tuple:
        return self._progress.bars_for_operation(operation_id)

    def cancel_operation(self, operation_id: int) -> Literal["cancelling", "cancelled", "finished"]:
        raise NotImplementedError("cancel by operation is not implemented")

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

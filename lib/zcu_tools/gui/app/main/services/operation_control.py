"""App-facing generic operation control facet for driving adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from zcu_tools.gui.session.operation_handles import AwaitResult


class OperationAwaitPort(Protocol):
    """Thread-safe await surface consumed by operation control."""

    def await_outcome(
        self, operation_id: int, timeout: float, /
    ) -> AwaitResult | None: ...


class OperationProgressPort(Protocol):
    """Operation-scoped progress read surface consumed by operation control."""

    def bars_for_operation(self, operation_id: int, /) -> tuple: ...


class OperationControlPort(Protocol):
    """App-facing op-agnostic operation handle/progress surface."""

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        """Block on any async operation handle from an off-main RPC worker."""
        ...

    def get_operation_progress(self, operation_id: int) -> tuple:
        """Return live progress bars for any operation id."""
        ...


class OperationControlFacet:
    """Composite adapter over operation handles and operation-scoped progress."""

    def __init__(
        self, *, handles: OperationAwaitPort, progress: OperationProgressPort
    ) -> None:
        self._handles = handles
        self._progress = progress

    def await_operation(self, operation_id: int, timeout: float) -> AwaitResult | None:
        return self._handles.await_outcome(operation_id, timeout)

    def get_operation_progress(self, operation_id: int) -> tuple:
        return self._progress.bars_for_operation(operation_id)

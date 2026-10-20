"""OperationGate — measure-gui's app-local hardware exclusion wrapper."""

from __future__ import annotations

from enum import Enum

from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.ports import OperationConflictError

__all__ = ["OperationKind", "OperationConflictError", "OperationGate"]


class OperationKind(str, Enum):
    """Measure-gui's app-specific operation kinds."""

    RUN = "run"
    STARTUP = "startup"


class OperationGate(RunBlocksHardwareGate):
    """Measure-gui hardware exclusion policy.

    The conflict matrix lives in the shared session implementation; this wrapper
    preserves the app-local import path and RUN enum value.
    """

    def __init__(self, bus: BaseEventBus) -> None:
        super().__init__(run_kind=OperationKind.RUN, bus=bus)

    def ensure_can_start(self, kind: str, *, resource_id: str | None = None) -> None:
        """Keep startup exclusive with Run, not with script hardware commands.

        Called on-owner by OperationRunner and normal Run admission. Existing
        session hardware conflicts remain unchanged; conflicts raise
        OperationConflictError before any handle or resource is allocated.
        """
        if kind == OperationKind.RUN and self.has_active(OperationKind.STARTUP):
            raise OperationConflictError("Cannot start run: startup is active")
        if kind == OperationKind.STARTUP and (
            self.has_active(OperationKind.RUN) or self.has_active(OperationKind.STARTUP)
        ):
            raise OperationConflictError(
                "Cannot start startup: run or startup is active"
            )
        super().ensure_can_start(kind, resource_id=resource_id)

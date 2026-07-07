"""OperationGate — autofluxdep-gui's app-local hardware exclusion wrapper."""

from __future__ import annotations

from enum import Enum

from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.ports import OperationConflictError

__all__ = ["OperationKind", "OperationConflictError", "OperationGate"]


class OperationKind(str, Enum):
    """autofluxdep-gui's app-specific operation kinds."""

    RUN = "run"


class OperationGate(RunBlocksHardwareGate):
    """autofluxdep-gui hardware exclusion policy.

    The conflict matrix lives in the shared session implementation; this wrapper
    preserves the app-local import path and RUN enum value.
    """

    def __init__(self) -> None:
        super().__init__(run_kind=OperationKind.RUN)

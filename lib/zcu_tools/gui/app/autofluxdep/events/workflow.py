"""Workflow-editing domain event definitions for autofluxdep-gui.

The workflow domain owns the ``WorkflowEvent`` enum and the payloads for
node-list edits and flux-value changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from zcu_tools.gui.event_bus import BasePayload


class WorkflowEvent(str, Enum):
    """Workflow-editing event names (wire names are the enum values)."""

    WORKFLOW_CHANGED = "workflow_changed"
    FLUX_CHANGED = "flux_changed"


@dataclass(frozen=True)
class _WorkflowPayload(BasePayload):
    """Base for workflow-domain EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[WorkflowEvent]


@dataclass(frozen=True)
class WorkflowChangedPayload(_WorkflowPayload):
    """Payload for WORKFLOW_CHANGED: nodes added/removed/reordered/renamed."""

    EVENT: ClassVar[WorkflowEvent] = WorkflowEvent.WORKFLOW_CHANGED
    name: str | None = None  # the affected node, or None for whole-list edits


@dataclass(frozen=True)
class FluxChangedPayload(_WorkflowPayload):
    """Payload for FLUX_CHANGED: the flux-point set was replaced."""

    EVENT: ClassVar[WorkflowEvent] = WorkflowEvent.FLUX_CHANGED
    count: int  # number of flux points now set

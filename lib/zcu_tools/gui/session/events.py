"""Session event vocabulary — the data / SoC / device / predictor events.

These are the events every measurement-session app raises as the session core
changes: the active context's md/ml, the SoC connection, the device set, and the
freq-predictor. They build on the shared :class:`BaseEventBus` (keyed by payload
*type*); each payload carries its own ``EVENT`` tag (a :class:`SessionEvent`) used
only for wire serialisation. Experiment-surface events (tabs / runs / node-sweep)
stay in each app's own event module.

Import-clean: only TYPE_CHECKING references to the session ``SocHandle`` surfaces
and source diagnostics, so this sits in ``gui/session`` below the apps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Literal

from zcu_tools.gui.event_bus import BasePayload, OriginKind

if TYPE_CHECKING:
    from zcu_tools.gui.session.context_types import (
        ComponentFocus,
        ContextSourceBinding,
        ContextSourceFault,
    )
    from zcu_tools.gui.session.types import SocCfgHandle, SocHandle


class SessionEvent(str, Enum):
    """Session-core event names (the wire-name vocabulary)."""

    MD_CHANGED = "md_changed"  # published point content/health changed
    ML_CHANGED = "ml_changed"  # published library content/health changed
    CONTEXT_SWITCHED = "context_switched"  # parameter binding switched
    COMPONENT_FOCUS_CHANGED = "component_focus_changed"
    SOC_CHANGED = "soc_changed"  # soc/soccfg connection changed
    PREDICTOR_CHANGED = "predictor_changed"  # predictor state or values changed
    DEVICE_CHANGED = "device_changed"  # device registered or dropped
    DEVICE_SETUP_STARTED = "device_setup_started"  # payload: DeviceSetupStartedPayload
    DEVICE_SETUP_FINISHED = (
        "device_setup_finished"  # payload: DeviceSetupFinishedPayload
    )
    HARDWARE_GATE_CHANGED = "hardware_gate_changed"
    CONNECTION_FINISHED = "connection_finished"
    SIMULATED_ENVIRONMENT_FINISHED = "simulated_environment_finished"
    DEVICE_OPERATION_FINISHED = "device_operation_finished"


@dataclass(frozen=True)
class SessionPayload(BasePayload):
    """Base for all session-core EventBus payloads. Subclasses set ``EVENT``."""

    EVENT: ClassVar[SessionEvent]


@dataclass(frozen=True)
class SimulatedEnvironmentFinishedPayload(SessionPayload):
    """The complete simulated-environment attempt has reached a terminal state."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.SIMULATED_ENVIRONMENT_FINISHED
    success: bool
    error_message: str | None = None


@dataclass(frozen=True)
class MdChangedPayload(SessionPayload):
    """Installed point content/health fact, without a writable resource.

    binding_id is the current internal source ID. content_changed distinguishes
    data change from health-only publication. fault is the detached diagnostic,
    or None after success. Observers read the already installed projection.
    """

    EVENT: ClassVar[SessionEvent] = SessionEvent.MD_CHANGED
    binding_id: int
    content_changed: bool
    fault: ContextSourceFault | None


@dataclass(frozen=True)
class ContextSwitchedPayload(SessionPayload):
    """Installed binding fact, emitted after parameters and focus are published.

    binding_id is the service-local source ID, or None for EMPTY.
    sources is the ordered persistent source binding, or None without sources.
    Neither field grants write access or changes a captured consumer's target.
    """

    EVENT: ClassVar[SessionEvent] = SessionEvent.CONTEXT_SWITCHED
    binding_id: int | None
    sources: ContextSourceBinding | None


@dataclass(frozen=True)
class ComponentFocusChangedPayload(SessionPayload):
    """Installed session selection; focus carries target, name, and ordered options."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.COMPONENT_FOCUS_CHANGED
    focus: ComponentFocus


@dataclass(frozen=True)
class MlChangedPayload(SessionPayload):
    """Installed library content/health fact, without a writable resource.

    binding_id is the current internal source ID. content_changed distinguishes
    data change from health-only publication. fault is the detached diagnostic,
    or None after success. Observers read the already installed projection.
    """

    EVENT: ClassVar[SessionEvent] = SessionEvent.ML_CHANGED
    binding_id: int
    content_changed: bool
    fault: ContextSourceFault | None


@dataclass(frozen=True)
class SocChangedPayload(SessionPayload):
    """Payload for SOC_CHANGED: soc/soccfg connection changed."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.SOC_CHANGED
    soc: SocHandle | None
    soccfg: SocCfgHandle | None
    is_mock: bool = False


@dataclass(frozen=True)
class PredictorChangedPayload(SessionPayload):
    """Payload for PREDICTOR_CHANGED: predictor state or values changed."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.PREDICTOR_CHANGED


@dataclass(frozen=True)
class DeviceChangedPayload(SessionPayload):
    """Payload for DEVICE_CHANGED: a device was registered or dropped."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.DEVICE_CHANGED
    name: str | None = None


@dataclass(frozen=True)
class DeviceSetupStartedPayload(SessionPayload):
    """Payload for DEVICE_SETUP_STARTED: a setup began on device ``name`` (its
    progress is pollable via operation.progress, by operation_id)."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.DEVICE_SETUP_STARTED
    name: str


@dataclass(frozen=True)
class DeviceSetupFinishedPayload(SessionPayload):
    """Payload for DEVICE_SETUP_FINISHED: the setup on device ``name`` reached a
    terminal state. ``outcome`` ∈ finished/failed/cancelled; ``error_message``
    set only on failure. (Mirrors RUN_STARTED / RUN_FINISHED.)"""

    EVENT: ClassVar[SessionEvent] = SessionEvent.DEVICE_SETUP_FINISHED
    name: str
    outcome: str  # 'finished' | 'failed' | 'cancelled'
    error_message: str | None = None


@dataclass(frozen=True)
class GatePresence:
    """Read model for one active hardware exclusion lease."""

    kind: str
    origin_kind: OriginKind
    note: str
    active_for_seconds: float


@dataclass(frozen=True)
class GateChangedPayload(SessionPayload):
    """In-process hint carrying the authoritative active-lease snapshot."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.HARDWARE_GATE_CHANGED
    active: tuple[GatePresence, ...]


@dataclass(frozen=True)
class ConnectionFinishedPayload(SessionPayload):
    """In-process terminal outcome for an asynchronous SoC connection."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.CONNECTION_FINISHED
    success: bool
    is_mock: bool
    error_message: str | None = None


@dataclass(frozen=True)
class DeviceOperationFinishedPayload(SessionPayload):
    """In-process terminal outcome for device connect/disconnect."""

    EVENT: ClassVar[SessionEvent] = SessionEvent.DEVICE_OPERATION_FINISHED
    name: str
    action: Literal["connect", "disconnect"]
    success: bool
    error_message: str | None = None

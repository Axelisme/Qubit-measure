"""build_session_services — construct the session-core service bundle.

An app's composition root provides the shared infrastructure (the exclusion gate,
the async-operation handles, the off-main executor, the progress hub, the project
IO adapter, the event bus, the state) and this builds the session services
(connection / context / device) on top. Each app then constructs its own
experiment-surface services around the returned bundle (measure:
``build_app_services`` adds tabs / run / analyze / save / startup / cfg_editor;
autofluxdep will add its node-sweep surface).

The app injects concrete infrastructure through the session *ports*
(``gui/session/ports``), so the session services stay app-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from zcu_tools.gui.session.services.connection import ConnectionService
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.mock_flux import MockFluxProvisioner
from zcu_tools.gui.session.services.startup import StartupService

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.ports import (
        BackgroundExecutor,
        DriverFactoryPort,
        ExclusionGate,
        ProgressHub,
        ProjectIOPort,
    )
    from zcu_tools.gui.session.state import SessionState


@dataclass(frozen=True)
class SessionServices:
    """The session-core services every measurement-session app builds on."""

    connection: ConnectionService
    context: ContextService
    device: DeviceService
    startup: StartupService
    # FLUX-AWARE-MOCK: self-wiring provisioner (subscribes SOC_CHANGED). Held on
    # the bundle so its lifetime matches the services + a test can inspect it; the
    # apps need no provisioning code of their own.
    mock_flux: MockFluxProvisioner


def build_session_services(
    *,
    state: SessionState,
    bus: BaseEventBus,
    gate: ExclusionGate,
    handles: OperationHandles,
    background: BackgroundExecutor,
    progress: ProgressHub,
    io_manager: ProjectIOPort,
    driver_factory: DriverFactoryPort | None = None,
) -> SessionServices:
    """Construct the session services from the app-provided infrastructure.

    ``gate`` / ``background`` / ``progress`` are the app's concrete exclusion gate,
    off-main executor and progress hub (injected via their session ports);
    ``io_manager`` is the project-IO adapter; ``driver_factory`` defaults to the
    device service's built-in hardware factory when omitted.
    """
    device = DeviceService(
        bus,
        state,
        gate,
        background,
        progress,
        handles=handles,
        driver_factory=driver_factory,
    )
    connection = ConnectionService(state, bus, gate, handles)
    context = ContextService(state, io_manager, bus)
    # StartupService bridges the two session services it commands through their
    # ports (context bootstrap + remembered-device registration) + State prefs.
    startup = StartupService(context, device, state)
    # FLUX-AWARE-MOCK: self-subscribes to SOC_CHANGED + chains the one-shot ramp
    # off device.device_connected — both apps get mock flux provisioning for free.
    mock_flux = MockFluxProvisioner(bus, device, connection)
    return SessionServices(
        connection=connection,
        context=context,
        device=device,
        startup=startup,
        mock_flux=mock_flux,
    )

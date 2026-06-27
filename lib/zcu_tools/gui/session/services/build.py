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
from typing import TYPE_CHECKING

from zcu_tools.gui.session.services.connection import SoCConnectionService
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.mock_flux import MockFluxProvisioner
from zcu_tools.gui.session.services.predictor import PredictorService
from zcu_tools.gui.session.services.startup import StartupService
from zcu_tools.gui.session.value_lookup import ValueLookup, ValueRegistry

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.operation_handles import OperationHandles
    from zcu_tools.gui.session.operation_runner import OperationRunner
    from zcu_tools.gui.session.ports import (
        BackgroundExecutor,
        DeviceRegistryPort,
        DriverFactoryPort,
        ExclusionGate,
        ProgressHub,
        ProjectIOPort,
    )
    from zcu_tools.gui.session.state import SessionState


@dataclass(frozen=True)
class SessionServices:
    """The session-core services every measurement-session app builds on."""

    soc_connection: SoCConnectionService
    predictor: PredictorService
    context: ContextService
    device: DeviceService
    startup: StartupService
    values: ValueLookup


def build_session_services(
    *,
    state: SessionState,
    bus: BaseEventBus,
    gate: ExclusionGate,
    handles: OperationHandles,
    background: BackgroundExecutor,
    progress: ProgressHub,
    io_manager: ProjectIOPort,
    runner: OperationRunner,
    driver_factory: DriverFactoryPort | None = None,
    device_registry: DeviceRegistryPort | None = None,
) -> SessionServices:
    """Construct the session services from the app-provided infrastructure.

    ``gate`` / ``background`` / ``progress`` are the app's concrete exclusion gate,
    off-main executor and progress hub (injected via their session ports);
    ``io_manager`` is the project-IO adapter; ``runner`` is the shared
    OperationRunner (ADR-0026 §1) used by DeviceService for operation lifecycle;
    ``driver_factory`` defaults to the device service's built-in hardware factory
    when omitted; ``device_registry`` defaults to the ``GlobalDeviceRegistryAdapter``
    (production singleton) when omitted — tests inject an in-memory fake.
    """
    value_registry = ValueRegistry()
    device = DeviceService(
        bus,
        state,
        gate,
        background,
        runner,
        handles,
        driver_factory=driver_factory,
        device_registry=device_registry,
    )
    soc_connection = SoCConnectionService(state, bus, gate, handles, runner)
    predictor = PredictorService(state, bus)
    context = ContextService(state, io_manager, bus, values=value_registry)
    # StartupService bridges the two session services it commands through their
    # ports (context bootstrap + remembered-device registration) + State prefs.
    startup = StartupService(context, device, state)
    # FLUX-AWARE-MOCK: self-subscribes to SOC_CHANGED + chains the one-shot ramp
    # off device.device_connected — both apps get mock flux provisioning for free.
    # Not exposed on the returned bundle: nothing reads it. Its lifetime is anchored
    # by the strong reference the EventBus holds to its bound subscriber (and the
    # device_connected signal connection), both of which outlive the bundle, so the
    # provisioner survives despite not being a field here.
    MockFluxProvisioner(bus, device, predictor)
    return SessionServices(
        soc_connection=soc_connection,
        predictor=predictor,
        context=context,
        device=device,
        startup=startup,
        values=value_registry,
    )

"""build_session_services — construct the session-core service bundle.

An app's composition root provides the shared infrastructure (the exclusion gate,
the async-operation handles, the off-main executor, the progress hub, the project
IO adapter, the event bus, the state) and this builds the session services
(connection / context / device) on top. Measure constructs its experiment-surface services around the returned
bundle: ``build_app_services`` adds tabs / run / analyze / save / cfg_editor.

The app injects concrete infrastructure through the session *ports*
(``gui/session/ports``), so the session services stay app-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.device_control import DeviceControlFacet
from zcu_tools.gui.session.operation_runner import OperationInfrastructure
from zcu_tools.gui.session.predictor_control import PredictorControlFacet
from zcu_tools.gui.session.progress_control import ProgressControlFacet
from zcu_tools.gui.session.services.arb_waveform import ArbWaveformService
from zcu_tools.gui.session.services.connection import SoCConnectionService
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.predictor import ParamsFor, PredictorService
from zcu_tools.gui.session.services.project_settings import ProjectSettingsService
from zcu_tools.gui.session.services.value_sources import ValueSourceBinder
from zcu_tools.gui.session.setup_control import SetupControlFacet
from zcu_tools.gui.session.value_lookup import ValueLookup, ValueRegistry

if TYPE_CHECKING:
    from zcu_tools.cfg_model import WaveformAssetInfo
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.aliases import AliasRow
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.operation_runner import OperationRunner
    from zcu_tools.gui.session.ports import (
        DeviceRegistryPort,
        DriverFactoryPort,
    )
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.progress_control import ProgressControlPort
    from zcu_tools.gui.session.services.progress import ProgressService
    from zcu_tools.gui.session.setup_control import SetupControlPort
    from zcu_tools.gui.session.state import SessionState


@dataclass(frozen=True)
class SessionServices:
    """Immutable session services and the infrastructure shared by their callers.

    Fields:
        runner: Injected operation lifecycle owner for startup, handles, optional
            exclusion/progress, cancellation hooks and terminal cleanup. Begin
            operations on the session owner thread; domain policy stays with callers.
        progress: Injected session progress owner. Publishes live bars by operation
            and owner id, with reads and listener delivery on the owner thread.
        soc_connection: SoC connection commands and cached connection queries.
        predictor: Predictor installation and calculation commands.
        predictor_control: Narrow predictor command/query/event facade.
        progress_control: Narrow owner-keyed progress query/subscription facade.
        context: Entry/point binding and fixed parameter capture owner.
        context_control: Narrow parameter and alias command/query facade.
        device: Device lifecycle commands and cached device queries.
        device_control: Narrow device command/query/progress/event facade.
        setup_control: Entry/point setup and independent SoC command facade.
        settings: Entry opening/creation and remembered setup preferences.
        values: Read-only lookup of the session value-source registry.
        value_sources: Binder publishing cached owner-specific value sources.
        arb_waveform: Qubit-scoped arbitrary waveform assets of the session
            database, shared by every host's Inspect and ARB dialog.

    ``runner`` and ``progress`` are the exact instances supplied through
    ``build_session_services``; this bundle does not create or replace them.
    """

    runner: OperationRunner
    progress: ProgressService
    soc_connection: SoCConnectionService
    predictor: PredictorService
    predictor_control: PredictorControlPort
    progress_control: ProgressControlPort
    context: ContextService
    context_control: ContextControlPort
    device: DeviceService
    device_control: DeviceControlPort
    setup_control: SetupControlPort
    settings: ProjectSettingsService
    values: ValueLookup
    value_sources: ValueSourceBinder
    arb_waveform: ArbWaveformService


# tolerated: Asset binding, entry defaults and optional driver/predictor ports have no shared caller-owned record.
def build_session_services(  # noqa: PLR0913
    *,
    state: SessionState,
    bus: BaseEventBus,
    operations: OperationInfrastructure,
    progress: ProgressService,
    bind_assets: Callable[[Path], WaveformAssetInfo],
    project_root: str = "",
    driver_factory: DriverFactoryPort | None = None,
    device_registry: DeviceRegistryPort | None = None,
    alias_rows: tuple[AliasRow, ...] = (),
    params_for: ParamsFor | None = None,
) -> SessionServices:
    """Construct the session services from the app-provided infrastructure.

    ``operations`` forwards the app's same exclusion gate, off-main executor,
    runner and handles (with driven adapters injected via session ports).
    ``progress`` is the shared ProgressService used by operation clients and the
    device-control facet;
    ``bind_assets`` binds each entry's explicit waveform asset directory without
    a global Database selection. ``operations.runner`` is the shared
    OperationRunner (ADR-0066) used by DeviceService for operation lifecycle;
    ``project_root`` anchors generated result/database paths and result-scope
    discovery. The returned bundle retains the exact runner and ``progress``
    instances so app-specific operation clients can use the same infrastructure;
    ``driver_factory`` defaults to the device service's built-in hardware factory
    when omitted; ``device_registry`` defaults to a new session-owned
    DeviceManager. Callers sharing drivers must supply the same registry.
    alias_rows supplies the caller's ordered fallback for each entry without a
    saved alias table. ContextService initializes an empty primary row on that
    entry's first nonempty point from the registered sole candidate. Saved tables
    are authoritative, including empty rows. Explicit Apply saves to the entry;
    ordinary set_alias_rows remains memory-only. Empty defaults enable no aliases.
    Syntax/order errors propagate at construction.
    params_for is the user-owned read-only component-to-simulator parameter
    conversion; None disables component prefill, not explicit-parameter queries.
    """
    value_registry = ValueRegistry()
    device = DeviceService(
        bus,
        state,
        operations,
        driver_factory=driver_factory,
        device_registry=device_registry,
    )
    soc_connection = SoCConnectionService(
        state, bus, operations.gate, operations.handles, operations.runner
    )
    progress_control = ProgressControlFacet(progress)
    context = ContextService(
        state,
        bus,
        values=value_registry,
        bind_assets=bind_assets,
        alias_rows=alias_rows,
    )
    context_control = ContextControlFacet(context=context)
    predictor = PredictorService(context_control, params_for)
    predictor_control = PredictorControlFacet(predictor)
    device_control = DeviceControlFacet(bus=bus, device=device, progress=progress)
    value_sources = ValueSourceBinder(state=state, bus=bus, registry=value_registry)
    # ProjectSettingsService bridges the two session services it commands through
    # their ports (project context + remembered-device registration) + State prefs.
    settings = ProjectSettingsService(
        context,
        device,
        state,
        result_root=Path(project_root) / "result",
        database_root=Path(project_root) / "Database",
    )
    setup_control = SetupControlFacet(
        bus=bus,
        settings=settings,
        context=context_control,
        connection=soc_connection,
        device=device_control,
    )
    return SessionServices(
        runner=operations.runner,
        progress=progress,
        soc_connection=soc_connection,
        predictor=predictor,
        predictor_control=predictor_control,
        progress_control=progress_control,
        context=context,
        context_control=context_control,
        device=device,
        device_control=device_control,
        setup_control=setup_control,
        settings=settings,
        values=value_registry,
        value_sources=value_sources,
        arb_waveform=ArbWaveformService(state),
    )

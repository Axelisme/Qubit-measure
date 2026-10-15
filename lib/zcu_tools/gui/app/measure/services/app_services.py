from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.app.measure.catalog import ExperimentAccess, ExperimentCatalogLoader
from zcu_tools.gui.form import FormEnv
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.alias_presets import AliasPresets, validate_alias_presets
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.arb_waveform import ArbWaveformService
from zcu_tools.gui.session.services.build import build_session_services
from zcu_tools.gui.session.services.connection import SoCConnectionService
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.predictor import ParamsFor, PredictorService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.services.project_settings import ProjectSettingsService
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from .analyze import AnalyzeService
from .experiment_reload import ExperimentReloadService
from .form_contracts import source_basis
from .guard import GuardService
from .load import LoadService
from .operation_control import OperationControlFacet
from .operation_gate import OperationGate
from .post_analyze import PostAnalyzeService
from .run import RunService
from .run_analyze_control import RunAnalyzeControlFacet
from .save import SaveService
from .save_control import SaveControlFacet
from .tab import TabService
from .tab_control import TabControlFacet
from .tab_form import TabFormResources, TabFormSources
from .workspace import WorkspaceService
from .writeback import WritebackService
from .writeback_control import WritebackControlFacet

if TYPE_CHECKING:
    from zcu_tools.gui.app.measure.registry import Registry
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus
    from zcu_tools.gui.session.context_control import ContextControlPort
    from zcu_tools.gui.session.device_control import DeviceControlPort
    from zcu_tools.gui.session.ports import (
        DeviceRegistryPort,
        OwnerScheduler,
        ProgressTransport,
    )
    from zcu_tools.gui.session.predictor_control import PredictorControlPort
    from zcu_tools.gui.session.progress_control import ProgressControlPort
    from zcu_tools.gui.session.setup_control import SetupControlPort

    from .operation_control import OperationControlPort
    from .run_analyze_control import RunAnalyzeControlPort, RunAnalyzeRenderHost
    from .save_control import SaveControlPort
    from .tab_control import TabControlPort
    from .writeback_control import WritebackControlPort


@dataclass(frozen=True)
class AppServices:
    """Immutable bundle of constructed domain services.

    Owns service construction and wiring so the Controller does not. The
    Controller holds this bundle and exposes services through its façade; the
    remote dispatch path reaches the same instances through the Controller.
    A single OperationGate (Exclusion) is shared across run / connection / device
    so hardware exclusion is global; a single OperationHandles (the async
    Handle / Cancel facet) is shared across those plus analyze (handle-only, no
    exclusion) so operation_id / await / cancel are uniform (ADR-0066).
    A single OperationRunner (the kind-agnostic lifecycle mechanism, ADR-0066)
    is shared by run / analyze / post-analyze / device-setup operations.
    """

    experiment_access: ExperimentAccess
    experiment_reload: ExperimentReloadService
    operation_gate: OperationGate
    operation_control: OperationControlPort
    handles: OperationHandles
    background: BackgroundRunner
    progress: ProgressService
    guard: GuardService
    device: DeviceService
    device_control: DeviceControlPort
    soc_connection: SoCConnectionService
    predictor: PredictorService
    predictor_control: PredictorControlPort
    progress_control: ProgressControlPort
    context: ContextService
    context_control: ContextControlPort
    setup_control: SetupControlPort
    tab: TabService
    tab_forms: TabFormResources
    tab_control: TabControlPort
    run_analyze_control: RunAnalyzeControlPort
    load: LoadService
    run: RunService
    analyze: AnalyzeService
    post_analyze: PostAnalyzeService
    save: SaveService
    save_control: SaveControlPort
    writeback: WritebackService
    writeback_control: WritebackControlPort
    workspace: WorkspaceService
    settings: ProjectSettingsService
    arb_waveform: ArbWaveformService


def build_app_services(
    *,
    state: State,
    bus: EventBus,
    registry: Registry,
    progress_transport: ProgressTransport,
    resource_versions: Callable[[], Mapping[str, int]],
    render_host: Callable[[], RunAnalyzeRenderHost | None],
    owner_scheduler: OwnerScheduler,
    project_root: str,
    catalog_loader: ExperimentCatalogLoader | None = None,
    params_for: ParamsFor | None = None,
    alias_presets: AliasPresets | None = None,
    device_registry: DeviceRegistryPort | None = None,
) -> AppServices:
    """Construct and wire every domain service into a frozen bundle.

    ``alias_presets`` maps names to ordered caller-owned rows. Omitted or None
    means no catalog and an empty initial table. A supplied mapping must contain
    "default" (possibly empty), or raise ValueError. All tables are validated;
    default initializes entries only when no saved aliases exist. Saved empty
    and nonempty tables take precedence. Softlinks and unbound rows are kept;
    the first nonempty component snapshot uses the role registry's unique focus
    candidate for the primary empty row. Explicit table edits disable that
    one-time initialization. Invalid syntax, duplicate aliases or forward links
    raise InvalidInputError(alias_invalid) during construction. No device or
    cache lookup is performed, and each bundle owns its own alias table.
    """
    validate_alias_presets(alias_presets)
    access = ExperimentAccess()
    operation_gate = OperationGate(bus)
    handles = OperationHandles()
    background = BackgroundRunner()
    progress = ProgressService(progress_transport)
    # OperationRunner: the kind-agnostic lifecycle mechanism (ADR-0066).
    # Shared by run / FIT-analyze / post-analyze / device ops. Interactive analyze
    # does not use runner (main-thread-user-paced, stage2c_spec.md).
    runner = OperationRunner(operation_gate, handles, progress, background, bus)
    session = build_session_services(
        state=state,
        bus=bus,
        device_registry=device_registry,
        gate=operation_gate,
        handles=handles,
        background=background,
        progress=progress,
        runner=runner,
        bind_assets=ArbWaveformDatabase.info_at,
        project_root=project_root,
        alias_rows=() if alias_presets is None else alias_presets["default"],
        params_for=params_for,
    )
    session.device.publish_initial_devices()
    context = session.context
    device = session.device
    arb_waveform = session.arb_waveform

    def captured_parameters(
        aliases: Mapping[str, str] | None,
    ) -> ParameterContext | None:
        if state.session_env.parameters is None and not aliases:
            return None
        return context.read_context(aliases=aliases)

    def form_sources(aliases: Mapping[str, str] | None) -> TabFormSources:
        parameters = captured_parameters(aliases)
        return TabFormSources(
            env=FormEnv(
                parameters=parameters,
                library=None if parameters is None else parameters.library,
                devices={
                    device.name: device.info.model_copy(deep=True)
                    for device in state.list_devices()
                    if device.info is not None
                },
            ),
            source_basis=source_basis(state),
        )

    tab_forms = TabFormResources(
        mutation_allowed=lambda tab_id: not state.is_tab_busy(tab_id),
    )
    writeback = WritebackService(
        ContextControlFacet(context=context),
        arb_names=arb_waveform.list_data_keys,
    )
    # TabService composes the tab render model and needs the writeback query port
    # (built above) — built after writeback (read-model dependency, ADR-0067).
    tab = TabService(
        state, registry, writeback, tab_forms, form_sources, captured_parameters
    )
    workspace = WorkspaceService(state, tab, bus)
    guard = GuardService(state)
    load = LoadService(
        state,
        writeback,
        context=ContextControlFacet(context=context),
        captured_parameters=captured_parameters,
    )
    run = RunService(
        state,
        session.runner,
        bus,
        handles,
        writeback,
        gate=operation_gate,
        devices=device.get_connected_devices,
    )
    analyze = AnalyzeService(state, session.runner, bus, writeback, handles)
    post_analyze = PostAnalyzeService(
        state, session.runner, bus, handles, writeback=writeback
    )
    save = SaveService(state, bus, access=access)
    run_analyze_control = RunAnalyzeControlFacet(
        state=state,
        bus=bus,
        guard=guard,
        tab=tab,
        load=load,
        run=run,
        analyze=analyze,
        post_analyze=post_analyze,
        render_host=render_host,
        owner_scheduler=owner_scheduler,
        run_background=lambda compute, on_done, on_error: background.submit(
            compute, on_done=on_done, on_error=on_error, run_in_pool=True
        ),
        access=access,
    )
    tab_control = TabControlFacet(
        state=state,
        tab=tab,
        workspace=workspace,
        bus=bus,
        load_tab_result=run_analyze_control.load_tab_result,
        access=access,
    )
    operation_control = OperationControlFacet(
        handles=handles,
        progress=session.progress,
        run_analyze=run_analyze_control,
        device=session.device_control,
    )
    save_control = SaveControlFacet(
        state=state, guard=guard, save=save, context=context
    )
    writeback_control = WritebackControlFacet(
        state=state,
        guard=guard,
        writeback=writeback,
        resource_versions=resource_versions,
        bus=bus,
        access=access,
    )
    reload_service = ExperimentReloadService(
        state=state,
        workspace=workspace,
        registry=registry,
        loader=catalog_loader,
        active_operations=handles.live_count,
        access=access,
    )
    return AppServices(
        experiment_access=access,
        experiment_reload=reload_service,
        operation_gate=operation_gate,
        operation_control=operation_control,
        handles=handles,
        background=background,
        progress=session.progress,
        guard=guard,
        device=device,
        device_control=session.device_control,
        soc_connection=session.soc_connection,
        predictor=session.predictor,
        predictor_control=session.predictor_control,
        progress_control=session.progress_control,
        context=context,
        context_control=session.context_control,
        setup_control=session.setup_control,
        tab=tab,
        tab_forms=tab_forms,
        tab_control=tab_control,
        run_analyze_control=run_analyze_control,
        load=load,
        run=run,
        analyze=analyze,
        # Second analysis layer (post_analysis cap) — handle-only off-main worker,
        # same runner/handles as the primary analyze (ADR-0066).
        post_analyze=post_analyze,
        save=save,
        save_control=save_control,
        writeback=writeback,
        writeback_control=writeback_control,
        workspace=workspace,
        settings=session.settings,
        arb_waveform=arb_waveform,
    )

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Protocol

from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

logger = logging.getLogger(__name__)

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .adapter import CfgSchema, ExpContext, SavePaths, SocCfgHandle, WritebackItem
from .event_bus import (
    EventBus,
    GuiEvent,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from .io_manager import IOManager
from .plot_host import FigureContainer
from .registry import Registry
from .role_catalog import RoleCatalog
from .runner import AnalyzeRunner, Runner, SaveDataRunner
from .services import (
    DEFAULT_LEFT_PANEL_WIDTH,
    ConnectDeviceRequest,
    DeviceSnapshot,
    DisconnectDeviceRequest,
    PersistedStartup,
    RestoreReport,
    SaveBothOutcome,
    SessionPersistenceError,
    SetupDeviceRequest,
    StartupConnectionRequest,
    StartupPersistenceError,
    StartupProjectRequest,
    TabViewSnapshot,
    build_app_services,
)
from .services.connection import (
    ConnectRequest,
    LoadPredictorRequest,
    PredictFreqRequest,
)
from .services.device import (
    DeviceEntry,
    DeviceSetupSnapshot,
)
from .services.ports import ContextWrites
from .services.remote.dialogs import DialogName
from .state import State

if TYPE_CHECKING:
    from .pbar_host import ProgressBarModel
    from .services.ports import ProgressTransport


# A View has two distinct down-channels from the Controller (ADR-0013):
#   - diagnostics (error / info) — fanned out to *every* attached View;
#   - render help (pbar / live container) — pulled from the *one* View that has
#     a real canvas.
# So the interface splits by channel, not lumped into a single "View".
# Render/snapshot/dialog *queries* are pulled by the RemoteControlAdapter
# through its own ``render_view`` (see RenderView), not by the Controller.

Severity = Literal["error", "info"]


class DiagnosticSink(Protocol):
    """A View that can receive a Controller diagnostic (error / info).

    Fanned out to every attached sink. Each View renders it its own way — the
    Qt window pops a dialog (error) or status bar (info); the remote adapter
    enqueues a diagnostic line. Diagnostics never go through EventBus (a channel
    that reports a fault must not be the faulty channel — ADR-0013).
    """

    def notify_diagnostic(
        self, severity: Severity, title: str, message: str
    ) -> None: ...


class RenderHost(Protocol):
    """The single canvas-bearing View the Controller asks for run/analyze Qt
    artefacts (the live figure container). Progress bars no longer come from the
    View — they are minted by ProgressService, bound to the operation."""

    def make_live_container(self, tab_id: str) -> Optional[FigureContainer]: ...


class RenderView(Protocol):
    """Pure-read View surface the RemoteControlAdapter pulls from (snapshot /
    screenshot / dialog management). Held by the adapter, not the Controller."""

    def get_view_snapshot(self) -> dict[str, object]: ...
    def take_screenshot(self, tab_id: Optional[str] = None) -> bytes: ...
    def take_figure_screenshot(self, tab_id: str) -> bytes: ...
    def take_dialog_screenshot(self, dialog_name: Any) -> bytes: ...
    def open_dialog(self, name: DialogName) -> None: ...
    def close_dialog(self, name: DialogName) -> None: ...
    def list_open_dialogs(self) -> list[DialogName]: ...
    def register_dialog(self, name: DialogName, dialog: Any) -> None: ...


class ViewProtocol(DiagnosticSink, RenderHost, RenderView, Protocol):
    """A full Qt View (``MainWindow``) implements all three channels."""


class Controller:
    """Façade for the GUI application. Delegates to domain services."""

    def __init__(
        self,
        state: State,
        runner: Runner,
        registry: Registry,
        io_manager: IOManager,
        view: Optional[ViewProtocol],
        bus: EventBus,
        role_catalog: Optional["RoleCatalog"] = None,
        progress_transport: Optional["ProgressTransport"] = None,
    ) -> None:
        self._state = state
        # Views attached as diagnostic sinks (fan-out target). A full Qt View is
        # also the single RenderHost (run/analyze Qt artefacts). The remote
        # adapter is a diagnostic-only View — it holds its own RenderView.
        self._diag_sinks: list[DiagnosticSink] = []
        self._render_host: Optional[RenderHost] = None
        if view is not None:
            self.add_view(view)
        self._bus = bus
        # Catalog of experiment-role templates (gui interface, populated by
        # experiment/v2_gui at startup). Optional so tests can construct a bare
        # Controller; create_from_role fails fast when absent.
        self._role_catalog = role_catalog

        # Construct and wire every domain service into an immutable bundle, then
        # alias them onto self for the façade's call sites.
        # ``cfg_editor_ctrl=self`` lets build_app_services build CfgEditorService
        # in the same bundle: the session only stores the Controller (as its
        # LiveModel env + ModuleLibrary registration surface) and never calls it
        # during construction, so passing the still-initialising self is safe.
        # The progress transport is the Qt marshal (a driven adapter). Default to
        # the Qt one so GUI/agent processes (which run a Qt event loop) work
        # without the entry point wiring it; tests inject a synchronous fake.
        transport: "ProgressTransport"
        if progress_transport is not None:
            transport = progress_transport
        else:
            from zcu_tools.gui.adapters.qt_progress_transport import (
                QtProgressTransport,
            )

            transport = QtProgressTransport()
        services = build_app_services(
            state=state,
            bus=bus,
            registry=registry,
            io_manager=io_manager,
            runner=runner,
            analyze_runner=AnalyzeRunner(),
            save_runner=SaveDataRunner(),
            cfg_editor_ctrl=self,
            progress_transport=transport,
        )
        self._services = services
        self._operation_gate = services.operation_gate
        self._progress_svc = services.progress
        self._guard_svc = services.guard
        self._dev_svc = services.device
        self._conn_svc = services.connection
        self._ctx_svc = services.context
        self._tab_svc = services.tab
        self._run_svc = services.run
        self._analyze_svc = services.analyze
        self._save_svc = services.save
        self._writeback_svc = services.writeback
        self._tab_view_svc = services.tab_view
        self._workspace_svc = services.workspace
        self._startup_svc = services.startup
        self._cfg_editor_svc = services.cfg_editor

        self._run_svc.run_finished.connect(self._on_run_finished)
        self._run_svc.run_failed.connect(self._on_run_failed)
        self._analyze_svc.analyze_finished.connect(self._on_analyze_finished)
        self._analyze_svc.analyze_failed.connect(self._on_analyze_failed)
        self._save_svc.save_finished.connect(self._on_save_finished)
        self._save_svc.save_failed.connect(self._on_save_failed)
        self._save_svc.save_both_finished.connect(self._on_save_both_finished)
        self._dev_svc.setup_finished.connect(self._on_device_setup_finished)
        self._dev_svc.setup_failed.connect(self._on_device_setup_failed)
        self._dev_svc.setup_cancelled.connect(self._on_device_setup_cancelled)
        self._dev_svc.device_connected.connect(self._on_device_connected)
        self._dev_svc.device_disconnected.connect(self._on_device_disconnected)
        self._dev_svc.operation_failed.connect(self._on_device_operation_failed)

    def add_view(self, view: ViewProtocol) -> None:
        """Attach a full Qt View. It is a diagnostic sink (fan-out target) and
        the single RenderHost / RenderView (run/analyze Qt artefacts + pure-read
        render queries). Attaching a second replaces the render role (last
        writer wins); diagnostic sinks accumulate."""
        if view not in self._diag_sinks:
            self._diag_sinks.append(view)
        self._render_host = view

    def add_diagnostic_sink(self, sink: DiagnosticSink) -> None:
        """Attach a diagnostic-only View (e.g. the remote adapter): it receives
        error/info fan-out but is not a RenderHost."""
        if sink not in self._diag_sinks:
            self._diag_sinks.append(sink)

    def remove_diagnostic_sink(self, sink: DiagnosticSink) -> None:
        if sink in self._diag_sinks:
            self._diag_sinks.remove(sink)

    def _notify(self, severity: Severity, title: str, message: str) -> None:
        """Fan a diagnostic out to every attached View (ADR-0013). Never via
        EventBus — diagnostics must not depend on the channel they report on."""
        for sink in list(self._diag_sinks):
            sink.notify_diagnostic(severity, title, message)

    def _info(self, message: str) -> None:
        """Transient status diagnostic (no title) — Qt status bar / wire line."""
        self._notify("info", "", message)

    def _report_persistence_error(self, title: str, error: Exception) -> None:
        self._notify("error", title, str(error))

    def _on_run_finished(self, tab_id: str, _result: object) -> None:
        # State is already updated in RunService/Runner
        self._tab_svc.initialize_tab_analyze_params(tab_id)
        self._bus.emit(
            GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
        )

    def _on_run_failed(self, _tab_id: str, error: Exception) -> None:
        self._notify("error", "Run failed", str(error))

    def _on_analyze_finished(self, tab_id: str, _result: object) -> None:
        self._bus.emit(
            GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
        )

    def _on_analyze_failed(self, _tab_id: str, error: Exception) -> None:
        self._notify("error", "Analyze failed", str(error))

    def _on_save_finished(self, tab_id: str, data_path: str) -> None:
        del tab_id
        self._info(f"Data saved to {data_path}")

    def _on_save_failed(self, tab_id: str, data_path: str, error: Exception) -> None:
        del tab_id, data_path
        self._notify("error", "Save data failed", str(error))

    def _on_save_both_finished(self, tab_id: str, outcome: SaveBothOutcome) -> None:
        del tab_id
        if outcome.data_error is None and outcome.image_error is None:
            self._info(
                f"Data saved to {outcome.data_path}; "
                f"image saved to {outcome.image_path}"
            )
            return
        if outcome.data_error is None:
            self._info(
                f"Data saved to {outcome.data_path}; image failed: {outcome.image_error}"
            )
            return
        if outcome.image_error is None:
            self._info(
                f"Data failed: {outcome.data_error}; "
                f"image saved to {outcome.image_path}"
            )
            return
        self._notify(
            "error",
            "Save Both failed",
            f"Data failed: {outcome.data_error}\nImage failed: {outcome.image_error}",
        )

    def _on_device_setup_finished(self, name: str) -> None:
        self._info(f"Device setup completed: {name}")

    def _on_device_setup_failed(self, name: str, error: str) -> None:
        self._notify(
            "error",
            f"Device setup failed: {name}",
            error,
        )

    def _on_device_setup_cancelled(self, name: str) -> None:
        self._info(f"Device setup cancelled: {name}")

    def _on_device_connected(self, req: ConnectDeviceRequest) -> None:
        # Persistence is a projection of device State, driven by StartupService's
        # DEVICE_CHANGED subscription — this handler only presents UI feedback.
        self._info(f"Device connected: {req.name}")

    def _on_device_disconnected(self, req: DisconnectDeviceRequest) -> None:
        self._info(f"Device disconnected: {req.name}")

    def _on_device_operation_failed(self, name: str, error: str) -> None:
        self._notify(
            "error",
            f"Device operation failed: {name}",
            error,
        )

    def get_bus(self) -> EventBus:
        return self._bus

    def restore_tabs_from_session(self) -> None:
        try:
            report = self._workspace_svc.restore_session()
        except SessionPersistenceError as exc:
            self._report_persistence_error("Session restore failed", exc)
            return
        self._present_restore_report(report)

    def persist_tabs_session(self) -> None:
        try:
            self._workspace_svc.persist_session()
        except SessionPersistenceError as exc:
            self._report_persistence_error("Session save failed", exc)

    def _present_restore_report(self, report: RestoreReport) -> None:
        if report.rejected_tabs:
            self._notify(
                "error",
                "Some session tabs were not restored",
                "\n".join(
                    f"{issue.subject}: {issue.message}"
                    for issue in report.rejected_tabs
                ),
            )

    # ------------------------------------------------------------------
    # ExpTab operations (TabService)
    # ------------------------------------------------------------------

    def new_tab(self, adapter_name: str) -> str:
        return self._workspace_svc.new_tab(adapter_name)

    def close_tab(self, tab_id: str) -> None:
        self._workspace_svc.close_tab(tab_id)

    def set_active_tab(self, tab_id: str) -> None:
        self._workspace_svc.set_active_tab(tab_id)

    # ------------------------------------------------------------------
    # Run flow (RunService & ContextService)
    # ------------------------------------------------------------------

    def has_project(self) -> bool:
        return self._ctx_svc.has_project()

    def has_context(self) -> bool:
        return self._ctx_svc.has_context()

    def has_startup_context(self) -> bool:
        return self._ctx_svc.has_startup_context()

    def has_active_context(self) -> bool:
        return self._ctx_svc.is_active_context()

    def get_running_tab_id(self) -> Optional[str]:
        return self._state.running_tab_id

    def has_soc(self) -> bool:
        return self._conn_svc.has_soc()

    def resources_versions(self) -> dict[str, int]:
        """Full resource-version snapshot (the resources.versions RPC payload)."""
        return self._state.version.snapshot()

    def start_run(self, tab_id: str) -> int:
        # GuardService proves context readiness + committed-cfg validity + soc
        # capability, and carries the worker payload in the permit. Both clients
        # acquire through the same path so guard logic cannot drift. Returns the
        # operation token (handle for operation.await).
        #
        # Terminal: this only launches the worker. The outcome lands on the Qt
        # main thread in ``RunService._on_run_finished`` / ``_on_run_failed``,
        # which bump ``tab:<id>:result`` (finished only), release the RUN lease
        # exactly-once, and emit ``RUN_LOCK_CHANGED`` with the outcome
        # (finished / failed / cancelled).
        permit = self._guard_svc.acquire_run_permit(tab_id)
        # Headless (no Qt RenderHost, e.g. a pure-agent process): RunService
        # tolerates a None live container. The progress factory is minted by
        # RunService from ProgressService (bound to the operation), not the View.
        host = self._render_host
        live_container = host.make_live_container(tab_id) if host is not None else None
        return self._run_svc.start_run(permit, live_container)

    def cancel_run(self) -> None:
        self._run_svc.cancel_run()

    def get_run_progress(self) -> tuple:
        return self._run_svc.get_run_progress()

    def get_tab_analyze_result(self, tab_id: str) -> object | None:
        return self._tab_svc.get_tab_analyze_result(tab_id)

    # ------------------------------------------------------------------
    # Analyze flow (TabService)
    # ------------------------------------------------------------------

    def analyze(self, tab_id: str, analyze_params_instance: object) -> None:
        permit = self._guard_svc.acquire_analyze_permit(tab_id)
        host = self._render_host
        figure_container = (
            host.make_live_container(tab_id) if host is not None else None
        )
        self._analyze_svc.start_analyze(
            permit, analyze_params_instance, figure_container
        )

    # ------------------------------------------------------------------
    # Writeback (TabService)
    # ------------------------------------------------------------------

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]:
        """Recompute the tab's writeback proposals (read-only, no permit).

        Returns [] when the tab has no run/analyze result yet.
        """
        return list(self._writeback_svc.get_tab_writeback_items(tab_id))

    def apply_writeback(self, tab_id: str) -> list[str]:
        """Apply the tab's persistent writeback draft as-is (no recompute)."""
        permit = self._guard_svc.acquire_writeback_permit(tab_id)
        return self._writeback_svc.apply_tab_writeback(permit)

    def set_writeback_item(self, tab_id: str, session_id: str, **changes: Any) -> None:
        """Edit a persistent writeback item (selected / target_name / value)."""
        self._guard_svc.acquire_writeback_permit(tab_id)
        self._writeback_svc.set_item_field(tab_id, session_id, **changes)

    # ------------------------------------------------------------------
    # Save (TabService)
    # ------------------------------------------------------------------

    def _resolve_save_paths(self, tab_id: str) -> "SavePaths":
        paths = self._tab_svc.get_tab_save_paths(tab_id)
        if paths is None:
            raise RuntimeError(
                f"Tab {tab_id!r} has no save paths configured — "
                "set paths via the Save panel or update_tab_save_paths()."
            )
        return paths

    def save_data(
        self, tab_id: str, data_path: Optional[str] = None, comment: str = ""
    ) -> None:
        permit = self._guard_svc.acquire_save_permit(tab_id)
        resolved = data_path or self._resolve_save_paths(tab_id).data_path
        self._save_svc.start_save_data(permit, resolved, comment=comment)

    def save_image(self, tab_id: str, image_path: Optional[str] = None) -> None:
        permit = self._guard_svc.acquire_save_permit(tab_id)
        resolved = image_path or self._resolve_save_paths(tab_id).image_path
        self._save_svc.save_image_sync(permit, resolved)
        self._info(f"Image saved to {resolved}")

    def save_both(
        self,
        tab_id: str,
        data_path: Optional[str] = None,
        image_path: Optional[str] = None,
        comment: str = "",
    ) -> None:
        permit = self._guard_svc.acquire_save_permit(tab_id)
        paths = self._resolve_save_paths(tab_id)
        resolved_data = data_path or paths.data_path
        resolved_image = image_path or paths.image_path
        self._save_svc.start_save_both(
            permit, resolved_data, resolved_image, comment=comment
        )

    # ------------------------------------------------------------------
    # Context / IO (ContextService)
    # ------------------------------------------------------------------

    def apply_startup_project(self, req: StartupProjectRequest) -> bool:
        try:
            self._startup_svc.apply_project(req)
        except StartupPersistenceError as exc:
            self._report_persistence_error("Startup settings save failed", exc)
            return False
        return True

    def use_context(self, label: str) -> None:
        self._ctx_svc.use_context(label)

    def new_context(
        self,
        value: Optional[float] = None,
        unit: str = "A",
        clone_from_current: bool = False,
    ) -> None:
        self._ctx_svc.new_context(value, unit, clone_from_current)

    def get_active_context_label(self) -> Optional[str]:
        return self._ctx_svc.get_active_context_label()

    def get_flux_dir(self) -> Optional[str]:
        return self._ctx_svc.get_flux_dir()

    def get_context_labels(self) -> list[str]:
        return self._ctx_svc.get_context_labels()

    def get_current_md(self) -> MetaDict:
        return self._ctx_svc.get_current_md()

    def set_md_attr(self, key: str, value: Any) -> None:
        self._ctx_svc.set_md_attr(key, value)

    def del_md_attr(self, key: str) -> None:
        self._ctx_svc.del_md_attr(key)

    def get_current_ml(self) -> ModuleLibrary:
        return self._ctx_svc.get_current_ml()

    def set_ml_module_from_schema(self, name: str, schema: CfgSchema) -> None:
        self._ctx_svc.set_ml_module_from_schema(name, schema)

    def del_ml_module(self, name: str) -> None:
        self._ctx_svc.del_ml_module(name)

    def set_ml_waveform_from_schema(self, name: str, schema: CfgSchema) -> None:
        self._ctx_svc.set_ml_waveform_from_schema(name, schema)

    def apply_writes(self, writes: "ContextWrites") -> None:
        self._ctx_svc.apply_writes(writes)

    def coerce_md_value(self, key: str, text: str) -> Any:
        return self._ctx_svc.coerce_md_value(key, text)

    def del_ml_waveform(self, name: str) -> None:
        self._ctx_svc.del_ml_waveform(name)

    def rename_ml_module(self, old: str, new: str) -> None:
        self._ctx_svc.rename_ml_module(old, new)

    def rename_ml_waveform(self, old: str, new: str) -> None:
        self._ctx_svc.rename_ml_waveform(old, new)

    # ------------------------------------------------------------------
    # Role templates — one-shot "create blank ml entry from a named role"
    # (shared by inspect UI and ml.create_from_role RPC). Editing afterwards
    # goes through the normal modify path (inspect / editor.open(from_name)).
    # ------------------------------------------------------------------

    def get_role_catalog(self) -> RoleCatalog:
        if self._role_catalog is None:
            raise RuntimeError("No role catalog is wired up.")
        return self._role_catalog

    def get_exp_context(self) -> "ExpContext":
        return self._ctx_svc.get_exp_context()

    def create_from_role(self, item_kind: str, role_id: str, name: str) -> None:
        """Seed a blank ml module/waveform from a named role and register it.

        The role's eval-aware factory produces md-linked defaults; lowering
        against the live md turns those into the md's current concrete values
        (ModuleLibrary stores concrete numbers, never md references).
        """
        from zcu_tools.gui.adapter import CfgSchema
        from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES
        from zcu_tools.gui.specs import make_waveform_spec_by_style

        if not name:
            raise RuntimeError("Entry name must not be empty.")
        entry = self.get_role_catalog().get(role_id)
        if entry.item_kind != item_kind:
            raise RuntimeError(
                f"Role {role_id!r} is a {entry.item_kind}, not a {item_kind}."
            )
        # create = new entry; a name clash is an error (the user/agent meant to
        # add, not silently overwrite an existing entry — register_module would
        # overwrite). Editing an existing entry goes through the modify path.
        self._require_new_ml_name(item_kind, name)

        ctx = self.get_exp_context()
        ref = entry.make_value(ctx)
        value = ref.value
        discriminator = self._discriminator_of(value, item_kind)
        if item_kind == "module":
            spec = _MODULE_SPEC_FACTORIES[discriminator]()
        else:
            spec = make_waveform_spec_by_style(discriminator)
        # ADR-0011: hand the un-lowered CfgSchema to the single write authority;
        # ContextService lowers (against live md) + registers. No UI-side lowering.
        schema = CfgSchema(spec=spec, value=value)
        if item_kind == "module":
            self.set_ml_module_from_schema(name, schema)
        else:
            self.set_ml_waveform_from_schema(name, schema)

    @staticmethod
    def _discriminator_of(value: Any, item_kind: str) -> str:
        """Read the type/style discriminator off a role factory's value."""
        key = "type" if item_kind == "module" else "style"
        field = value.fields.get(key)
        disc = getattr(field, "value", None)
        if not isinstance(disc, str):
            raise RuntimeError(
                f"Role value has no usable {key!r} discriminator (got {disc!r})."
            )
        return disc

    def has_ml_entry(self, item_kind: str, name: str) -> bool:
        """Whether an ml module/waveform of this name already exists."""
        ml = self.get_current_ml()
        store = ml.modules if item_kind == "module" else ml.waveforms
        return name in store

    def _require_new_ml_name(self, item_kind: str, name: str) -> None:
        """Fail fast if a create would overwrite an existing ml entry."""
        if self.has_ml_entry(item_kind, name):
            raise RuntimeError(
                f"A {item_kind} named {name!r} already exists; "
                "use Modify to change it, or pick a different name."
            )

    # ------------------------------------------------------------------
    # CfgEditor sessions (CfgEditorService) — headless ml editing for RPC
    # ------------------------------------------------------------------

    def open_cfg_editor(
        self,
        item_kind: str,
        *,
        discriminator: Optional[str] = None,
        from_name: Optional[str] = None,
        gc: bool = True,
        owner_key: Optional[str] = None,
    ) -> tuple[str, list[dict[str, object]]]:
        """Open a CfgEditor session. The ``editor.open`` RPC only uses
        ``from_name`` (edit an existing entry); ``discriminator`` (blank seed)
        remains an internal seam — creating a blank goes through
        ``create_from_role`` (``<disc>:blank`` roles), not the RPC.
        """
        return self._cfg_editor_svc.open(
            item_kind,
            discriminator=discriminator,
            from_name=from_name,
            gc=gc,
            owner_key=owner_key,
        )

    def open_seeded_cfg_editor(
        self, seed: Any, *, gc: bool = False, owner_key: Optional[str] = None
    ) -> tuple[str, list[dict[str, object]]]:
        """Open a service-owned cfg model seeded from an existing CfgSchema.

        Used by UI surfaces (tab cfg / writeback item) that own a non-ml-entry
        draft; the owning widget then ``attach``es to ``get_cfg_editor_root``.
        """
        return self._cfg_editor_svc.open_seeded(seed, gc=gc, owner_key=owner_key)

    def get_cfg_editor_root(self, editor_id: str) -> Any:
        """Return the service-owned LiveModel for a widget to ``attach`` to."""
        return self._cfg_editor_svc.get_root(editor_id)

    def teardown_cfg_editor(self, editor_id: str) -> None:
        """Tear down a UI-owned (gc=False) cfg-editor session + its LiveModel."""
        self._cfg_editor_svc.teardown(editor_id)

    def editor_id_for_owner(self, owner_key: str) -> Optional[str]:
        return self._cfg_editor_svc.editor_id_for_owner(owner_key)

    def set_cfg_editor_change_listener(self, listener: Any) -> None:
        """Wire the per-session push listener (remote layer injects this)."""
        self._cfg_editor_svc.set_change_listener(listener)

    def bump_editor_version(self, editor_id: str) -> None:
        """Bump an editor session's draft version (editor.commit guard input).

        Symmetric teardown is ``drop_editor_version`` (called from
        ``CfgEditorService._remove``): a session that ends must drop its key, or
        a stale dependency would spuriously match a retained version.
        """
        self._state.version.bump(f"editor:{editor_id}")

    def drop_editor_version(self, editor_id: str) -> None:
        """Forget an editor session's version on teardown (symmetric to tab/
        device drop). A stale dependency on a gone editor then reads version 0,
        so the guard treats it as stale rather than spuriously matching."""
        self._state.version.drop_prefix(f"editor:{editor_id}")

    def cfg_editor_set_field(
        self, editor_id: str, path: str, value: object
    ) -> dict[str, object]:
        return self._cfg_editor_svc.set_field(editor_id, path, value)

    def owner_of_editor(self, editor_id: str) -> Optional[str]:
        """The owner_key a cfg-editor session is keyed to (tab_id for tab cfg)."""
        return self._cfg_editor_svc.owner_of_editor(editor_id)

    def cfg_editor_get(self, editor_id: str) -> list[dict[str, object]]:
        return self._cfg_editor_svc.get(editor_id)

    def commit_cfg_editor(self, editor_id: str, name: str) -> None:
        self._cfg_editor_svc.commit(editor_id, name)

    def discard_cfg_editor(self, editor_id: str) -> None:
        self._cfg_editor_svc.discard(editor_id)

    def discard_cfg_editors(self, editor_ids: list[str]) -> None:
        self._cfg_editor_svc.discard_for_client(editor_ids)

    # ------------------------------------------------------------------
    # Device (DeviceService)
    # ------------------------------------------------------------------

    def start_connect_device(self, req: ConnectDeviceRequest) -> int:
        return self._dev_svc.start_connect_device(req)

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> int:
        return self._dev_svc.start_disconnect_device(req)

    def list_devices(self) -> list[DeviceEntry]:
        return self._dev_svc.list_devices()

    def list_device_names(self) -> list[str]:
        return self._dev_svc.list_device_names()

    def get_device_unit(self, name: str) -> str:
        return self._dev_svc.get_device_unit(name)

    def get_device_value_for_new_context(self, name: str) -> Optional[float]:
        return self._dev_svc.get_device_value_for_new_context(name)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        return self._dev_svc.get_device_info(name)

    def start_setup_device(self, req: SetupDeviceRequest) -> int:
        return self._dev_svc.start_setup_device(req)

    def get_active_device_setup(self) -> Optional[DeviceSetupSnapshot]:
        return self._dev_svc.get_active_setup()

    def get_device_setup_progress(self) -> tuple:
        return self._dev_svc.setup_progress()

    def cancel_device_operation(self, name: str) -> None:
        self._dev_svc.cancel_device_operation(name)

    def start_reconnect_device(self, name: str) -> None:
        self._dev_svc.start_reconnect_device(name)

    def forget_device(self, name: str) -> None:
        # Removing the device from State emits DEVICE_CHANGED, which re-projects
        # the remembered-device set onto disk via StartupService.
        self._dev_svc.forget_device(name)

    def is_memory_device(self, name: str) -> bool:
        return self._dev_svc.is_memory_device(name)

    def get_memory_device_address(self, name: str) -> Optional[str]:
        """Return the persisted address for a memory-only device, or None."""
        return self._dev_svc.get_memory_device_address(name)

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        return self._dev_svc.get_device_snapshot(name)

    def get_active_device_operation(self) -> DeviceSnapshot | None:
        return self._dev_svc.get_active_device_operation()

    def attach_progress(
        self, owner_id: str, listener: Callable[[], None]
    ) -> Callable[[], None]:
        """A View subscribes (by its own tab_id / device_name) to progress
        changes for that owner; returns a disposer. The listener fires whenever
        the owner's live operation's bars change (and across operation rotation),
        and re-reads via ``progress_bars``."""
        return self._progress_svc.attach_by_owner(owner_id, listener)

    def progress_bars(
        self, owner_id: str
    ) -> tuple[tuple[int, "ProgressBarModel"], ...]:
        """Live (handle_id, ProgressBarModel) pairs for the owner's current
        operation (empty if none live)."""
        return self._progress_svc.bars_for_owner(owner_id)

    def await_operation(self, operation_id: int, timeout: float):
        """Block until an async operation settles; return its OperationOutcome.

        Runs on an off-main IO thread (operation.await is off_main_thread) — only
        touches the gate's thread-safe registry, no main-thread-owned state.
        """
        return self._operation_gate.await_outcome(operation_id, timeout)

    # ------------------------------------------------------------------
    # Startup application workflow (StartupService)
    # ------------------------------------------------------------------

    def restore_startup_settings(self) -> None:
        try:
            self._startup_svc.restore_devices()
        except StartupPersistenceError as exc:
            self._report_persistence_error("Startup settings restore failed", exc)

    def get_persisted_startup(self) -> Optional[PersistedStartup]:
        try:
            return self._startup_svc.get_persisted()
        except StartupPersistenceError as exc:
            self._report_persistence_error("Startup settings restore failed", exc)
            return None

    def remember_startup_connection(self, req: StartupConnectionRequest) -> None:
        try:
            self._startup_svc.remember_connection(req)
        except StartupPersistenceError as exc:
            self._report_persistence_error("Startup settings save failed", exc)

    def get_persisted_left_panel_width(self) -> int:
        try:
            return self._startup_svc.get_left_panel_width()
        except StartupPersistenceError as exc:
            self._report_persistence_error("Startup settings restore failed", exc)
            return DEFAULT_LEFT_PANEL_WIDTH

    def save_left_panel_width(self, width: int) -> None:
        try:
            self._startup_svc.save_left_panel_width(width)
        except StartupPersistenceError as exc:
            self._report_persistence_error("Startup settings save failed", exc)

    # ------------------------------------------------------------------
    # Connection / Predictor (ConnectionService)
    # ------------------------------------------------------------------

    def start_connect(self, req: ConnectRequest) -> int:
        return self._conn_svc.start_connect(req)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        """Bind the single connection observer without exposing the service.

        Single-observer model: only one observer (the currently-open SetupDialog)
        should hear connection outcomes at a time. SetupDialog is re-created on
        every open (``MainWindow._make_dialog`` → popped from the registry on
        ``finished``), so a prior dialog's bound methods would otherwise stay
        connected and leak. We therefore drop **all** existing slots on these
        signals before connecting the new observer — a no-arg ``disconnect()``
        removes every connection — guaranteeing exactly the latest observer.
        """
        for signal in (
            self._conn_svc.connection_finished,
            self._conn_svc.connection_failed,
        ):
            try:
                signal.disconnect()
            except (TypeError, RuntimeError):
                pass  # no existing connections
        self._conn_svc.connection_finished.connect(on_finished)
        self._conn_svc.connection_failed.connect(on_failed)

    def load_predictor(self, req: LoadPredictorRequest) -> None:
        self._conn_svc.load_predictor(req)

    def clear_predictor(self) -> None:
        self._conn_svc.clear_predictor()

    def predict_freq(self, req: PredictFreqRequest) -> float:
        return self._conn_svc.predict_freq(req)

    def get_soccfg(self) -> Optional[SocCfgHandle]:
        return self._conn_svc.get_soccfg()

    def get_predictor(self) -> Optional[FluxoniumPredictor]:
        return self._conn_svc.get_predictor()

    def get_predictor_info(self) -> Optional[dict]:
        return self._conn_svc.get_predictor_info()

    # ------------------------------------------------------------------
    # View query interface (TabService) — strict APIs; callers must check
    # has_tab() and short-circuit if false. State / TabService raise KeyError
    # on unknown tab_id, which is a fatal contract violation.
    # ------------------------------------------------------------------

    def has_tab(self, tab_id: str) -> bool:
        return tab_id in self._state.tabs

    def list_tab_ids(self) -> list[str]:
        return list(self._state.tabs.keys())

    def get_tab_adapter_name(self, tab_id: str) -> str:
        return self._state.get_tab(tab_id).adapter_name

    def get_tab_cfg_schema(self, tab_id: str) -> CfgSchema:
        return self._state.get_tab(tab_id).cfg_schema

    def get_tab_result(self, tab_id: str) -> Optional[object]:
        return self._tab_svc.get_tab_result(tab_id)

    def get_tab_snapshot(self, tab_id: str) -> TabViewSnapshot:
        return self._tab_view_svc.get_snapshot(tab_id)

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None:
        """Auto-commit boundary for tab CfgFormWidget.

        Writes the latest form draft into ``State.cfg_schema`` as the committed
        truth. Cfg edits do not change ``TabInteractionState`` (run / analyze /
        save availability), so no ``TAB_INTERACTION_CHANGED`` is emitted here;
        the form's own ``validity_changed`` signal drives any UI refresh.

        Do not call from dialog / writeback local LiveModel paths — those keep
        their drafts off of ``State`` until their own Apply boundary.

        Terminal: → ``TabService.update_tab_cfg`` → ``State.update_tab_cfg_schema``,
        which bumps ``tab:<id>:cfg`` and emits no event.
        """
        self._tab_svc.update_tab_cfg(tab_id, schema)

    def update_tab_analyze_param_instance(self, tab_id: str, instance: object) -> None:
        self._tab_svc.update_tab_analyze_param_instance(tab_id, instance)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def update_tab_save_paths(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        self._tab_svc.update_tab_save_path_overrides(tab_id, data_path, image_path)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def get_adapter_names(self) -> list[str]:
        return self._tab_svc.list_adapter_names()

    def get_adapter_cfg_spec(self, adapter_name: str):
        """Static cfg spec of an adapter (no tab/context needed)."""
        return self._tab_svc.adapter_cfg_spec(adapter_name)

    def get_adapter_analyze_params(self, adapter_name: str) -> list[dict]:
        """Static analyze-params field spec of an adapter ([] if unsupported)."""
        return self._tab_svc.adapter_analyze_params(adapter_name)

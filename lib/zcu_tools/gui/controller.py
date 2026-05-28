from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Protocol

from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

logger = logging.getLogger(__name__)

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .adapter import CfgSchema, SocCfgHandle, WritebackItem
from .event_bus import (
    EventBus,
    GuiEvent,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from .io_manager import IOManager
from .plot_host import FigureContainer
from .registry import Registry
from .runner import AnalyzeRunner, Runner, SaveDataRunner
from .services import (
    DEFAULT_LEFT_PANEL_WIDTH,
    AnalyzeService,
    ConnectDeviceRequest,
    ConnectionService,
    ContextService,
    DeviceService,
    DeviceSnapshot,
    DisconnectDeviceRequest,
    OperationGate,
    PersistedStartup,
    RestoreReport,
    RunService,
    SaveBothOutcome,
    SaveService,
    SessionPersistenceError,
    SessionPersistenceService,
    SetDeviceValueRequest,
    SetupDeviceRequest,
    StartupConnectionRequest,
    StartupPersistenceError,
    StartupPersistenceService,
    StartupProjectRequest,
    StartupService,
    TabService,
    TabViewService,
    TabViewSnapshot,
    WorkspaceService,
    WritebackService,
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
from .services.remote.dialogs import DialogName
from .state import State


class ViewProtocol(Protocol):
    """Minimal interface Controller uses to update the View."""

    def show_status_message(self, message: str) -> None: ...
    def make_pbar_factory(self, tab_id: str) -> Optional[object]: ...
    def make_live_container(self, tab_id: str) -> Optional[FigureContainer]: ...
    def show_error_dialog(self, title: str, message: str) -> None: ...

    # Dialog API — shared between UI clicks and RemoteControlService.
    def open_dialog(self, name: DialogName) -> None: ...
    def close_dialog(self, name: DialogName) -> None: ...
    def list_open_dialogs(self) -> list[DialogName]: ...
    def register_dialog(self, name: DialogName, dialog: Any) -> None: ...

    # Remote query helpers.
    def get_view_snapshot(self) -> dict[str, object]: ...
    def take_screenshot(self, tab_id: Optional[str] = None) -> bytes: ...
    def get_tab_live_model_root(self, tab_id: str) -> Any: ...


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
    ) -> None:
        self._state = state
        self._view = view
        self._bus = bus
        self._operation_gate = OperationGate()

        # Initialize domain services
        self._dev_svc = DeviceService(bus, self._operation_gate)
        self._conn_svc = ConnectionService(state, bus, self._operation_gate)
        self._ctx_svc = ContextService(state, io_manager, bus)
        self._tab_svc = TabService(state, registry)
        self._run_svc = RunService(state, runner, bus, self._operation_gate)
        self._analyze_svc = AnalyzeService(state, AnalyzeRunner(), bus)
        self._save_svc = SaveService(state, SaveDataRunner(), bus)
        self._writeback_svc = WritebackService(state, bus)
        self._tab_view_svc = TabViewService(
            state,
            self._tab_svc,
            self._writeback_svc,
            self._ctx_svc,
        )
        self._workspace_svc = WorkspaceService(
            state,
            self._tab_svc,
            SessionPersistenceService(),
            bus,
        )
        self._startup_svc = StartupService(
            self._ctx_svc,
            self._dev_svc,
            StartupPersistenceService(),
        )

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
        self._dev_svc.value_set.connect(self._on_device_value_set)
        self._dev_svc.operation_failed.connect(self._on_device_operation_failed)

    def set_view(self, view: ViewProtocol) -> None:
        self._view = view

    def _require_view(self) -> ViewProtocol:
        if self._view is None:
            raise RuntimeError("Controller view is not attached")
        return self._view

    def _report_persistence_error(self, title: str, error: Exception) -> None:
        self._require_view().show_error_dialog(title, str(error))

    def _on_run_finished(self, tab_id: str, _result: object) -> None:
        # State is already updated in RunService/Runner
        self._tab_svc.initialize_tab_analyze_params(tab_id)
        self._bus.emit(
            GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
        )

    def _on_run_failed(self, _tab_id: str, error: Exception) -> None:
        self._require_view().show_error_dialog("Run failed", str(error))

    def _on_analyze_finished(self, tab_id: str, _result: object) -> None:
        self._bus.emit(
            GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
        )

    def _on_analyze_failed(self, _tab_id: str, error: Exception) -> None:
        self._require_view().show_error_dialog("Analyze failed", str(error))

    def _on_save_finished(self, tab_id: str, data_path: str) -> None:
        del tab_id
        self._require_view().show_status_message(f"Data saved to {data_path}")

    def _on_save_failed(self, tab_id: str, data_path: str, error: Exception) -> None:
        del tab_id, data_path
        self._require_view().show_error_dialog("Save data failed", str(error))

    def _on_save_both_finished(self, tab_id: str, outcome: SaveBothOutcome) -> None:
        del tab_id
        if outcome.data_error is None and outcome.image_error is None:
            self._require_view().show_status_message(
                f"Data saved to {outcome.data_path}; "
                f"image saved to {outcome.image_path}"
            )
            return
        if outcome.data_error is None:
            self._require_view().show_status_message(
                f"Data saved to {outcome.data_path}; image failed: {outcome.image_error}"
            )
            return
        if outcome.image_error is None:
            self._require_view().show_status_message(
                f"Data failed: {outcome.data_error}; "
                f"image saved to {outcome.image_path}"
            )
            return
        self._require_view().show_error_dialog(
            "Save Both failed",
            f"Data failed: {outcome.data_error}\nImage failed: {outcome.image_error}",
        )

    def _on_device_setup_finished(self, name: str) -> None:
        self._require_view().show_status_message(f"Device setup completed: {name}")

    def _on_device_setup_failed(self, name: str, error: str) -> None:
        self._require_view().show_error_dialog(
            f"Device setup failed: {name}",
            error,
        )

    def _on_device_setup_cancelled(self, name: str) -> None:
        self._require_view().show_status_message(f"Device setup cancelled: {name}")

    def _on_device_connected(self, req: ConnectDeviceRequest) -> None:
        if req.remember:
            try:
                self._startup_svc.remember_device(req)
            except StartupPersistenceError as exc:
                self._report_persistence_error("Startup settings save failed", exc)
        self._require_view().show_status_message(f"Device connected: {req.name}")

    def _on_device_disconnected(self, req: DisconnectDeviceRequest) -> None:
        if not req.remember:
            try:
                self._startup_svc.forget_device(req.name)
            except StartupPersistenceError as exc:
                self._report_persistence_error("Startup settings save failed", exc)
        self._require_view().show_status_message(f"Device disconnected: {req.name}")

    def _on_device_value_set(self, name: str) -> None:
        self._require_view().show_status_message(f"Device value set: {name}")

    def _on_device_operation_failed(self, name: str, error: str) -> None:
        self._require_view().show_error_dialog(
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
            self._require_view().show_error_dialog(
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

    def _require_active_context(self, operation: str) -> None:
        if not self.has_active_context():
            raise RuntimeError(
                f"Cannot {operation} without an active file-backed context."
            )

    def get_running_tab_id(self) -> Optional[str]:
        return self._state.running_tab_id

    def has_soc(self) -> bool:
        return self._conn_svc.has_soc()

    def start_run(self, tab_id: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        self._require_active_context("run")

        # Read committed cfg from State, not from the live form. The tab's
        # CfgFormWidget auto-commits every change via update_tab_cfg, so the
        # State value is authoritative.
        schema = self._state.get_tab(tab_id).cfg_schema
        view = self._require_view()
        pbar_factory = view.make_pbar_factory(tab_id)
        live_container = view.make_live_container(tab_id)
        self._run_svc.start_run(tab_id, schema, pbar_factory, live_container)

    def cancel_run(self) -> None:
        self._run_svc.cancel_run()

    # ------------------------------------------------------------------
    # Analyze flow (TabService)
    # ------------------------------------------------------------------

    def analyze(self, tab_id: str, analyze_params_instance: object) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        figure_container = self._require_view().make_live_container(tab_id)
        self._analyze_svc.start_analyze(
            tab_id, analyze_params_instance, figure_container
        )

    # ------------------------------------------------------------------
    # Writeback (TabService)
    # ------------------------------------------------------------------

    def apply_writeback_items(
        self, tab_id: str, items: list[WritebackItem]
    ) -> list[str]:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        return self._writeback_svc.apply_tab_writeback_items(tab_id, items)

    # ------------------------------------------------------------------
    # Save (TabService)
    # ------------------------------------------------------------------

    def save_data(self, tab_id: str, data_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        self._require_active_context("save data")
        self._save_svc.start_save_data(tab_id, data_path)

    def save_image(self, tab_id: str, image_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        self._require_active_context("save image")
        self._save_svc.save_image_sync(tab_id, image_path)
        self._require_view().show_status_message(f"Image saved to {image_path}")

    def save_both(self, tab_id: str, data_path: str, image_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        self._require_active_context("save data and image")
        self._save_svc.start_save_both(tab_id, data_path, image_path)

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

    def set_ml_module(self, name: str, module: Any) -> None:
        self._ctx_svc.set_ml_module(name, module)

    def set_ml_module_from_raw(self, name: str, raw_dict: dict) -> None:
        self._ctx_svc.set_ml_module_from_raw(name, raw_dict)

    def del_ml_module(self, name: str) -> None:
        self._ctx_svc.del_ml_module(name)

    def set_ml_waveform(self, name: str, waveform: Any) -> None:
        self._ctx_svc.set_ml_waveform(name, waveform)

    def set_ml_waveform_from_raw(self, name: str, raw_dict: dict) -> None:
        self._ctx_svc.set_ml_waveform_from_raw(name, raw_dict)

    def coerce_md_value(self, key: str, text: str) -> Any:
        return self._ctx_svc.coerce_md_value(key, text)

    def del_ml_waveform(self, name: str) -> None:
        self._ctx_svc.del_ml_waveform(name)

    # ------------------------------------------------------------------
    # Device (DeviceService)
    # ------------------------------------------------------------------

    def start_connect_device(self, req: ConnectDeviceRequest) -> None:
        self._dev_svc.start_connect_device(req)

    def start_disconnect_device(self, req: DisconnectDeviceRequest) -> None:
        self._dev_svc.start_disconnect_device(req)

    def list_devices(self) -> list[DeviceEntry]:
        return self._dev_svc.list_devices()

    def list_device_names(self) -> list[str]:
        return self._dev_svc.list_device_names()

    def get_device_unit(self, name: str) -> str:
        return self._dev_svc.get_device_unit(name)

    def get_device_value_for_new_context(self, name: str) -> Optional[float]:
        return self._dev_svc.get_device_value_for_new_context(name)

    def start_set_device_value(self, req: SetDeviceValueRequest) -> None:
        self._dev_svc.start_set_device_value(req)

    def get_device_value(self, name: str) -> object:
        return self._dev_svc.get_device_value(name)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        return self._dev_svc.get_device_info(name)

    def start_setup_device(self, req: SetupDeviceRequest) -> None:
        self._dev_svc.start_setup_device(req)

    def get_active_device_setup(self) -> Optional[DeviceSetupSnapshot]:
        return self._dev_svc.get_active_setup()

    def cancel_device_operation(self, name: str) -> None:
        self._dev_svc.cancel_device_operation(name)

    def start_reconnect_device(self, name: str) -> None:
        self._dev_svc.start_reconnect_device(name)

    def forget_device(self, name: str) -> None:
        self._dev_svc.forget_device(name)
        try:
            self._startup_svc.forget_device(name)
        except StartupPersistenceError as exc:
            self._report_persistence_error("Startup settings save failed", exc)

    def is_memory_device(self, name: str) -> bool:
        return self._dev_svc.is_memory_device(name)

    def get_memory_device_address(self, name: str) -> Optional[str]:
        """Return the persisted address for a memory-only device, or None."""
        return self._dev_svc.get_memory_device_address(name)

    def get_device_snapshot(self, name: str) -> DeviceSnapshot | None:
        return self._dev_svc.get_device_snapshot(name)

    def get_active_device_operation(self) -> DeviceSnapshot | None:
        return self._dev_svc.get_active_device_operation()

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

    def start_connect(self, req: ConnectRequest) -> None:
        self._conn_svc.start_connect(req)

    def bind_connection_outcome(
        self,
        on_finished: Callable[[], None],
        on_failed: Callable[[str], None],
    ) -> None:
        """Bind one dialog observer without exposing the connection service."""
        try:
            self._conn_svc.connection_finished.disconnect(on_finished)
        except (TypeError, RuntimeError):
            pass
        try:
            self._conn_svc.connection_failed.disconnect(on_failed)
        except (TypeError, RuntimeError):
            pass
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

    # ------------------------------------------------------------------
    # Dialog / view-query facades (Phase 81a)
    # ------------------------------------------------------------------

    def open_dialog(self, name: DialogName) -> None:
        self._require_view().open_dialog(name)

    def close_dialog(self, name: DialogName) -> None:
        self._require_view().close_dialog(name)

    def list_open_dialogs(self) -> list[DialogName]:
        return list(self._require_view().list_open_dialogs())

    def get_view_snapshot(self) -> dict[str, object]:
        return self._require_view().get_view_snapshot()

    def take_screenshot(self, tab_id: Optional[str] = None) -> bytes:
        return self._require_view().take_screenshot(tab_id)

    def get_tab_live_model_root(self, tab_id: str):
        return self._require_view().get_tab_live_model_root(tab_id)

    def set_tab_field(self, tab_id: str, path: str, value: object) -> None:
        """Mutate a single cfg field on the tab's live LiveModel (Phase 81b).

        Goes through the form's live tree so the change auto-commits to
        ``State.cfg_schema`` via the existing ``schema_changed`` path,
        keeping the visible widget in sync (WYSIWYG).
        """
        from .services.remote.path_resolver import resolve_and_set

        root = self._require_view().get_tab_live_model_root(tab_id)
        resolve_and_set(root, path, value)

    def get_adapter_names(self) -> list[str]:
        return self._tab_svc.list_adapter_names()

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol

from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

logger = logging.getLogger(__name__)

from matplotlib.figure import Figure

from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .adapter import CfgSchema, SavePaths, SocCfgHandle, WritebackItem
from .event_bus import (
    EventBus,
    GuiEvent,
    TabAddedPayload,
    TabClosedPayload,
    TabContentChangedPayload,
    TabInteractionChangedPayload,
)
from .io_manager import IOManager
from .plot_host import FigureContainer
from .registry import Registry
from .runner import AnalyzeRunner, Runner, SaveDataRunner
from .services import (
    AnalyzeService,
    ConnectionService,
    ContextService,
    DeviceService,
    RunService,
    SaveBothOutcome,
    SaveService,
    TabService,
    WritebackService,
)
from .services.connection import (
    ConnectRequest,
    LoadPredictorRequest,
    PredictFreqRequest,
)
from .services.device import (
    DeviceSetupSnapshot,
    RegisterDeviceRequest,
)
from .state import State


class ViewProtocol(Protocol):
    """Minimal interface Controller uses to update the View."""

    def show_status_message(self, message: str) -> None: ...
    def make_pbar_factory(self, tab_id: str) -> Optional[object]: ...
    def make_live_container(self, tab_id: str) -> Optional[FigureContainer]: ...
    def show_error_dialog(self, title: str, message: str) -> None: ...


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

        # Initialize domain services
        self._dev_svc = DeviceService(state, bus)
        self._conn_svc = ConnectionService(state, bus)
        self._ctx_svc = ContextService(state, io_manager, bus)
        self._tab_svc = TabService(state, registry, bus)
        self._run_svc = RunService(state, runner, bus)
        self._analyze_svc = AnalyzeService(state, AnalyzeRunner(), bus)
        self._save_svc = SaveService(state, SaveDataRunner(), bus)
        self._writeback_svc = WritebackService(state, bus)

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

    def set_view(self, view: ViewProtocol) -> None:
        self._view = view

    def _require_view(self) -> ViewProtocol:
        if self._view is None:
            raise RuntimeError("Controller view is not attached")
        return self._view

    def _on_run_finished(self, tab_id: str, _result: object) -> None:
        # State is already updated in RunService/Runner
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

    def get_bus(self) -> EventBus:
        return self._bus

    # ------------------------------------------------------------------
    # ExpTab operations (TabService)
    # ------------------------------------------------------------------

    def new_tab(self, adapter_name: str) -> str:
        tab_id = self._tab_svc.new_tab(adapter_name)
        self._bus.emit(
            GuiEvent.TAB_ADDED,
            TabAddedPayload(tab_id=tab_id, adapter_name=adapter_name),
        )
        return tab_id

    def close_tab(self, tab_id: str) -> None:
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError("Cannot close a busy tab")
        self._tab_svc.close_tab(tab_id)
        self._bus.emit(GuiEvent.TAB_CLOSED, TabClosedPayload(tab_id=tab_id))

    # ------------------------------------------------------------------
    # Run flow (RunService & ContextService)
    # ------------------------------------------------------------------

    def has_project(self) -> bool:
        return self._ctx_svc.has_project()

    def has_context(self) -> bool:
        return self._ctx_svc.has_context()

    def has_startup_context(self) -> bool:
        return self._ctx_svc.has_startup_context()

    def is_run_active(self) -> bool:
        return self._state.is_run_active()

    def is_tab_running(self, tab_id: str) -> bool:
        return self._state.is_tab_running(tab_id)

    def is_tab_analyzing(self, tab_id: str) -> bool:
        return self._state.is_tab_analyzing(tab_id)

    def is_tab_saving_data(self, tab_id: str) -> bool:
        return self._state.is_tab_saving_data(tab_id)

    def is_tab_busy(self, tab_id: str) -> bool:
        return self._state.is_tab_busy(tab_id)

    def has_soc(self) -> bool:
        return self._conn_svc.has_soc()

    def start_run(self, tab_id: str, schema: CfgSchema) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        if not self.has_soc():
            raise RuntimeError("No ZCU connection. Please connect first.")

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
        self._save_svc.start_save_data(tab_id, data_path)

    def save_image(self, tab_id: str, image_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        self._save_svc.save_image_sync(tab_id, image_path)
        self._require_view().show_status_message(f"Image saved to {image_path}")

    def save_both(self, tab_id: str, data_path: str, image_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        self._save_svc.start_save_both(tab_id, data_path, image_path)

    # ------------------------------------------------------------------
    # Context / IO (ContextService)
    # ------------------------------------------------------------------

    def set_startup_context(
        self,
        md: MetaDict,
        ml: ModuleLibrary,
        chip_name: str = "unknown_chip",
        qub_name: str = "unknown_qubit",
        res_name: str = "unknown_resonator",
        result_dir: str = "",
        database_path: str = "",
    ) -> None:
        self._ctx_svc.set_startup_context(
            md, ml, chip_name, qub_name, res_name, result_dir, database_path
        )

    def setup_project(self, result_dir: str) -> None:
        self._ctx_svc.setup_project(result_dir)

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

    def register_device(self, req: RegisterDeviceRequest) -> None:
        self._dev_svc.register_device(req)

    def drop_device(self, name: str) -> None:
        self._dev_svc.drop_device(name)

    def list_devices(self) -> dict[str, str]:
        return self._dev_svc.list_devices()

    def list_device_names(self) -> list[str]:
        return self._dev_svc.list_device_names()

    def get_device_unit(self, name: str) -> str:
        return self._dev_svc.get_device_unit(name)

    def get_device_value_for_new_context(self, name: str) -> Optional[float]:
        return self._dev_svc.get_device_value_for_new_context(name)

    def set_device_value(self, name: str, value: Any) -> object:
        return self._dev_svc.set_device_value(name, value)

    def get_device_value(self, name: str) -> object:
        return self._dev_svc.get_device_value(name)

    def get_device_info(self, name: str) -> BaseDeviceInfo | None:
        return self._dev_svc.get_device_info(name)

    def setup_device(self, name: str, info: BaseDeviceInfo) -> None:
        self._dev_svc.setup_device(name, info)

    def get_active_device_setup(self) -> Optional[DeviceSetupSnapshot]:
        return self._dev_svc.get_active_setup()

    def cancel_device_setup(self) -> None:
        self._dev_svc.cancel_setup()

    # ------------------------------------------------------------------
    # Connection / Predictor (ConnectionService)
    # ------------------------------------------------------------------

    def start_connect(self, req: ConnectRequest) -> None:
        self._conn_svc.start_connect(req)

    def get_connection_service(self) -> ConnectionService:
        """Return the ConnectionService so views can subscribe to its Qt signals."""
        return self._conn_svc

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

    def get_tab_default_cfg(self, tab_id: str) -> CfgSchema:
        return self._tab_svc.get_tab_default_cfg(tab_id)

    def get_tab_fresh_cfg(self, tab_id: str) -> CfgSchema:
        return self._tab_svc.get_tab_fresh_cfg(tab_id)

    def get_tab_result(self, tab_id: str) -> Optional[object]:
        return self._tab_svc.get_tab_result(tab_id)

    def has_run_result(self, tab_id: str) -> bool:
        return self._tab_svc.has_run_result(tab_id)

    def has_analyze_result(self, tab_id: str) -> bool:
        return self._tab_svc.has_analyze_result(tab_id)

    def has_figure(self, tab_id: str) -> bool:
        return self._tab_svc.get_tab_figure(tab_id) is not None

    def get_tab_figure(self, tab_id: str) -> Optional[Figure]:
        return self._tab_svc.get_tab_figure(tab_id)

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]:
        return self._writeback_svc.get_tab_writeback_items(tab_id)

    def get_tab_analyze_params(self, tab_id: str) -> object:
        return self._tab_svc.get_tab_analyze_params(tab_id)

    def get_tab_analyze_param_instance(self, tab_id: str) -> object | None:
        return self._tab_svc.get_tab_analyze_param_instance(tab_id)

    def get_tab_save_paths(self, tab_id: str) -> Optional[SavePaths]:
        return self._tab_svc.get_tab_save_paths(tab_id)

    def update_tab_cfg(self, tab_id: str, schema: CfgSchema) -> None:
        self._tab_svc.update_tab_cfg(tab_id, schema)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )

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

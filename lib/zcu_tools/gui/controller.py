from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol

logger = logging.getLogger(__name__)

from matplotlib.figure import Figure

from zcu_tools.meta_tool import MetaDict, ModuleLibrary

from .adapter import CfgSchema, WritebackItem
from .device_manager import DeviceManager
from .event_bus import EventBus, GuiEvent
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
    SaveService,
    TabService,
    WritebackService,
)
from .state import State

if TYPE_CHECKING:
    pass


@dataclass
class _PendingSaveBoth:
    data_path: str
    image_path: str
    image_error: Optional[str] = None


class ViewProtocol(Protocol):
    """Minimal interface Controller uses to update the View."""

    def show_status_message(self, message: str) -> None: ...
    def make_pbar_factory(self, tab_id: str) -> Optional[object]: ...
    def make_live_container(self, tab_id: str) -> Optional[FigureContainer]: ...


class Controller:
    """Façade for the GUI application. Delegates to domain services."""

    def __init__(
        self,
        state: State,
        runner: Runner,
        registry: Registry,
        io_manager: IOManager,
        device_manager: DeviceManager,
        view: Optional[ViewProtocol],
        bus: EventBus,
    ) -> None:
        self._state = state
        self._view = view
        self._bus = bus

        # Initialize domain services
        self._dev_svc = DeviceService(state, device_manager)
        self._conn_svc = ConnectionService(state, bus)
        self._ctx_svc = ContextService(state, io_manager, bus)
        self._tab_svc = TabService(state, registry, bus)
        self._run_svc = RunService(state, runner, bus)
        self._analyze_svc = AnalyzeService(state, AnalyzeRunner(), bus)
        self._save_svc = SaveService(state, SaveDataRunner(), bus)
        self._writeback_svc = WritebackService(state, bus)
        self._pending_save_both: dict[str, _PendingSaveBoth] = {}

        self._run_svc.run_finished.connect(self._on_run_finished)
        self._run_svc.run_failed.connect(self._on_run_failed)
        self._analyze_svc.analyze_finished.connect(self._on_analyze_finished)
        self._analyze_svc.analyze_failed.connect(self._on_analyze_failed)
        self._save_svc.save_finished.connect(self._on_save_finished)
        self._save_svc.save_failed.connect(self._on_save_failed)

    def set_view(self, view: ViewProtocol) -> None:
        self._view = view

    def _require_view(self) -> ViewProtocol:
        if self._view is None:
            raise RuntimeError("Controller view is not attached")
        return self._view

    def _on_run_finished(self, tab_id: str, _result: object) -> None:
        # State is already updated in RunService/Runner
        self._bus.emit(GuiEvent.TAB_CONTENT_CHANGED, tab_id)

    def _on_run_failed(self, _tab_id: str, error: Exception) -> None:
        self._require_view().show_status_message(f"Run failed: {error}")

    def _on_analyze_finished(self, tab_id: str, _result: object) -> None:
        self._bus.emit(GuiEvent.TAB_CONTENT_CHANGED, tab_id)

    def _on_analyze_failed(self, _tab_id: str, error: Exception) -> None:
        self._require_view().show_status_message(f"Analyze failed: {error}")

    def _on_save_finished(self, tab_id: str, data_path: str) -> None:
        pending = self._pending_save_both.pop(tab_id, None)
        if pending is None:
            self._require_view().show_status_message(f"Data saved to {data_path}")
            return
        if pending.image_error is None:
            self._require_view().show_status_message(
                f"Data saved to {data_path}; image saved to {pending.image_path}"
            )
        else:
            self._require_view().show_status_message(
                f"Data saved to {data_path}; image failed: {pending.image_error}"
            )

    def _on_save_failed(self, tab_id: str, data_path: str, error: Exception) -> None:
        pending = self._pending_save_both.pop(tab_id, None)
        if pending is None:
            self._require_view().show_status_message(f"Save data failed: {error}")
            return
        if pending.image_error is None:
            self._require_view().show_status_message(
                f"Data failed: {error}; image saved to {pending.image_path}"
            )
        else:
            self._require_view().show_status_message(
                f"Data failed: {error}; image failed: {pending.image_error}"
            )

    def get_bus(self) -> EventBus:
        return self._bus

    # ------------------------------------------------------------------
    # ExpTab operations (TabService)
    # ------------------------------------------------------------------

    def new_tab(self, adapter_name: str) -> str:
        tab_id = self._tab_svc.new_tab(adapter_name)
        self._bus.emit(GuiEvent.TAB_ADDED, tab_id, adapter_name)
        return tab_id

    def close_tab(self, tab_id: str) -> None:
        if self.is_running() and self._state.active_tab_id == tab_id:
            self.cancel_run()
        self._tab_svc.close_tab(tab_id)
        self._bus.emit(GuiEvent.TAB_CLOSED, tab_id)

    # ------------------------------------------------------------------
    # Run flow (RunService & ContextService)
    # ------------------------------------------------------------------

    def has_project(self) -> bool:
        return self._ctx_svc.has_project()

    def has_context(self) -> bool:
        return self._ctx_svc.has_context()

    def has_startup_context(self) -> bool:
        return self._ctx_svc.has_startup_context()

    def is_running(self) -> bool:
        return self._state.has_active_long_task

    def is_run_active(self) -> bool:
        return self._state.is_running

    def is_analyzing(self) -> bool:
        return self._state.is_analyzing

    def is_saving_data(self) -> bool:
        return self._state.is_saving_data

    def has_soc(self) -> bool:
        return self._conn_svc.has_soc()

    def start_run(
        self, tab_id: str, schema: CfgSchema, user_params: dict[str, object]
    ) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        if not self.has_soc():
            raise RuntimeError("No ZCU connection. Please connect first.")

        view = self._require_view()
        pbar_factory = view.make_pbar_factory(tab_id)
        live_container = view.make_live_container(tab_id)
        self._run_svc.start_run(
            tab_id, schema, user_params, pbar_factory, live_container
        )

    def cancel_run(self) -> None:
        self._run_svc.cancel_run()

    # ------------------------------------------------------------------
    # Analyze flow (TabService)
    # ------------------------------------------------------------------

    def analyze(self, tab_id: str, analyze_params: dict[str, object]) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        try:
            figure_container = self._require_view().make_live_container(tab_id)
            self._analyze_svc.start_analyze(tab_id, analyze_params, figure_container)
        except Exception as exc:
            logger.warning("analyze: failed tab_id=%r exc=%r", tab_id, exc)
            self._require_view().show_status_message(f"Analyze failed: {exc}")

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
        try:
            return self._writeback_svc.apply_tab_writeback_items(tab_id, items)
        except Exception as exc:
            logger.warning("apply_writeback: failed tab_id=%r exc=%r", tab_id, exc)
            self._require_view().show_status_message(f"Writeback failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Save (TabService)
    # ------------------------------------------------------------------

    def save_data(self, tab_id: str, data_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        try:
            self._save_svc.start_save_data(tab_id, data_path)
        except Exception as exc:
            logger.warning("save_data: failed tab_id=%r exc=%r", tab_id, exc)
            self._require_view().show_status_message(f"Save data failed: {exc}")

    def save_image(self, tab_id: str, image_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        try:
            self._tab_svc.save_image(tab_id, image_path)
            self._require_view().show_status_message(f"Image saved to {image_path}")
        except Exception as exc:
            logger.warning("save_image: failed tab_id=%r exc=%r", tab_id, exc)
            self._require_view().show_status_message(f"Save image failed: {exc}")

    def save_both(self, tab_id: str, data_path: str, image_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        pending = _PendingSaveBoth(data_path=data_path, image_path=image_path)
        data_started = False
        try:
            self._save_svc.start_save_data(tab_id, data_path)
            self._pending_save_both[tab_id] = pending
            data_started = True
        except Exception as exc:
            pending.image_error = None
            logger.warning(
                "save_both: save_data start failed tab_id=%r exc=%r", tab_id, exc
            )
            try:
                self._tab_svc.save_image(tab_id, image_path)
                self._require_view().show_status_message(
                    f"Data failed: {exc}; image saved to {image_path}"
                )
            except Exception as image_exc:
                self._require_view().show_status_message(
                    f"Data failed: {exc}; image failed: {image_exc}"
                )
            return

        try:
            self._tab_svc.save_image(tab_id, image_path)
        except Exception as exc:
            pending.image_error = str(exc)
            logger.warning("save_both: save_image failed tab_id=%r exc=%r", tab_id, exc)
            if not data_started:
                self._require_view().show_status_message(f"Save image failed: {exc}")

    # ------------------------------------------------------------------
    # Context / IO (ContextService)
    # ------------------------------------------------------------------

    def set_startup_context(
        self,
        md: Any,
        ml: Any,
        chip_name: str = "unknown_chip",
        qub_name: str = "unknown_qubit",
        res_name: str = "unknown_resonator",
        result_dir: str = "",
        database_path: str = "",
    ) -> None:
        self._ctx_svc.set_startup_context(
            md, ml, chip_name, qub_name, res_name, result_dir, database_path
        )
        self._bus.emit(GuiEvent.INSPECT_CHANGED, md)

    def setup_project(self, result_dir: str) -> None:
        self._ctx_svc.setup_project(result_dir)
        ctx = self._state.exp_context
        self._bus.emit(GuiEvent.CONTEXT_CHANGED, ctx.md, ctx.ml)

    def use_context(self, label: str) -> None:
        self._ctx_svc.use_context(label)
        self._bus.emit(GuiEvent.INSPECT_CHANGED, self.get_current_md())

    def new_context(
        self,
        value: Optional[float] = None,
        unit: str = "A",
        clone_from_current: bool = False,
    ) -> None:
        self._ctx_svc.new_context(value, unit, clone_from_current)
        self._bus.emit(GuiEvent.INSPECT_CHANGED, self.get_current_md())

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

    # ------------------------------------------------------------------
    # Device (DeviceService)
    # ------------------------------------------------------------------

    def register_device(self, name: str, device: Any) -> None:
        self._dev_svc.register_device(name, device)

    def drop_device(self, name: str) -> None:
        self._dev_svc.drop_device(name)

    def list_devices(self) -> dict[str, str]:
        return self._dev_svc.list_devices()

    def set_device_value(self, name: str, value: Any) -> Any:
        return self._dev_svc.set_device_value(name, value)

    def get_device_value(self, name: str) -> Any:
        return self._dev_svc.get_device_value(name)

    def get_device_info(self, name: str) -> Any:
        return self._dev_svc.get_device_info(name)

    def setup_device(
        self, name: str, info: Any, pbar_factory: Optional[Any] = None
    ) -> Any:
        return self._dev_svc.setup_device(name, info, pbar_factory)

    # ------------------------------------------------------------------
    # Connection / Predictor (ConnectionService)
    # ------------------------------------------------------------------

    def set_connection(self, soc: Any, soccfg: Any) -> None:
        self._conn_svc.set_connection(soc, soccfg)

    def set_predictor(
        self, predictor: Optional[Any], path: Optional[str] = None
    ) -> None:
        self._conn_svc.set_predictor(predictor, path)
        self._bus.emit(GuiEvent.PREDICTOR_CHANGED)

    def get_soccfg(self) -> Any:
        return self._conn_svc.get_soccfg()

    def get_predictor(self) -> Optional[Any]:
        return self._conn_svc.get_predictor()

    def get_predictor_info(self) -> Optional[dict]:
        return self._conn_svc.get_predictor_info()

    # ------------------------------------------------------------------
    # View query interface (TabService) — Tolerant APIs
    # ------------------------------------------------------------------

    def has_tab(self, tab_id: str) -> bool:
        return tab_id in self._state.tabs

    def get_tab_default_cfg(self, tab_id: str) -> Optional[CfgSchema]:
        if not self.has_tab(tab_id):
            logger.debug("get_tab_default_cfg: tab_id %r not found", tab_id)
            return None
        return self._tab_svc.get_tab_default_cfg(tab_id)

    def get_tab_fresh_cfg(self, tab_id: str) -> Optional[CfgSchema]:
        if not self.has_tab(tab_id):
            logger.debug("get_tab_fresh_cfg: tab_id %r not found", tab_id)
            return None
        return self._tab_svc.get_tab_fresh_cfg(tab_id)

    def get_tab_result(self, tab_id: str) -> Optional[object]:
        if not self.has_tab(tab_id):
            logger.debug("get_tab_result: tab_id %r not found", tab_id)
            return None
        return self._tab_svc.get_tab_result(tab_id)

    def has_run_result(self, tab_id: str) -> bool:
        if not self.has_tab(tab_id):
            return False
        return self._tab_svc.has_run_result(tab_id)

    def has_analyze_result(self, tab_id: str) -> bool:
        if not self.has_tab(tab_id):
            return False
        return self._tab_svc.has_analyze_result(tab_id)

    def get_tab_figure(self, tab_id: str) -> Optional[Figure]:
        if not self.has_tab(tab_id):
            logger.debug("get_tab_figure: tab_id %r not found", tab_id)
            return None
        return self._tab_svc.get_tab_figure(tab_id)

    def get_tab_writeback_items(self, tab_id: str) -> list:
        if not self.has_tab(tab_id):
            logger.debug("get_tab_writeback_items: tab_id %r not found", tab_id)
            return []
        return self._writeback_svc.get_tab_writeback_items(tab_id)

    def get_tab_analyze_params(self, tab_id: str) -> list:
        if not self.has_tab(tab_id):
            logger.debug("get_tab_analyze_params: tab_id %r not found", tab_id)
            return []
        return self._tab_svc.get_tab_analyze_params(tab_id)

    def get_tab_save_paths(self, tab_id: str) -> Any:
        if not self.has_tab(tab_id):
            logger.debug("get_tab_save_paths: tab_id %r not found", tab_id)
            return None
        return self._tab_svc.get_tab_save_paths(tab_id)

    def get_adapter_names(self) -> list[str]:
        # Simple passthrough to registry
        return self._tab_svc._registry.list_names()

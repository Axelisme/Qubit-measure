from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import TYPE_CHECKING, Any, Optional, Protocol

logger = logging.getLogger(__name__)

from .adapter import ExpContext
from .device_manager import DeviceManager
from .io_manager import IOManager
from .registry import Registry
from .runner import Runner
from .state import State

if TYPE_CHECKING:
    pass


class ViewProtocol(Protocol):
    """Minimal interface Controller uses to update the View."""

    def refresh_tab(self, tab_id: str) -> None: ...
    def refresh_run_state(self, is_running: bool) -> None: ...
    def refresh_context_panel(self) -> None: ...
    def refresh_config_panels(self) -> None: ...
    def show_status_message(self, message: str) -> None: ...


class Controller:
    def __init__(
        self,
        state: State,
        runner: Runner,
        registry: Registry,
        io_manager: IOManager,
        device_manager: DeviceManager,
        view: ViewProtocol,
    ) -> None:
        self._state = state
        self._runner = runner
        self._registry = registry
        self._io = io_manager
        self._dm = device_manager
        self._view = view

        runner.run_finished.connect(self._on_run_finished)
        runner.run_failed.connect(self._on_run_failed)

    # ------------------------------------------------------------------
    # ExpTab operations
    # ------------------------------------------------------------------

    def new_tab(self, adapter_name: str) -> str:
        adapter = self._registry.create(adapter_name)
        tab_id = str(uuid.uuid4())
        logger.info("new_tab: adapter=%r tab_id=%r", adapter_name, tab_id)
        self._state.add_tab(tab_id, adapter)
        return tab_id

    def close_tab(self, tab_id: str) -> None:
        logger.info("close_tab: tab_id=%r", tab_id)
        if self._runner.is_running and self._state.active_tab_id == tab_id:
            self._runner.cancel()
        self._state.remove_tab(tab_id)

    # ------------------------------------------------------------------
    # Run flow
    # ------------------------------------------------------------------

    def start_run(self, tab_id: str, schema: Any, user_params: dict) -> None:
        if self._state.is_running:
            raise RuntimeError("Another run is already active")
        logger.info("start_run: tab_id=%r user_params=%r", tab_id, list(user_params))
        tab = self._state.get_tab(tab_id)
        self._state.set_running(True)
        self._view.refresh_run_state(True)
        self._runner.start_run(
            tab_id,
            tab.adapter,
            self._state.exp_context,
            schema,
            user_params,
        )

    def cancel_run(self) -> None:
        logger.info("cancel_run")
        self._runner.cancel()

    def _on_run_finished(self, tab_id: str, result: Any) -> None:
        logger.info(
            "_on_run_finished: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        self._state.update_tab_result(tab_id, result, cfg=None)
        self._state.set_running(False)
        self._view.refresh_run_state(False)
        self._view.refresh_tab(tab_id)

    def _on_run_failed(self, tab_id: str, error: Exception) -> None:  # noqa: ARG002
        logger.warning("_on_run_failed: tab_id=%r error=%r", tab_id, error)
        self._state.set_running(False)
        self._view.refresh_run_state(False)
        self._view.show_status_message(f"Run failed: {error}")

    # ------------------------------------------------------------------
    # Analyze flow  (Phase 9)
    # ------------------------------------------------------------------

    def analyze(self, tab_id: str, user_params: dict) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Writeback  (Phase 9)
    # ------------------------------------------------------------------

    def apply_writeback(self, tab_id: str, selected_keys: list[str]) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Save  (Phase 9)
    # ------------------------------------------------------------------

    def save_data(self, tab_id: str, data_path: str) -> None:
        raise NotImplementedError

    def save_image(self, tab_id: str, image_path: str) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Context / IO  (Phase 10)
    # ------------------------------------------------------------------

    def use_context(self, label: str) -> None:
        raise NotImplementedError

    def new_context(
        self,
        value: Optional[float] = None,
        unit: str = "A",
        clone_from_current: bool = False,
    ) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Device  (Phase 10)
    # ------------------------------------------------------------------

    def register_device(self, name: str, device: Any) -> None:
        if self._state.is_running:
            raise RuntimeError("Cannot register device while a run is active")
        self._dm.register_device(name, device)

    def set_device_value(self, name: str, value: Any) -> None:
        if self._state.is_running:
            raise RuntimeError("Cannot set device value while a run is active")
        self._dm.set_device_value(name, value)

    def get_device_value(self, name: str) -> Any:
        return self._dm.get_device_value(name)

    # ------------------------------------------------------------------
    # Connection / Predictor  (Phase 10)
    # ------------------------------------------------------------------

    def set_connection(self, soc: Any, soccfg: Any) -> None:
        new_ctx = dataclasses.replace(self._state.exp_context, soc=soc, soccfg=soccfg)
        self._state.set_context(new_ctx)

    def set_predictor(self, predictor: Optional[Any]) -> None:
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=predictor)
        self._state.set_context(new_ctx)

    # ------------------------------------------------------------------
    # View query interface (pull model)
    # ------------------------------------------------------------------

    def get_tab_result(self, tab_id: str) -> Any:
        return self._state.get_tab(tab_id).last_result

    def get_tab_figure(self, tab_id: str) -> Any:
        return self._state.get_tab(tab_id).last_figure

    def get_tab_writeback_spec(self, tab_id: str) -> list:
        raise NotImplementedError

    def get_context_labels(self) -> list[str]:
        return self._io.list_contexts()

    def get_adapter_names(self) -> list[str]:
        return self._registry.list_names()

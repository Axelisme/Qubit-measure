from __future__ import annotations

import dataclasses
import logging
import uuid
from typing import TYPE_CHECKING, Any, Optional, Protocol

logger = logging.getLogger(__name__)

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
    def refresh_predictor_panel(self) -> None: ...
    def show_status_message(self, message: str) -> None: ...
    def make_pbar_factory(self, tab_id: str) -> Any: ...
    def make_live_container(self, tab_id: str) -> Any: ...


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
        self._predictor_path: Optional[str] = None

        runner.run_finished.connect(self._on_run_finished)
        runner.run_failed.connect(self._on_run_failed)

    # ------------------------------------------------------------------
    # ExpTab operations
    # ------------------------------------------------------------------

    def new_tab(self, adapter_name: str) -> str:
        adapter = self._registry.create(adapter_name)
        tab_id = str(uuid.uuid4())
        logger.info("new_tab: adapter=%r tab_id=%r", adapter_name, tab_id)
        self._state.add_tab(tab_id, adapter, self._state.exp_context)
        return tab_id

    def close_tab(self, tab_id: str) -> None:
        logger.info("close_tab: tab_id=%r", tab_id)
        if self._runner.is_running and self._state.active_tab_id == tab_id:
            self._runner.cancel()
        self._state.remove_tab(tab_id)

    # ------------------------------------------------------------------
    # Run flow
    # ------------------------------------------------------------------

    def has_project(self) -> bool:
        return self._io.has_project

    def has_context(self) -> bool:
        """True when any valid context exists (startup empty ctx or file-backed flux ctx)."""
        return self._io.has_context or self._state.has_startup_context

    def has_soc(self) -> bool:
        return self._state.exp_context.soc is not None

    def start_run(self, tab_id: str, schema: Any, user_params: dict) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        if not self.has_soc():
            raise RuntimeError("No ZCU connection. Please connect first.")
        if self._state.is_running:
            raise RuntimeError("Another run is already active")
        logger.info("start_run: tab_id=%r user_params=%r", tab_id, list(user_params))
        tab = self._state.get_tab(tab_id)
        self._state.set_running(True)
        self._view.refresh_run_state(True)
        pbar_factory = self._view.make_pbar_factory(tab_id)
        live_container = self._view.make_live_container(tab_id)
        if live_container is not None:
            from zcu_tools.liveplot.backend.qt import register_pending_container

            register_pending_container(live_container)
        self._runner.start_run(
            tab_id,
            tab.adapter,
            self._state.exp_context,
            schema,
            user_params,
            pbar_factory=pbar_factory,
        )

    def cancel_run(self) -> None:
        logger.info("cancel_run")
        self._runner.cancel()

    def _on_run_finished(self, tab_id: str, result: Any) -> None:
        logger.info(
            "_on_run_finished: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        self._clear_live_container()
        self._state.update_tab_result(tab_id, result, cfg=None)
        self._state.set_running(False)
        self._view.refresh_run_state(False)
        self._view.refresh_tab(tab_id)

    def _on_run_failed(self, tab_id: str, error: Exception) -> None:  # noqa: ARG002
        logger.warning("_on_run_failed: tab_id=%r error=%r", tab_id, error)
        self._clear_live_container()
        self._state.set_running(False)
        self._view.refresh_run_state(False)
        self._view.show_status_message(f"Run failed: {error}")

    def _clear_live_container(self) -> None:
        from zcu_tools.liveplot.backend.qt import clear_pending_container

        clear_pending_container()

    # ------------------------------------------------------------------
    # Analyze flow  (Phase 9)
    # ------------------------------------------------------------------

    def analyze(self, tab_id: str, user_params: dict) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        tab = self._state.get_tab(tab_id)
        if tab.last_result is None:
            raise RuntimeError("No run result available to analyze")
        logger.info("analyze: tab_id=%r user_params=%r", tab_id, list(user_params))
        try:
            ctx = self._state.exp_context
            analyze_result = tab.adapter.analyze(tab.last_result, ctx, **user_params)
            figure = tab.adapter.get_figure(analyze_result)
            self._state.update_tab_analyze(tab_id, analyze_result, figure)
            self._view.refresh_tab(tab_id)
        except Exception as exc:
            logger.warning("analyze: failed tab_id=%r exc=%r", tab_id, exc)
            self._view.show_status_message(f"Analyze failed: {exc}")

    # ------------------------------------------------------------------
    # Writeback  (Phase 9)
    # ------------------------------------------------------------------

    def apply_writeback(self, tab_id: str, selected_keys: list[str]) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        tab = self._state.get_tab(tab_id)
        if tab.last_analyze_result is None:
            raise RuntimeError("No analyze result available for writeback")
        logger.info("apply_writeback: tab_id=%r keys=%r", tab_id, selected_keys)
        try:
            ctx = self._state.exp_context
            tab.adapter.apply_writeback(ctx, tab.last_analyze_result, selected_keys)
            self._view.refresh_config_panels()
        except Exception as exc:
            logger.warning("apply_writeback: failed tab_id=%r exc=%r", tab_id, exc)
            self._view.show_status_message(f"Writeback failed: {exc}")

    # ------------------------------------------------------------------
    # Save  (Phase 9)
    # ------------------------------------------------------------------

    def save_data(self, tab_id: str, data_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        tab = self._state.get_tab(tab_id)
        if tab.last_result is None:
            raise RuntimeError("No run result available to save")
        logger.info("save_data: tab_id=%r path=%r", tab_id, data_path)
        try:
            ctx = self._state.exp_context
            tab.adapter.save(data_path, tab.last_result, ctx)
            self._view.show_status_message(f"Data saved to {data_path}")
        except Exception as exc:
            logger.warning("save_data: failed tab_id=%r exc=%r", tab_id, exc)
            self._view.show_status_message(f"Save data failed: {exc}")

    def save_image(self, tab_id: str, image_path: str) -> None:
        if not self.has_context():
            raise RuntimeError(
                "No experiment context. Use Project… to set up chip/qubit or load a project."
            )
        tab = self._state.get_tab(tab_id)
        if tab.last_figure is None:
            raise RuntimeError("No figure available to save")
        logger.info("save_image: tab_id=%r path=%r", tab_id, image_path)
        try:
            tab.last_figure.savefig(image_path)
            self._view.show_status_message(f"Image saved to {image_path}")
        except Exception as exc:
            logger.warning("save_image: failed tab_id=%r exc=%r", tab_id, exc)
            self._view.show_status_message(f"Save image failed: {exc}")

    # ------------------------------------------------------------------
    # Context / IO  (Phase 11)
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
        """Set an in-memory (no file sync) startup context from the startup dialog."""
        logger.info(
            "set_startup_context: chip=%r qub=%r res=%r result_dir=%r db=%r",
            chip_name,
            qub_name,
            res_name,
            result_dir,
            database_path,
        )
        new_ctx = dataclasses.replace(
            self._state.exp_context,
            md=md,
            ml=ml,
            chip_name=chip_name,
            qub_name=qub_name,
            res_name=res_name,
            result_dir=result_dir,
            database_path=database_path,
        )
        self._state.set_context(new_ctx)
        self._state.has_startup_context = True
        self._view.refresh_context_panel()
        self._view.refresh_run_state(self._state.is_running)
        self._view.refresh_config_panels()

    def setup_project(self, result_dir: str) -> None:
        logger.info("setup_project: result_dir=%r", result_dir)
        self._io.setup(result_dir)
        self._view.refresh_context_panel()
        self._view.refresh_run_state(self._state.is_running)

    def use_context(self, label: str) -> None:
        logger.info("use_context: label=%r", label)
        new_ctx = self._io.use_context(label, self._state.exp_context)
        new_ctx = dataclasses.replace(new_ctx, active_label=label)
        self._state.set_context(new_ctx)
        self._view.refresh_context_panel()
        self._view.refresh_config_panels()

    def new_context(
        self,
        value: Optional[float] = None,
        unit: str = "A",
        clone_from_current: bool = False,
    ) -> None:
        logger.info(
            "new_context: value=%r unit=%r clone=%r", value, unit, clone_from_current
        )
        new_ctx = self._io.new_context(
            self._state.exp_context,
            value=value,
            unit=unit,
            clone_from_current=clone_from_current,
        )
        label = self._io.get_active_label() or ""
        new_ctx = dataclasses.replace(new_ctx, active_label=label)
        self._state.set_context(new_ctx)
        self._view.refresh_context_panel()
        self._view.refresh_config_panels()

    def get_active_context_label(self) -> Optional[str]:
        return self._io.get_active_label()

    def get_flux_dir(self) -> Optional[str]:
        """Return result_dir/exps/{label}/image path for the active context, or None."""
        import os

        ctx = self._state.exp_context
        label = self._io.get_active_label()
        if ctx.result_dir and label:
            return os.path.join(ctx.result_dir, "exps", label)
        return None

    # ------------------------------------------------------------------
    # Device  (Phase 11)
    # ------------------------------------------------------------------

    def register_device(self, name: str, device: Any) -> None:
        if self._state.is_running:
            raise RuntimeError("Cannot register device while a run is active")
        self._dm.register_device(name, device)

    def drop_device(self, name: str) -> None:
        if self._state.is_running:
            raise RuntimeError("Cannot drop device while a run is active")
        self._dm.drop_device(name)

    def list_devices(self) -> dict[str, str]:
        return self._dm.list_devices()

    def set_device_value(self, name: str, value: Any) -> Any:
        if self._state.is_running:
            raise RuntimeError("Cannot set device value while a run is active")
        return self._dm.set_device_value(name, value)

    def get_device_value(self, name: str) -> Any:
        return self._dm.get_device_value(name)

    # ------------------------------------------------------------------
    # Connection / Predictor  (Phase 10)
    # ------------------------------------------------------------------

    def set_connection(self, soc: Any, soccfg: Any) -> None:
        new_ctx = dataclasses.replace(self._state.exp_context, soc=soc, soccfg=soccfg)
        self._state.set_context(new_ctx)
        self._view.refresh_run_state(self._state.is_running)

    def set_predictor(
        self, predictor: Optional[Any], path: Optional[str] = None
    ) -> None:
        self._predictor_path = path
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=predictor)
        self._state.set_context(new_ctx)
        self._view.refresh_predictor_panel()

    def get_predictor_info(self) -> Optional[dict]:
        predictor = self._state.exp_context.predictor
        if predictor is None:
            return None
        return {"path": self._predictor_path, "flux_bias": predictor.flux_bias}

    # ------------------------------------------------------------------
    # View query interface (pull model)
    # ------------------------------------------------------------------

    def get_tab_default_cfg(self, tab_id: str) -> Any:
        return self._state.get_tab(tab_id).last_cfg

    def get_tab_result(self, tab_id: str) -> Any:
        return self._state.get_tab(tab_id).last_result

    def get_tab_figure(self, tab_id: str) -> Any:
        return self._state.get_tab(tab_id).last_figure

    def get_tab_writeback_spec(self, tab_id: str) -> list:
        tab = self._state.get_tab(tab_id)
        if tab.last_analyze_result is None:
            return []
        ctx = self._state.exp_context
        return tab.adapter.get_writeback_spec(tab.last_analyze_result, ctx)

    def get_tab_analyze_params(self, tab_id: str) -> dict:
        tab = self._state.get_tab(tab_id)
        return tab.adapter.get_analyze_params()

    def get_tab_save_paths(self, tab_id: str) -> Any:
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        return tab.adapter.make_save_paths(ctx)

    def get_context_labels(self) -> list[str]:
        return self._io.list_contexts()

    def get_adapter_names(self) -> list[str]:
        return self._registry.list_names()

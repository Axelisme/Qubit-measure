from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.adapter import RunRequest
from zcu_tools.gui.event_bus import GuiEvent
from zcu_tools.gui.plot_host import FigureContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.runner import Runner
    from zcu_tools.gui.state import State


class RunService(QObject):
    """Encapsulates execution of an experiment adapter via a Runner."""

    run_finished: Signal = Signal(str, object)
    run_failed: Signal = Signal(str, object)

    def __init__(self, state: "State", runner: "Runner", bus: "EventBus") -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._bus = bus

        self._runner.run_finished.connect(self._on_run_finished)
        self._runner.run_failed.connect(self._on_run_failed)

    def start_run(
        self,
        tab_id: str,
        schema: Any,
        user_params: dict,
        pbar_factory: Optional[Any] = None,
        live_container: Optional[FigureContainer] = None,
    ) -> None:
        if self._state.is_run_active():
            raise RuntimeError("Another run is already active")
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        ctx = self._state.exp_context
        req = RunRequest(
            md=ctx.md,
            ml=ctx.ml,
            soc=ctx.soc,
            soccfg=ctx.soccfg,
        )

        # Validate schema before starting the worker
        schema.to_raw_dict(req)
        logger.info("start_run: tab_id=%r user_params=%r", tab_id, list(user_params))

        tab = self._state.get_tab(tab_id)

        try:
            self._runner.start_run(
                tab_id,
                tab.adapter,
                req,
                schema,
                user_params,
                pbar_factory=pbar_factory,
                figure_container=live_container,
            )
        except Exception:
            raise
        self._state.set_tab_running(tab_id, True)
        self._bus.emit(GuiEvent.TAB_INTERACTION_CHANGED, tab_id)
        self._bus.emit(GuiEvent.RUN_LOCK_CHANGED, tab_id)

    def cancel_run(self) -> None:
        logger.info("cancel_run")
        self._runner.cancel()

    def _on_run_finished(self, tab_id: str, result: Any) -> None:
        logger.info(
            "_on_run_finished: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        self._state.update_tab_result(tab_id, result)
        self._state.set_tab_running(tab_id, False)
        self._bus.emit(GuiEvent.TAB_INTERACTION_CHANGED, tab_id)
        self._bus.emit(GuiEvent.RUN_LOCK_CHANGED, None)
        self.run_finished.emit(tab_id, result)

    def _on_run_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_run_failed: tab_id=%r error=%r", tab_id, error)
        self._state.set_tab_running(tab_id, False)
        self._bus.emit(GuiEvent.TAB_INTERACTION_CHANGED, tab_id)
        self._bus.emit(GuiEvent.RUN_LOCK_CHANGED, None)
        self.run_failed.emit(tab_id, error)

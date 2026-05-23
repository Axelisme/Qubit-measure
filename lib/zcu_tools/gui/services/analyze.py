from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.adapter import AnalyzeRequest
from zcu_tools.gui.event_bus import GuiEvent
from zcu_tools.gui.runner import AnalyzeRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State


class AnalyzeService(QObject):
    analyze_finished: Signal = Signal(str, object)
    analyze_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: "State",
        runner: AnalyzeRunner,
        bus: "EventBus",
    ) -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._bus = bus

        self._runner.analyze_finished.connect(self._on_analyze_finished)
        self._runner.analyze_failed.connect(self._on_analyze_failed)

    def start_analyze(self, tab_id: str, analyze_params: dict[str, object]) -> None:
        if self._state.has_active_long_task:
            raise RuntimeError("Another long-running task is already active")

        tab = self._state.get_tab(tab_id)
        if tab.run_result is None:
            raise RuntimeError("No run result available to analyze")

        ctx = self._state.exp_context
        req = AnalyzeRequest(
            run_result=tab.run_result,
            analyze_params=analyze_params,
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
        logger.info(
            "start_analyze: tab_id=%r analyze_params=%r", tab_id, list(analyze_params)
        )
        self._runner.start_analyze(tab_id, tab.adapter, req)
        self._state.set_analyzing(True)
        self._bus.emit(GuiEvent.RUN_STATE_CHANGED)

    def _on_analyze_finished(self, tab_id: str, analyze_result: Any) -> None:
        logger.info(
            "_on_analyze_finished: tab_id=%r result_type=%s",
            tab_id,
            type(analyze_result).__name__,
        )
        self._state.update_tab_analyze(tab_id, analyze_result, analyze_result.figure)
        self._state.set_analyzing(False)
        self._bus.emit(GuiEvent.RUN_STATE_CHANGED)
        self.analyze_finished.emit(tab_id, analyze_result)

    def _on_analyze_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_analyze_failed: tab_id=%r error=%r", tab_id, error)
        self._state.set_analyzing(False)
        self._bus.emit(GuiEvent.RUN_STATE_CHANGED)
        self.analyze_failed.emit(tab_id, error)

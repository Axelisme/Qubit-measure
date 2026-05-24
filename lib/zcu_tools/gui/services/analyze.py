from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.adapter import AnalyzeRequest
from zcu_tools.gui.event_bus import GuiEvent, TabInteractionChangedPayload
from zcu_tools.gui.plot_host import FigureContainer
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

    def start_analyze(
        self,
        tab_id: str,
        analyze_params_instance: object,
        figure_container: Optional[FigureContainer] = None,
    ) -> None:
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        if tab.run_result is None:
            raise RuntimeError("No run result available to analyze")

        ctx = self._state.exp_context
        req = AnalyzeRequest(
            run_result=tab.run_result,
            analyze_params=analyze_params_instance,
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
        logger.info(
            "start_analyze: tab_id=%r analyze_params_type=%s",
            tab_id,
            type(analyze_params_instance).__name__,
        )
        self._runner.start_analyze(
            tab_id,
            tab.adapter,
            req,
            figure_container=figure_container,
        )
        self._state.set_tab_analyzing(tab_id, True)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def _on_analyze_finished(self, tab_id: str, analyze_result: Any) -> None:
        logger.info(
            "_on_analyze_finished: tab_id=%r result_type=%s",
            tab_id,
            type(analyze_result).__name__,
        )
        self._state.update_tab_analyze(tab_id, analyze_result, analyze_result.figure)
        self._state.set_tab_analyzing(tab_id, False)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self.analyze_finished.emit(tab_id, analyze_result)

    def _on_analyze_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_analyze_failed: tab_id=%r error=%r", tab_id, error)
        self._state.set_tab_analyzing(tab_id, False)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self.analyze_failed.emit(tab_id, error)

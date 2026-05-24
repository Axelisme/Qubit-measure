from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.adapter import SaveDataRequest
from zcu_tools.gui.event_bus import GuiEvent, TabInteractionChangedPayload
from zcu_tools.gui.runner import SaveBothOutcome, SaveDataRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State


class SaveService(QObject):
    save_finished: Signal = Signal(str, str)
    save_failed: Signal = Signal(str, str, object)
    save_both_finished: Signal = Signal(str, object)

    def __init__(
        self,
        state: "State",
        runner: SaveDataRunner,
        bus: "EventBus",
    ) -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._bus = bus
        self._active_paths: dict[str, str] = {}

        self._runner.save_finished.connect(self._on_save_finished)
        self._runner.save_failed.connect(self._on_save_failed)
        self._runner.save_both_finished.connect(self._on_save_both_finished)

    def start_save_data(self, tab_id: str, data_path: str) -> None:
        req = self._make_save_data_request(tab_id, data_path)
        tab = self._state.get_tab(tab_id)
        logger.info("start_save_data: tab_id=%r path=%r", tab_id, data_path)
        self._runner.start_save(tab_id, tab.adapter, req)
        self._active_paths[tab_id] = data_path
        self._mark_saving(tab_id, True)

    def start_save_both(self, tab_id: str, data_path: str, image_path: str) -> None:
        """Save data and image on a worker thread; emit save_both_finished."""
        req = self._make_save_data_request(tab_id, data_path)
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise RuntimeError("No figure available to save")
        logger.info(
            "start_save_both: tab_id=%r data_path=%r image_path=%r",
            tab_id,
            data_path,
            image_path,
        )
        self._runner.start_save_both(tab_id, tab.adapter, req, tab.figure, image_path)
        self._active_paths[tab_id] = data_path
        self._mark_saving(tab_id, True)

    def save_image_sync(self, tab_id: str, image_path: str) -> None:
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise RuntimeError("No figure available to save")
        logger.info("save_image_sync: tab_id=%r path=%r", tab_id, image_path)
        tab.figure.savefig(image_path)

    def _make_save_data_request(self, tab_id: str, data_path: str) -> SaveDataRequest:
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        if tab.run_result is None:
            raise RuntimeError("No run result available to save")

        ctx = self._state.exp_context
        req = SaveDataRequest(
            run_result=tab.run_result,
            data_path=data_path,
            md=ctx.md,
            ml=ctx.ml,
            chip_name=ctx.chip_name,
            qub_name=ctx.qub_name,
            res_name=ctx.res_name,
            active_label=ctx.active_label,
        )
        return req

    def _mark_saving(self, tab_id: str, saving_data: bool) -> None:
        self._state.set_tab_saving_data(tab_id, saving_data)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def _on_save_finished(self, tab_id: str) -> None:
        path = self._active_paths.pop(tab_id)
        logger.info("_on_save_finished: tab_id=%r path=%r", tab_id, path)
        self._mark_saving(tab_id, False)
        self.save_finished.emit(tab_id, path)

    def _on_save_failed(self, tab_id: str, error: Exception) -> None:
        path = self._active_paths.pop(tab_id, "")
        logger.warning(
            "_on_save_failed: tab_id=%r path=%r error=%r", tab_id, path, error
        )
        self._mark_saving(tab_id, False)
        self.save_failed.emit(tab_id, path, error)

    def _on_save_both_finished(self, tab_id: str, outcome: SaveBothOutcome) -> None:
        self._active_paths.pop(tab_id, None)
        logger.info(
            "_on_save_both_finished: tab_id=%r data_error=%r image_error=%r",
            tab_id,
            outcome.data_error,
            outcome.image_error,
        )
        self._mark_saving(tab_id, False)
        self.save_both_finished.emit(tab_id, outcome)

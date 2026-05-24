from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.adapter import SaveDataRequest
from zcu_tools.gui.event_bus import GuiEvent, TabInteractionChangedPayload
from zcu_tools.gui.runner import SaveDataRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State


@dataclass(frozen=True)
class SaveBothOutcome:
    """Result envelope for a save_both operation."""

    data_path: str
    image_path: str
    data_error: Optional[str] = None
    image_error: Optional[str] = None


@dataclass
class _SaveBothPending:
    image_path: str
    image_error: Optional[str]


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
        self._pending_save_both: dict[str, _SaveBothPending] = {}

        self._runner.save_finished.connect(self._on_save_finished)
        self._runner.save_failed.connect(self._on_save_failed)

    def start_save_data(self, tab_id: str, data_path: str) -> None:
        self._start_save_data(tab_id, data_path)

    def start_save_both(
        self, tab_id: str, data_path: str, image_path: str
    ) -> None:
        """Save data (async) and image (sync) for a tab; emit save_both_finished.

        Image save runs in the calling (GUI) thread before the data worker
        finishes. The outcome is reported as a single SaveBothOutcome carrying
        both error slots so the caller does not need to reassemble state.
        """
        if tab_id in self._pending_save_both:
            raise RuntimeError(f"Tab {tab_id!r} already has a save_both in flight")

        image_error: Optional[str] = None
        try:
            self._save_image(tab_id, image_path)
        except Exception as exc:
            image_error = str(exc)
            logger.warning(
                "start_save_both: save_image failed tab_id=%r exc=%r", tab_id, exc
            )

        self._pending_save_both[tab_id] = _SaveBothPending(
            image_path=image_path, image_error=image_error
        )
        try:
            self._start_save_data(tab_id, data_path)
        except Exception:
            self._pending_save_both.pop(tab_id, None)
            raise

    def _save_image(self, tab_id: str, image_path: str) -> None:
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise RuntimeError("No figure available to save")
        logger.info("_save_image: tab_id=%r path=%r", tab_id, image_path)
        tab.figure.savefig(image_path)

    def _start_save_data(self, tab_id: str, data_path: str) -> None:
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
        logger.info("start_save_data: tab_id=%r path=%r", tab_id, data_path)
        self._runner.start_save(tab_id, tab.adapter, req)
        self._active_paths[tab_id] = data_path
        self._state.set_tab_saving_data(tab_id, True)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def _on_save_finished(self, tab_id: str) -> None:
        path = self._active_paths.pop(tab_id)
        logger.info("_on_save_finished: tab_id=%r path=%r", tab_id, path)
        self._state.set_tab_saving_data(tab_id, False)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        pending = self._pending_save_both.pop(tab_id, None)
        if pending is None:
            self.save_finished.emit(tab_id, path)
            return
        outcome = SaveBothOutcome(
            data_path=path,
            image_path=pending.image_path,
            data_error=None,
            image_error=pending.image_error,
        )
        self.save_both_finished.emit(tab_id, outcome)

    def _on_save_failed(self, tab_id: str, error: Exception) -> None:
        path = self._active_paths.pop(tab_id, "")
        logger.warning(
            "_on_save_failed: tab_id=%r path=%r error=%r", tab_id, path, error
        )
        self._state.set_tab_saving_data(tab_id, False)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        pending = self._pending_save_both.pop(tab_id, None)
        if pending is None:
            self.save_failed.emit(tab_id, path, error)
            return
        outcome = SaveBothOutcome(
            data_path=path,
            image_path=pending.image_path,
            data_error=str(error),
            image_error=pending.image_error,
        )
        self.save_both_finished.emit(tab_id, outcome)

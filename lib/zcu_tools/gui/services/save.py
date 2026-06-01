from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.adapter import SaveDataRequest
from zcu_tools.gui.event_bus import GuiEvent, TabInteractionChangedPayload
from zcu_tools.gui.figure_export import save_figure_to_path
from zcu_tools.gui.runner import SaveDataRunner
from zcu_tools.gui.services.guard import SavePermit

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State


@dataclass(frozen=True)
class SaveBothOutcome:
    """Combined result envelope for a save_both operation."""

    data_path: str
    image_path: str
    data_error: Optional[str] = None
    image_error: Optional[str] = None


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
        # image_path and pre-captured image_error for pending save_both operations
        self._pending_image: dict[str, tuple[str, Optional[str]]] = {}

        self._runner.save_finished.connect(self._on_save_finished)
        self._runner.save_failed.connect(self._on_save_failed)

    def start_save_data(
        self, permit: SavePermit, data_path: str, comment: str = ""
    ) -> None:
        tab_id = permit.tab_id
        req = self._make_save_data_request(tab_id, data_path, comment=comment)
        tab = self._state.get_tab(tab_id)
        logger.info("start_save_data: tab_id=%r path=%r", tab_id, data_path)
        self._ensure_parent_directory(data_path)
        self._runner.start_save(tab_id, tab.adapter, req)
        self._active_paths[tab_id] = data_path
        self._mark_saving(tab_id, True)

    def start_save_both(
        self, permit: SavePermit, data_path: str, image_path: str, comment: str = ""
    ) -> None:
        """Sync-save image on the main thread, then async-save data on a worker thread."""
        tab_id = permit.tab_id
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise RuntimeError("No figure available to save")

        # savefig runs on the main thread — no cross-thread canvas repaint.
        image_error: Optional[str] = None
        try:
            logger.info(
                "start_save_both: savefig tab_id=%r image_path=%r", tab_id, image_path
            )
            self._ensure_parent_directory(image_path)
            save_figure_to_path(tab.figure, image_path)
        except Exception as exc:
            image_error = str(exc)
            logger.warning(
                "start_save_both: image failed tab_id=%r exc=%r", tab_id, exc
            )

        req = self._make_save_data_request(tab_id, data_path, comment=comment)
        logger.info(
            "start_save_both: start_save tab_id=%r data_path=%r", tab_id, data_path
        )
        self._ensure_parent_directory(data_path)
        self._runner.start_save(tab_id, tab.adapter, req)
        self._active_paths[tab_id] = data_path
        self._pending_image[tab_id] = (image_path, image_error)
        self._mark_saving(tab_id, True)

    def save_image_sync(self, permit: SavePermit, image_path: str) -> None:
        tab_id = permit.tab_id
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise RuntimeError("No figure available to save")
        logger.info("save_image_sync: tab_id=%r path=%r", tab_id, image_path)
        self._ensure_parent_directory(image_path)
        save_figure_to_path(tab.figure, image_path)

    @staticmethod
    def _ensure_parent_directory(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _make_save_data_request(
        self, tab_id: str, data_path: str, comment: str = ""
    ) -> SaveDataRequest:
        # Run-result presence is proven by the SavePermit; tab-busy is the
        # dynamic check that stays at the operation boundary.
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
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
            comment=comment,
        )
        return req

    def _mark_saving(self, tab_id: str, saving_data: bool) -> None:
        self._state.set_tab_saving_data(tab_id, saving_data)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )

    def _on_save_finished(self, tab_id: str) -> None:
        # Default-pop for symmetry with _on_save_failed: the entry is normally
        # present (set in start_save_* before the worker can fire), but a missing
        # entry must degrade to "" rather than raise inside a terminal slot.
        path = self._active_paths.pop(tab_id, "")
        logger.info("_on_save_finished: tab_id=%r path=%r", tab_id, path)
        self._mark_saving(tab_id, False)
        if tab_id in self._pending_image:
            image_path, image_error = self._pending_image.pop(tab_id)
            logger.info(
                "_on_save_finished: save_both_finished tab_id=%r image_error=%r",
                tab_id,
                image_error,
            )
            self.save_both_finished.emit(
                tab_id,
                SaveBothOutcome(
                    data_path=path,
                    image_path=image_path,
                    data_error=None,
                    image_error=image_error,
                ),
            )
        else:
            self.save_finished.emit(tab_id, path)

    def _on_save_failed(self, tab_id: str, error: Exception) -> None:
        path = self._active_paths.pop(tab_id, "")
        logger.warning(
            "_on_save_failed: tab_id=%r path=%r error=%r", tab_id, path, error
        )
        self._mark_saving(tab_id, False)
        if tab_id in self._pending_image:
            image_path, image_error = self._pending_image.pop(tab_id)
            logger.warning(
                "_on_save_failed: save_both_finished tab_id=%r data_error=%r image_error=%r",
                tab_id,
                error,
                image_error,
            )
            self.save_both_finished.emit(
                tab_id,
                SaveBothOutcome(
                    data_path=path,
                    image_path=image_path,
                    data_error=str(error),
                    image_error=image_error,
                ),
            )
        else:
            self.save_failed.emit(tab_id, path, error)

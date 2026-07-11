from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.app.main.adapter import SaveDataRequest
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.figure_export import save_figure_to_path
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.ports import BackgroundExecutor
from zcu_tools.utils.datasaver import reserve_labber_filepath

from .guard import SavePermit

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus


@dataclass(frozen=True)
class SaveResultOutcome:
    """Combined result envelope for a save_result (data + image) operation."""

    data_path: str
    image_path: str
    data_error: str | None = None
    image_error: str | None = None


class SaveService(QObject):
    save_finished: Signal = Signal(str, str)
    save_failed: Signal = Signal(str, str, object)
    save_result_finished: Signal = Signal(str, object)

    def __init__(
        self,
        state: State,
        bg: BackgroundExecutor,
        bus: EventBus,
    ) -> None:
        super().__init__()
        self._state = state
        self._bg = bg
        self._bus = bus
        self._active_paths: dict[str, str] = {}
        # image_path and pre-captured image_error for pending save_both operations
        self._pending_image: dict[str, tuple[str, str | None]] = {}

    def _start_save(self, tab_id: str, req: SaveDataRequest) -> None:
        """Save the data file off-main (OffMain fire-forget strategy, no scopes,
        no handle — ADR-0019): adapter.save returns None, so on_done just flips
        the saving flag. The data path is already known synchronously by the
        caller; the worker only writes."""
        adapter = self._state.get_tab(tab_id).adapter
        self._bg.submit(
            lambda: adapter.save(req),
            run_in_pool=False,
            on_done=lambda _result: self._on_save_finished(tab_id),
            on_error=lambda exc: self._on_save_failed(tab_id, exc),
        )

    def start_save_data(
        self, permit: SavePermit, data_path: str, comment: str = ""
    ) -> str:
        tab_id = permit.tab_id
        # Reserve the final data path in the GUI orchestration layer so the
        # worker receives the exact file it must write.
        data_path = reserve_labber_filepath(data_path)
        req = self._make_save_data_request(tab_id, data_path, comment=comment)
        logger.info("start_save_data: tab_id=%r path=%r", tab_id, data_path)
        self._ensure_parent_directory(data_path)
        self._start_save(tab_id, req)
        self._active_paths[tab_id] = data_path
        self._mark_saving(tab_id, True, TabInteractionFact.SAVE_STARTED)
        return data_path

    def start_save_result(
        self, permit: SavePermit, data_path: str, image_path: str, comment: str = ""
    ) -> str:
        """Sync-save image on the main thread, then async-save data on a worker thread.

        Returns the resolved data path the worker will write (``.hdf5`` +
        uniqueness suffix), known synchronously up front."""
        tab_id = permit.tab_id
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise FailedPreconditionError("No figure available to save")

        # Reserve the final data path up front. Low-level persistence now writes
        # the exact path it receives, so the GUI must own uniqueness policy before
        # it submits the worker and before it reports the destination.
        data_path = reserve_labber_filepath(data_path)

        # savefig runs on the main thread — no cross-thread canvas repaint.
        image_error: str | None = None
        try:
            logger.info(
                "start_save_result: savefig tab_id=%r image_path=%r", tab_id, image_path
            )
            self._ensure_parent_directory(image_path)
            save_figure_to_path(tab.figure, image_path)
        except Exception as exc:
            image_error = str(exc)
            logger.warning(
                "start_save_result: image failed tab_id=%r exc=%r", tab_id, exc
            )

        req = self._make_save_data_request(tab_id, data_path, comment=comment)
        logger.info(
            "start_save_result: start_save tab_id=%r data_path=%r", tab_id, data_path
        )
        self._ensure_parent_directory(data_path)
        self._start_save(tab_id, req)
        self._active_paths[tab_id] = data_path
        self._pending_image[tab_id] = (image_path, image_error)
        self._mark_saving(tab_id, True, TabInteractionFact.SAVE_STARTED)
        return data_path

    def save_image_sync(self, permit: SavePermit, image_path: str) -> None:
        tab_id = permit.tab_id
        tab = self._state.get_tab(tab_id)
        if tab.figure is None:
            raise FailedPreconditionError("No figure available to save")
        logger.info("save_image_sync: tab_id=%r path=%r", tab_id, image_path)
        self._ensure_parent_directory(image_path)
        save_figure_to_path(tab.figure, image_path)

    def save_post_image_sync(self, permit: SavePermit, image_path: str) -> None:
        """Save the tab's *post-analysis* figure (``tab.post_figure``) — the post
        sub-tab's own Save Image. Mirrors ``save_image_sync`` but targets the
        post layer's figure, which is distinct from the primary ``tab.figure``
        (the two are separate State fields though they share one container)."""
        tab_id = permit.tab_id
        tab = self._state.get_tab(tab_id)
        if tab.post_figure is None:
            raise FailedPreconditionError("No post-analysis figure available to save")
        logger.info("save_post_image_sync: tab_id=%r path=%r", tab_id, image_path)
        self._ensure_parent_directory(image_path)
        save_figure_to_path(tab.post_figure, image_path)

    @staticmethod
    def _ensure_parent_directory(path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _make_save_data_request(
        self, tab_id: str, data_path: str, comment: str = ""
    ) -> SaveDataRequest:
        # Run-result presence is proven by the SavePermit; tab-busy is the
        # dynamic check that stays at the operation boundary.
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

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

    def _mark_saving(
        self,
        tab_id: str,
        saving_data: bool,
        fact: TabInteractionFact,
    ) -> None:
        self._state.set_tab_saving_data(tab_id, saving_data)
        self._bus.emit(
            TabInteractionChangedPayload(tab_id=tab_id, fact=fact),
        )

    def _on_save_finished(self, tab_id: str) -> None:
        # Default-pop for symmetry with _on_save_failed: the entry is normally
        # present (set in start_save_* before the worker can fire), but a missing
        # entry must degrade to "" rather than raise inside a terminal slot.
        path = self._active_paths.pop(tab_id, "")
        logger.info("_on_save_finished: tab_id=%r path=%r", tab_id, path)
        self._mark_saving(tab_id, False, TabInteractionFact.SAVE_SUCCEEDED)
        if tab_id in self._pending_image:
            image_path, image_error = self._pending_image.pop(tab_id)
            logger.info(
                "_on_save_finished: save_result_finished tab_id=%r image_error=%r",
                tab_id,
                image_error,
            )
            self.save_result_finished.emit(
                tab_id,
                SaveResultOutcome(
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
        self._mark_saving(tab_id, False, TabInteractionFact.SAVE_FAILED)
        if tab_id in self._pending_image:
            image_path, image_error = self._pending_image.pop(tab_id)
            logger.warning(
                "_on_save_failed: save_result_finished tab_id=%r data_error=%r image_error=%r",
                tab_id,
                error,
                image_error,
            )
            self.save_result_finished.emit(
                tab_id,
                SaveResultOutcome(
                    data_path=path,
                    image_path=image_path,
                    data_error=str(error),
                    image_error=image_error,
                ),
            )
        else:
            self.save_failed.emit(tab_id, path, error)

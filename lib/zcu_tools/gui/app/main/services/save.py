from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import SaveDataRequest
from zcu_tools.gui.app.main.artifact_tracker import ArtifactKind
from zcu_tools.gui.app.main.events.completion import SaveDataFinishedPayload
from zcu_tools.gui.app.main.events.tab import (
    TabInteractionChangedPayload,
    TabInteractionFact,
)
from zcu_tools.gui.app.main.figure_export import save_figure_to_path
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import (
    BgResult,
    OperationRunner,
    OperationSpec,
    SettleFn,
)
from zcu_tools.utils.datasaver import reserve_labber_filepath

from .guard import SavePermit
from .ports import ActiveSaveOperation, SaveDataSubmission

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus


class SaveService:
    def __init__(
        self,
        state: State,
        runner: OperationRunner,
        bus: EventBus,
    ) -> None:
        self._state = state
        self._runner = runner
        self._bus = bus
        self._active_paths: dict[str, str] = {}
        self._active_operations: dict[str, int] = {}

    def active_save_operations(self) -> tuple[ActiveSaveOperation, ...]:
        return tuple(
            ActiveSaveOperation(token, tab_id)
            for tab_id, token in self._active_operations.items()
        )

    def _start_save(self, tab_id: str, req: SaveDataRequest) -> int:
        """Submit non-cancellable I/O to the shared handle lifecycle, without a lease."""
        adapter = self._state.get_tab(tab_id).adapter

        def on_terminal(result: BgResult, settle: SettleFn) -> None:
            error = (
                None
                if result.ok
                else result.error or RuntimeError("save failed without an error")
            )
            payload = self._finish_save_state(tab_id, error)
            settle(
                OperationOutcome(
                    "finished" if error is None else "failed", payload.error
                )
            )
            self._emit_save_finished(payload)

        return self._runner.begin(
            OperationSpec(
                exclusion=None,
                owner_id=tab_id,
                wants_progress=False,
                cancel_hook=None,
                work=lambda _factory: adapter.save(req),
                run_in_pool=False,
                on_terminal=on_terminal,
            )
        )

    def start_save_data(
        self, permit: SavePermit, data_path: str, comment: str = ""
    ) -> SaveDataSubmission:
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        # Reserve the final data path in the GUI orchestration layer so the
        # worker receives the exact file it must write.
        data_path = reserve_labber_filepath(data_path)
        req = self._make_save_data_request(tab_id, data_path, comment=comment)
        logger.info("start_save_data: tab_id=%r path=%r", tab_id, data_path)
        self._ensure_parent_directory(data_path)
        self._state.get_artifact_snapshots(tab_id)
        tracker = self._state.get_tab(tab_id).artifacts
        tracker.started(ArtifactKind.DATA)
        self._active_paths[tab_id] = data_path
        self._mark_saving(tab_id, True, TabInteractionFact.SAVE_STARTED)
        try:
            token = self._start_save(tab_id, req)
        except Exception as error:
            # OperationRunner has already settled a synchronous submission failure.
            self._emit_save_finished(self._finish_save_state(tab_id, error))
            raise
        # Inline executors may have delivered terminal before begin returns.
        if tab_id in self._active_paths:
            self._active_operations[tab_id] = token
        return SaveDataSubmission(token, data_path)

    def save_image_sync(self, permit: SavePermit, image_path: str) -> None:
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        tab = self._state.get_tab(tab_id)
        if tab.analysis.figure is None:
            raise FailedPreconditionError("No figure available to save")
        logger.info("save_image_sync: tab_id=%r path=%r", tab_id, image_path)
        self._ensure_parent_directory(image_path)
        self._state.get_artifact_snapshots(tab_id)
        tab.artifacts.started(ArtifactKind.ANALYSIS)
        try:
            save_figure_to_path(tab.analysis.figure, image_path)
        except Exception:
            tab.artifacts.failed(ArtifactKind.ANALYSIS)
            raise
        tab.artifacts.succeeded(ArtifactKind.ANALYSIS, image_path)

    def save_post_image_sync(self, permit: SavePermit, image_path: str) -> None:
        """Save the tab's *post-analysis* figure (``tab.post_analysis.figure``) — the post
        sub-tab's own Save Image. Mirrors ``save_image_sync`` but targets the
        post layer's figure, which is distinct from the primary ``tab.analysis.figure``
        (the two are separate pane fields though they share container routing)."""
        tab_id = permit.tab_id
        self._require_tab_idle(tab_id)
        tab = self._state.get_tab(tab_id)
        if tab.post_analysis.figure is None:
            raise FailedPreconditionError("No post-analysis figure available to save")
        logger.info("save_post_image_sync: tab_id=%r path=%r", tab_id, image_path)
        self._ensure_parent_directory(image_path)
        self._state.get_artifact_snapshots(tab_id)
        tab.artifacts.started(ArtifactKind.POST_ANALYSIS)
        try:
            save_figure_to_path(tab.post_analysis.figure, image_path)
        except Exception:
            tab.artifacts.failed(ArtifactKind.POST_ANALYSIS)
            raise
        tab.artifacts.succeeded(ArtifactKind.POST_ANALYSIS, image_path)

    def _require_tab_idle(self, tab_id: str) -> None:
        """Reject every save entry point while another tab operation owns it.

        ``GuardService`` owns static preconditions; the tab's busy state is a
        dynamic resource check and must happen before path reservation or image
        export so a Save permit cannot bypass same-tab operation exclusion.
        """
        if self._state.is_tab_busy(tab_id):
            raise FailedPreconditionError(f"Tab {tab_id!r} is busy")

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
            run_result=tab.run.result,
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

    def _finish_save_state(
        self, tab_id: str, error: Exception | None
    ) -> SaveDataFinishedPayload:
        path = self._active_paths.pop(tab_id, "")
        self._active_operations.pop(tab_id, None)
        tracker = self._state.get_tab(tab_id).artifacts
        if error is None:
            logger.info("save finished: tab_id=%r path=%r", tab_id, path)
            tracker.succeeded(ArtifactKind.DATA, path)
        else:
            logger.warning(
                "save failed: tab_id=%r path=%r error=%r", tab_id, path, error
            )
            tracker.failed(ArtifactKind.DATA)
        self._state.set_tab_saving_data(tab_id, False)
        return SaveDataFinishedPayload(
            tab_id=tab_id, data_path=path, error=None if error is None else str(error)
        )

    def _emit_save_finished(self, payload: SaveDataFinishedPayload) -> None:
        """Publish only after State and the shared handle agree on terminal state."""
        fact = (
            TabInteractionFact.SAVE_SUCCEEDED
            if payload.error is None
            else TabInteractionFact.SAVE_FAILED
        )
        self._bus.emit(TabInteractionChangedPayload(tab_id=payload.tab_id, fact=fact))
        self._bus.emit(payload)

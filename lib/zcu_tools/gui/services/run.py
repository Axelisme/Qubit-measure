from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

from qtpy.QtCore import QObject, Signal  # type: ignore[attr-defined]

from zcu_tools.gui.adapter import CfgSchema, RunRequest, require_soc_handles
from zcu_tools.gui.event_bus import (
    GuiEvent,
    RunFailedPayload,
    RunFinishedPayload,
    RunLockChangedPayload,
    RunProgressChangedPayload,
    TabInteractionChangedPayload,
)
from zcu_tools.gui.plot_host import FigureContainer
from zcu_tools.gui.services.operation_gate import (
    OperationGate,
    OperationKind,
    OperationLease,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.runner import Runner
    from zcu_tools.gui.state import State


class RunService(QObject):
    """Encapsulates execution of an experiment adapter via a Runner."""

    run_finished: Signal = Signal(str, object)
    run_failed: Signal = Signal(str, object)

    def __init__(
        self,
        state: "State",
        runner: "Runner",
        bus: "EventBus",
        gate: OperationGate,
    ) -> None:
        super().__init__()
        self._state = state
        self._runner = runner
        self._bus = bus
        self._gate = gate
        self._active_lease: Optional[OperationLease] = None
        self._active_pbar_factory: Optional[Any] = None

        self._runner.run_finished.connect(self._on_run_finished)
        self._runner.run_failed.connect(self._on_run_failed)

    def start_run(
        self,
        tab_id: str,
        schema: CfgSchema,
        pbar_factory: Optional[Any] = None,
        live_container: Optional[FigureContainer] = None,
    ) -> None:
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        ctx = self._state.exp_context
        req = RunRequest(
            md=ctx.md,
            ml=ctx.ml,
            soc=ctx.soc,
            soccfg=ctx.soccfg,
        )

        self._active_pbar_factory = pbar_factory
        if pbar_factory is not None and hasattr(pbar_factory, "set_progress_callback"):
            pbar_factory.set_progress_callback(self._emit_progress_changed)

        # Validate schema before starting the worker
        schema.to_raw_dict(req)
        logger.info("start_run: tab_id=%r", tab_id)

        tab = self._state.get_tab(tab_id)
        if tab.adapter.capabilities.requires_soc:
            require_soc_handles(req)
        lease = self._gate.acquire(OperationKind.RUN, owner_id=tab_id)
        self._active_lease = lease
        try:
            self._runner.start_run(
                tab_id,
                tab.adapter,
                req,
                schema,
                pbar_factory=pbar_factory,
                figure_container=live_container,
            )
        except Exception:
            self._active_lease = None
            self._gate.release(lease)
            raise
        self._state.set_tab_running(tab_id, True)
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
            GuiEvent.RUN_LOCK_CHANGED, RunLockChangedPayload(running_tab_id=tab_id)
        )

    def get_run_progress(self) -> Tuple[Any, ...]:
        from zcu_tools.progress_bar.backend.qt import QtProgressBarFactory

        factory = self._active_pbar_factory
        if not isinstance(factory, QtProgressBarFactory):
            return ()
        return factory.get_all_snapshots()

    def cancel_run(self) -> None:
        logger.info("cancel_run")
        self._runner.cancel()

    def _clear_active_factory(self) -> None:
        self._active_pbar_factory = None

    def _emit_progress_changed(self) -> None:
        tab_id = self._state.running_tab_id or ""
        self._bus.emit(
            GuiEvent.RUN_PROGRESS_CHANGED, RunProgressChangedPayload(tab_id=tab_id)
        )

    def _on_run_finished(self, tab_id: str, result: Any) -> None:
        logger.info(
            "_on_run_finished: tab_id=%r result_type=%s", tab_id, type(result).__name__
        )
        self._clear_active_factory()
        self._state.update_tab_result(tab_id, result)
        self._state.set_tab_running(tab_id, False)
        self._release_lease()
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
            GuiEvent.RUN_LOCK_CHANGED, RunLockChangedPayload(running_tab_id=None)
        )
        self._bus.emit(GuiEvent.RUN_FINISHED, RunFinishedPayload(tab_id=tab_id))
        self.run_finished.emit(tab_id, result)

    def _on_run_failed(self, tab_id: str, error: Exception) -> None:
        logger.warning("_on_run_failed: tab_id=%r error=%r", tab_id, error)
        self._clear_active_factory()
        self._state.set_tab_running(tab_id, False)
        self._release_lease()
        self._bus.emit(
            GuiEvent.TAB_INTERACTION_CHANGED,
            TabInteractionChangedPayload(tab_id=tab_id),
        )
        self._bus.emit(
            GuiEvent.RUN_LOCK_CHANGED, RunLockChangedPayload(running_tab_id=None)
        )
        self._bus.emit(
            GuiEvent.RUN_FAILED,
            RunFailedPayload(tab_id=tab_id, error_message=str(error)),
        )
        self.run_failed.emit(tab_id, error)

    def _release_lease(self) -> None:
        lease = self._active_lease
        if lease is None:
            raise RuntimeError("Run completed without an active operation lease")
        self._active_lease = None
        self._gate.release(lease)

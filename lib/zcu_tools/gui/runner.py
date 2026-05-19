from __future__ import annotations

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

from qtpy.QtCore import QObject, QThread, Signal  # type: ignore[attr-defined]

from zcu_tools.experiment.v2.runner.base import ActiveTask
from zcu_tools.progress_bar.interface import use_pbar_factory

from .adapter import AbsExpAdapter, CfgSchema, ExpContext


class RunWorker(QThread):
    run_finished: Signal = Signal(object)  # result
    run_failed: Signal = Signal(object)  # Exception

    def __init__(
        self,
        adapter: AbsExpAdapter,
        ctx: ExpContext,
        schema: CfgSchema,
        user_params: dict,
        pbar_factory: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._ctx = ctx
        self._schema = schema
        self._user_params = user_params
        self._pbar_factory = pbar_factory
        self._stop_event = threading.Event()

    def run(self) -> None:
        logger.debug("RunWorker.run: start adapter=%s", type(self._adapter).__name__)
        try:
            if self._pbar_factory is not None:
                with ActiveTask(self._stop_event), use_pbar_factory(self._pbar_factory):
                    result = self._adapter.run(
                        self._ctx, self._schema, **self._user_params
                    )
            else:
                with ActiveTask(self._stop_event):
                    result = self._adapter.run(
                        self._ctx, self._schema, **self._user_params
                    )
            logger.debug(
                "RunWorker.run: finished result_type=%s", type(result).__name__
            )
            self.run_finished.emit(result)
        except Exception as exc:
            logger.warning("RunWorker.run: failed exc=%r", exc)
            self.run_failed.emit(exc)

    def cancel(self) -> None:
        logger.debug("RunWorker.cancel: setting stop event")
        self._stop_event.set()


class Runner(QObject):
    """Manages a single RunWorker; forwards its signals with the tab_id attached."""

    run_finished: Signal = Signal(str, object)  # tab_id, result
    run_failed: Signal = Signal(str, object)  # tab_id, Exception

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[RunWorker] = None
        self._active_tab_id: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start_run(
        self,
        tab_id: str,
        adapter: AbsExpAdapter,
        ctx: ExpContext,
        schema: CfgSchema,
        user_params: dict,
        pbar_factory: Optional[Any] = None,
    ) -> None:
        if self.is_running:
            raise RuntimeError(
                f"Cannot start run for tab {tab_id!r}: another run is already active"
            )
        logger.debug(
            "Runner.start_run: tab_id=%r adapter=%s", tab_id, type(adapter).__name__
        )
        self._active_tab_id = tab_id
        worker = RunWorker(adapter, ctx, schema, user_params, pbar_factory)
        worker.run_finished.connect(self._on_worker_finished)
        worker.run_failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def cancel(self) -> None:
        if self._worker is not None:
            logger.debug("Runner.cancel: requesting stop")
            self._worker.cancel()

    def _on_worker_finished(self, result: Any) -> None:
        tab_id = self._active_tab_id or ""
        logger.debug("Runner._on_worker_finished: tab_id=%r", tab_id)
        self._worker = None
        self._active_tab_id = None
        self.run_finished.emit(tab_id, result)

    def _on_worker_failed(self, exc: Exception) -> None:
        tab_id = self._active_tab_id or ""
        logger.warning("Runner._on_worker_failed: tab_id=%r exc=%r", tab_id, exc)
        self._worker = None
        self._active_tab_id = None
        self.run_failed.emit(tab_id, exc)

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

from qtpy.QtCore import QObject, QThread, Signal  # type: ignore[attr-defined]

from zcu_tools.experiment.v2.runner.base import ActiveTask
from zcu_tools.progress_bar.interface import use_pbar_factory

from .adapter import (
    AnalyzeRequest,
    CfgSchema,
    ExpAdapterProtocol,
    RunRequest,
    SaveDataRequest,
)
from .plot_host import FigureContainer
from .plot_routing import routing_scope

logger = logging.getLogger(__name__)

AdapterHandle = ExpAdapterProtocol
NO_RESULT = object()


class RunWorker(QThread):
    run_finished: Signal = Signal(object)
    run_failed: Signal = Signal(object)
    run_cancelled: Signal = Signal(object)  # partial result, or NO_RESULT

    def __init__(
        self,
        adapter: AdapterHandle,
        req: RunRequest,
        schema: CfgSchema,
        stop_event: threading.Event,
        pbar_factory: Optional[Callable[..., Any]] = None,
        figure_container: Optional[FigureContainer] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._adapter = adapter
        self._req = req
        self._schema = schema
        self._pbar_factory = pbar_factory
        self._figure_container = figure_container
        # The cancellation flag is owned by RunService and passed in, so the same
        # handle is registered with the OperationGate (which sets it on cancel).
        self._stop_event = stop_event
        self._result: object = NO_RESULT
        self._error: Optional[Exception] = None
        self.finished.connect(self._emit_outcome)

    def run(self) -> None:
        logger.debug("RunWorker.run: start adapter=%s", type(self._adapter).__name__)
        try:
            with routing_scope(self._figure_container), ActiveTask(self._stop_event):
                if self._pbar_factory is not None:
                    with use_pbar_factory(self._pbar_factory):
                        result = self._adapter.run(self._req, self._schema)
                else:
                    result = self._adapter.run(self._req, self._schema)
            logger.debug(
                "RunWorker.run: finished result_type=%s", type(result).__name__
            )
            self._result = result
        except Exception as exc:
            logger.warning("RunWorker.run: failed exc=%r", exc)
            self._error = exc

    def _emit_outcome(self) -> None:
        # Self-judge cancellation first: a cancelled run either raises (the task
        # interrupted itself on the stop_event) or returns a partial result —
        # both read as 'cancelled' once the stop_event is set, so the worker
        # reports it directly instead of the service tracking "who I cancelled".
        if self._stop_event.is_set():
            self.run_cancelled.emit(self._result)
        elif self._error is not None:
            self.run_failed.emit(self._error)
        elif self._result is not NO_RESULT:
            self.run_finished.emit(self._result)
        else:
            raise RuntimeError("RunWorker stopped without an outcome")


class AnalyzeWorker(QThread):
    analyze_finished: Signal = Signal(object)
    analyze_failed: Signal = Signal(object)

    def __init__(
        self,
        adapter: AdapterHandle,
        req: AnalyzeRequest[Any, Any],
        figure_container: Optional[FigureContainer] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._adapter = adapter
        self._req = req
        self._figure_container = figure_container
        self._result: object = NO_RESULT
        self._error: Optional[Exception] = None
        self.finished.connect(self._emit_outcome)

    def run(self) -> None:
        logger.debug(
            "AnalyzeWorker.run: start adapter=%s", type(self._adapter).__name__
        )
        try:
            with routing_scope(self._figure_container):
                result = self._adapter.analyze(self._req)
            logger.debug(
                "AnalyzeWorker.run: finished result_type=%s", type(result).__name__
            )
            self._result = result
        except Exception as exc:
            logger.warning("AnalyzeWorker.run: failed exc=%r", exc)
            self._error = exc

    def _emit_outcome(self) -> None:
        if self._error is not None:
            self.analyze_failed.emit(self._error)
        elif self._result is not NO_RESULT:
            self.analyze_finished.emit(self._result)
        else:
            raise RuntimeError("AnalyzeWorker stopped without an outcome")


class SaveDataWorker(QThread):
    save_finished: Signal = Signal()
    save_failed: Signal = Signal(object)

    def __init__(
        self,
        adapter: AdapterHandle,
        req: SaveDataRequest[Any],
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._adapter = adapter
        self._req = req
        self._completed = False
        self._error: Optional[Exception] = None
        self.finished.connect(self._emit_outcome)

    def run(self) -> None:
        logger.debug(
            "SaveDataWorker.run: start adapter=%s path=%r",
            type(self._adapter).__name__,
            self._req.data_path,
        )
        try:
            self._adapter.save(self._req)
            self._completed = True
        except Exception as exc:
            logger.warning("SaveDataWorker.run: failed exc=%r", exc)
            self._error = exc

    def _emit_outcome(self) -> None:
        if self._error is not None:
            self.save_failed.emit(self._error)
        elif self._completed:
            self.save_finished.emit()
        else:
            raise RuntimeError("SaveDataWorker stopped without an outcome")


class Runner(QObject):
    """Manages a single RunWorker; forwards its signals with the tab_id attached.

    Cancellation flows through the stop_event the RunService owns and passes to
    ``start_run`` (and to the OperationGate); the Runner never sets it. The
    worker self-judges cancelled (stop_event set) and emits ``run_cancelled``
    with whatever partial result it has, which the Runner forwards.
    """

    run_finished: Signal = Signal(str, object)
    run_failed: Signal = Signal(str, object)
    run_cancelled: Signal = Signal(str, object)

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
        adapter: AdapterHandle,
        req: RunRequest,
        schema: CfgSchema,
        stop_event: threading.Event,
        pbar_factory: Optional[Callable[..., Any]] = None,
        figure_container: Optional[FigureContainer] = None,
    ) -> None:
        if self.is_running:
            raise RuntimeError(
                f"Cannot start run for tab {tab_id!r}: another run is already active"
            )
        logger.debug(
            "Runner.start_run: tab_id=%r adapter=%s", tab_id, type(adapter).__name__
        )
        self._active_tab_id = tab_id
        worker = RunWorker(
            adapter,
            req,
            schema,
            stop_event,
            pbar_factory,
            figure_container,
            parent=self,
        )
        worker.finished.connect(worker.deleteLater)
        worker.run_finished.connect(self._on_worker_finished)
        worker.run_failed.connect(self._on_worker_failed)
        worker.run_cancelled.connect(self._on_worker_cancelled)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self, result: Any) -> None:
        tab_id = self._take_active_tab_id("finished")
        logger.debug("Runner._on_worker_finished: tab_id=%r", tab_id)
        self.run_finished.emit(tab_id, result)

    def _on_worker_failed(self, exc: Exception) -> None:
        tab_id = self._take_active_tab_id("failed")
        logger.warning("Runner._on_worker_failed: tab_id=%r exc=%r", tab_id, exc)
        self.run_failed.emit(tab_id, exc)

    def _on_worker_cancelled(self, result: Any) -> None:
        tab_id = self._take_active_tab_id("cancelled")
        logger.debug("Runner._on_worker_cancelled: tab_id=%r", tab_id)
        self.run_cancelled.emit(tab_id, result)

    def _take_active_tab_id(self, terminal: str) -> str:
        if self._active_tab_id is None:
            raise RuntimeError(f"RunWorker {terminal} without an active tab id")
        tab_id = self._active_tab_id
        self._worker = None
        self._active_tab_id = None
        return tab_id


class AnalyzeRunner(QObject):
    analyze_finished: Signal = Signal(str, object)
    analyze_failed: Signal = Signal(str, object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._workers: dict[str, AnalyzeWorker] = {}

    @property
    def is_running(self) -> bool:
        return any(worker.isRunning() for worker in self._workers.values())

    def start_analyze(
        self,
        tab_id: str,
        adapter: AdapterHandle,
        req: AnalyzeRequest[Any, Any],
        figure_container: Optional[FigureContainer] = None,
    ) -> None:
        if tab_id in self._workers and self._workers[tab_id].isRunning():
            raise RuntimeError(
                f"Cannot start analyze for tab {tab_id!r}: another analyze is already active for this tab"
            )
        logger.debug(
            "AnalyzeRunner.start_analyze: tab_id=%r adapter=%s",
            tab_id,
            type(adapter).__name__,
        )
        worker = AnalyzeWorker(adapter, req, figure_container, parent=self)
        worker.finished.connect(worker.deleteLater)
        worker.analyze_finished.connect(
            lambda result, tid=tab_id: self._on_worker_finished(tid, result)
        )
        worker.analyze_failed.connect(
            lambda exc, tid=tab_id: self._on_worker_failed(tid, exc)
        )
        self._workers[tab_id] = worker
        worker.start()

    def _on_worker_finished(self, tab_id: str, result: Any) -> None:
        self._workers.pop(tab_id, None)
        self.analyze_finished.emit(tab_id, result)

    def _on_worker_failed(self, tab_id: str, exc: Exception) -> None:
        self._workers.pop(tab_id, None)
        self.analyze_failed.emit(tab_id, exc)


class SaveDataRunner(QObject):
    save_finished: Signal = Signal(str)
    save_failed: Signal = Signal(str, object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._workers: dict[str, SaveDataWorker] = {}

    @property
    def is_running(self) -> bool:
        return any(worker.isRunning() for worker in self._workers.values())

    def start_save(
        self,
        tab_id: str,
        adapter: AdapterHandle,
        req: SaveDataRequest[Any],
    ) -> None:
        if tab_id in self._workers and self._workers[tab_id].isRunning():
            raise RuntimeError(
                f"Cannot start save for tab {tab_id!r}: another save is already active for this tab"
            )
        logger.debug(
            "SaveDataRunner.start_save: tab_id=%r adapter=%s",
            tab_id,
            type(adapter).__name__,
        )
        worker = SaveDataWorker(adapter, req, parent=self)
        worker.finished.connect(worker.deleteLater)
        worker.save_finished.connect(lambda tid=tab_id: self._on_worker_finished(tid))
        worker.save_failed.connect(
            lambda exc, tid=tab_id: self._on_worker_failed(tid, exc)
        )
        self._workers[tab_id] = worker
        worker.start()

    def _on_worker_finished(self, tab_id: str) -> None:
        self._workers.pop(tab_id, None)
        self.save_finished.emit(tab_id)

    def _on_worker_failed(self, tab_id: str, exc: Exception) -> None:
        self._workers.pop(tab_id, None)
        self.save_failed.emit(tab_id, exc)

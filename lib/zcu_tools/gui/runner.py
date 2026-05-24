from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from matplotlib.figure import Figure
from qtpy.QtCore import QObject, QThread, Signal  # type: ignore[attr-defined]

from zcu_tools.experiment.v2.runner.base import ActiveTask
from zcu_tools.progress_bar.interface import use_pbar_factory

from .adapter import (
    AbsExpAdapter,
    AnalyzeRequest,
    CfgSchema,
    RunRequest,
    SaveDataRequest,
)
from .plot_host import FigureContainer
from .plot_routing import routing_scope

logger = logging.getLogger(__name__)

AdapterHandle = AbsExpAdapter[Any, Any]


@dataclass(frozen=True)
class SaveBothOutcome:
    """Result envelope for a save_both operation."""

    data_path: str
    image_path: str
    data_error: Optional[str] = None
    image_error: Optional[str] = None


class RunWorker(QThread):
    run_finished: Signal = Signal(object)
    run_failed: Signal = Signal(object)

    def __init__(
        self,
        adapter: AdapterHandle,
        req: RunRequest,
        schema: CfgSchema,
        pbar_factory: Optional[Callable[..., Any]] = None,
        figure_container: Optional[FigureContainer] = None,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._req = req
        self._schema = schema
        self._pbar_factory = pbar_factory
        self._figure_container = figure_container
        self._stop_event = threading.Event()

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
            self.run_finished.emit(result)
        except Exception as exc:
            logger.warning("RunWorker.run: failed exc=%r", exc)
            self.run_failed.emit(exc)

    def cancel(self) -> None:
        logger.debug("RunWorker.cancel: setting stop event")
        self._stop_event.set()


class AnalyzeWorker(QThread):
    analyze_finished: Signal = Signal(object)
    analyze_failed: Signal = Signal(object)

    def __init__(
        self,
        adapter: AdapterHandle,
        req: AnalyzeRequest[Any],
        figure_container: Optional[FigureContainer] = None,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._req = req
        self._figure_container = figure_container

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
            self.analyze_finished.emit(result)
        except Exception as exc:
            logger.warning("AnalyzeWorker.run: failed exc=%r", exc)
            self.analyze_failed.emit(exc)


class SaveDataWorker(QThread):
    save_finished: Signal = Signal()
    save_failed: Signal = Signal(object)

    def __init__(self, adapter: AdapterHandle, req: SaveDataRequest[Any]) -> None:
        super().__init__()
        self._adapter = adapter
        self._req = req

    def run(self) -> None:
        logger.debug(
            "SaveDataWorker.run: start adapter=%s path=%r",
            type(self._adapter).__name__,
            self._req.data_path,
        )
        try:
            self._adapter.save(self._req)
            self.save_finished.emit()
        except Exception as exc:
            logger.warning("SaveDataWorker.run: failed exc=%r", exc)
            self.save_failed.emit(exc)


class SaveBothWorker(QThread):
    save_both_finished: Signal = Signal(object)

    def __init__(
        self,
        adapter: AdapterHandle,
        req: SaveDataRequest[Any],
        figure: Figure,
        image_path: str,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._req = req
        self._figure = figure
        self._image_path = image_path

    def run(self) -> None:
        logger.debug(
            "SaveBothWorker.run: start adapter=%s data_path=%r image_path=%r",
            type(self._adapter).__name__,
            self._req.data_path,
            self._image_path,
        )
        data_error: Optional[str] = None
        image_error: Optional[str] = None
        try:
            self._figure.savefig(self._image_path)
        except Exception as exc:
            image_error = str(exc)
            logger.warning("SaveBothWorker.run: image failed exc=%r", exc)

        try:
            self._adapter.save(self._req)
        except Exception as exc:
            data_error = str(exc)
            logger.warning("SaveBothWorker.run: data failed exc=%r", exc)

        self.save_both_finished.emit(
            SaveBothOutcome(
                data_path=self._req.data_path,
                image_path=self._image_path,
                data_error=data_error,
                image_error=image_error,
            )
        )


class Runner(QObject):
    """Manages a single RunWorker; forwards its signals with the tab_id attached."""

    run_finished: Signal = Signal(str, object)
    run_failed: Signal = Signal(str, object)

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
            pbar_factory,
            figure_container,
        )
        worker.run_finished.connect(self._on_worker_finished)
        worker.run_failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def cancel(self) -> None:
        if self._worker is not None:
            logger.debug("Runner.cancel: requesting stop")
            self._worker.cancel()

    def _on_worker_finished(self, result: Any) -> None:
        if self._active_tab_id is None:
            raise RuntimeError("RunWorker finished without an active tab id")
        tab_id = self._active_tab_id
        logger.debug("Runner._on_worker_finished: tab_id=%r", tab_id)
        self._worker = None
        self._active_tab_id = None
        self.run_finished.emit(tab_id, result)

    def _on_worker_failed(self, exc: Exception) -> None:
        if self._active_tab_id is None:
            raise RuntimeError("RunWorker failed without an active tab id")
        tab_id = self._active_tab_id
        logger.warning("Runner._on_worker_failed: tab_id=%r exc=%r", tab_id, exc)
        self._worker = None
        self._active_tab_id = None
        self.run_failed.emit(tab_id, exc)


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
        req: AnalyzeRequest[Any],
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
        worker = AnalyzeWorker(adapter, req, figure_container)
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
    save_both_finished: Signal = Signal(str, object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._workers: dict[str, SaveDataWorker | SaveBothWorker] = {}

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
        worker = SaveDataWorker(adapter, req)
        worker.save_finished.connect(lambda tid=tab_id: self._on_worker_finished(tid))
        worker.save_failed.connect(
            lambda exc, tid=tab_id: self._on_worker_failed(tid, exc)
        )
        self._workers[tab_id] = worker
        worker.start()

    def start_save_both(
        self,
        tab_id: str,
        adapter: AdapterHandle,
        req: SaveDataRequest[Any],
        figure: Figure,
        image_path: str,
    ) -> None:
        if tab_id in self._workers and self._workers[tab_id].isRunning():
            raise RuntimeError(
                f"Cannot start save_both for tab {tab_id!r}: another save is already active for this tab"
            )
        logger.debug(
            "SaveDataRunner.start_save_both: tab_id=%r adapter=%s",
            tab_id,
            type(adapter).__name__,
        )
        worker = SaveBothWorker(adapter, req, figure, image_path)
        worker.save_both_finished.connect(
            lambda outcome, tid=tab_id: self._on_save_both_finished(tid, outcome)
        )
        self._workers[tab_id] = worker
        worker.start()

    def _on_worker_finished(self, tab_id: str) -> None:
        self._workers.pop(tab_id, None)
        self.save_finished.emit(tab_id)

    def _on_worker_failed(self, tab_id: str, exc: Exception) -> None:
        self._workers.pop(tab_id, None)
        self.save_failed.emit(tab_id, exc)

    def _on_save_both_finished(self, tab_id: str, outcome: SaveBothOutcome) -> None:
        self._workers.pop(tab_id, None)
        self.save_both_finished.emit(tab_id, outcome)

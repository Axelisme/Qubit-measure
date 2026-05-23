from __future__ import annotations

import logging
import threading
from typing import Any, Optional

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

logger = logging.getLogger(__name__)


class RunWorker(QThread):
    run_finished: Signal = Signal(object)
    run_failed: Signal = Signal(object)

    def __init__(
        self,
        adapter: AbsExpAdapter,
        req: RunRequest,
        schema: CfgSchema,
        user_params: dict[str, object],
        pbar_factory: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self._adapter = adapter
        self._req = req
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
                        self._req, self._schema, **self._user_params
                    )
            else:
                with ActiveTask(self._stop_event):
                    result = self._adapter.run(
                        self._req, self._schema, **self._user_params
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


class AnalyzeWorker(QThread):
    analyze_finished: Signal = Signal(object)
    analyze_failed: Signal = Signal(object)

    def __init__(self, adapter: AbsExpAdapter, req: AnalyzeRequest) -> None:
        super().__init__()
        self._adapter = adapter
        self._req = req

    def run(self) -> None:
        logger.debug(
            "AnalyzeWorker.run: start adapter=%s", type(self._adapter).__name__
        )
        try:
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

    def __init__(self, adapter: AbsExpAdapter, req: SaveDataRequest) -> None:
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
        adapter: AbsExpAdapter,
        req: RunRequest,
        schema: CfgSchema,
        user_params: dict[str, object],
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
        worker = RunWorker(adapter, req, schema, user_params, pbar_factory)
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


class AnalyzeRunner(QObject):
    analyze_finished: Signal = Signal(str, object)
    analyze_failed: Signal = Signal(str, object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[AnalyzeWorker] = None
        self._active_tab_id: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start_analyze(
        self,
        tab_id: str,
        adapter: AbsExpAdapter,
        req: AnalyzeRequest,
    ) -> None:
        if self.is_running:
            raise RuntimeError(
                f"Cannot start analyze for tab {tab_id!r}: another analyze is already active"
            )
        logger.debug(
            "AnalyzeRunner.start_analyze: tab_id=%r adapter=%s",
            tab_id,
            type(adapter).__name__,
        )
        self._active_tab_id = tab_id
        worker = AnalyzeWorker(adapter, req)
        worker.analyze_finished.connect(self._on_worker_finished)
        worker.analyze_failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self, result: Any) -> None:
        tab_id = self._active_tab_id or ""
        self._worker = None
        self._active_tab_id = None
        self.analyze_finished.emit(tab_id, result)

    def _on_worker_failed(self, exc: Exception) -> None:
        tab_id = self._active_tab_id or ""
        self._worker = None
        self._active_tab_id = None
        self.analyze_failed.emit(tab_id, exc)


class SaveDataRunner(QObject):
    save_finished: Signal = Signal(str)
    save_failed: Signal = Signal(str, object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[SaveDataWorker] = None
        self._active_tab_id: Optional[str] = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start_save(
        self,
        tab_id: str,
        adapter: AbsExpAdapter,
        req: SaveDataRequest,
    ) -> None:
        if self.is_running:
            raise RuntimeError(
                f"Cannot start save for tab {tab_id!r}: another save is already active"
            )
        logger.debug(
            "SaveDataRunner.start_save: tab_id=%r adapter=%s",
            tab_id,
            type(adapter).__name__,
        )
        self._active_tab_id = tab_id
        worker = SaveDataWorker(adapter, req)
        worker.save_finished.connect(self._on_worker_finished)
        worker.save_failed.connect(self._on_worker_failed)
        self._worker = worker
        worker.start()

    def _on_worker_finished(self) -> None:
        tab_id = self._active_tab_id or ""
        self._worker = None
        self._active_tab_id = None
        self.save_finished.emit(tab_id)

    def _on_worker_failed(self, exc: Exception) -> None:
        tab_id = self._active_tab_id or ""
        self._worker = None
        self._active_tab_id = None
        self.save_failed.emit(tab_id, exc)

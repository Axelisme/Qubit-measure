"""Qt-free background executor backed by :mod:`concurrent.futures`."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from zcu_tools.gui.session.ports import OwnerScheduler

logger = logging.getLogger(__name__)


@dataclass
class _Job:
    worker_done: bool = False
    delivery_done: bool = False
    dispatch_failed: bool = False


class ThreadPoolBackgroundExecutor:
    """Run work off-owner and marshal every terminal callback to ``owner``.

    Short work shares one pool. Long work receives a dedicated single-worker
    executor so it cannot starve short helpers. ``quiesce`` is a terminal close:
    it stops new submissions and waits for both worker completion and owner-loop
    delivery acknowledgement.
    """

    def __init__(
        self,
        owner: OwnerScheduler,
        *,
        max_pool_workers: int | None = None,
    ) -> None:
        self._owner = owner
        self._pool = ThreadPoolExecutor(
            max_workers=max_pool_workers,
            thread_name_prefix="zcu-gui-pool",
        )
        self._condition = threading.Condition()
        self._accepting = True
        self._jobs: dict[Future[Any], _Job] = {}
        self._dedicated: dict[Future[Any], ThreadPoolExecutor] = {}
        self._dispatch_failed = False

    def submit(
        self,
        work: Callable[[], Any],
        *,
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
        run_in_pool: bool = True,
    ) -> None:
        if not callable(work):
            raise TypeError("background work must be callable")

        with self._condition:
            if not self._accepting:
                raise RuntimeError("background executor is quiescing")
            executor = self._pool
            dedicated: ThreadPoolExecutor | None = None
            if not run_in_pool:
                dedicated = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="zcu-gui-dedicated",
                )
                executor = dedicated
            try:
                future = executor.submit(work)
            except BaseException:
                if dedicated is not None:
                    dedicated.shutdown(wait=False, cancel_futures=False)
                raise
            self._jobs[future] = _Job()
            if dedicated is not None:
                self._dedicated[future] = dedicated

        future.add_done_callback(
            lambda completed: self._worker_settled(completed, on_done, on_error)
        )

    def quiesce(self, timeout: float = 5.0) -> bool:
        """Close submissions and wait for workers plus owner delivery acks.

        Calling from the owner thread returns ``False`` once only queued owner
        deliveries remain; blocking there would prevent those deliveries and
        deadlock. The caller may pump its owner loop and call ``quiesce`` again.
        """

        if timeout < 0:
            raise ValueError("quiesce timeout must be non-negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._accepting:
                self._accepting = False
                self._pool.shutdown(wait=False, cancel_futures=False)

            while True:
                if not self._jobs:
                    return not self._dispatch_failed

                workers_done = all(job.worker_done for job in self._jobs.values())
                if workers_done and self._is_owner_thread():
                    return False

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)

    def _worker_settled(
        self,
        future: Future[Any],
        on_done: Callable[[Any], None],
        on_error: Callable[[Exception], None],
    ) -> None:
        terminal_callback: Callable[[], None]
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001 - delivered to owner callback
            logger.error("background worker failed", exc_info=exc)
            terminal_callback = lambda error=exc: on_error(error)
        else:
            terminal_callback = lambda value=result: on_done(value)

        def deliver_and_ack() -> None:
            try:
                terminal_callback()
            except Exception:  # noqa: BLE001 - owner callback boundary
                logger.exception("background terminal callback failed")
            finally:
                self._mark_delivery_done(future)

        dispatch_failed = False
        try:
            self._owner.post(deliver_and_ack)
        except Exception:  # noqa: BLE001 - scheduler boundary
            logger.exception("owner scheduler rejected background delivery")
            dispatch_failed = True

        dedicated: ThreadPoolExecutor | None
        with self._condition:
            job = self._jobs[future]
            job.worker_done = True
            job.dispatch_failed = dispatch_failed
            if dispatch_failed:
                job.delivery_done = True
            dedicated = self._dedicated.pop(future, None)
            self._retire_if_complete(future)
            self._condition.notify_all()
        if dedicated is not None:
            dedicated.shutdown(wait=False, cancel_futures=False)

    def _mark_delivery_done(self, future: Future[Any]) -> None:
        with self._condition:
            self._jobs[future].delivery_done = True
            self._retire_if_complete(future)
            self._condition.notify_all()

    def _retire_if_complete(self, future: Future[Any]) -> None:
        job = self._jobs[future]
        if not (job.worker_done and job.delivery_done):
            return
        self._dispatch_failed = self._dispatch_failed or job.dispatch_failed
        del self._jobs[future]

    def _is_owner_thread(self) -> bool:
        return self._owner.is_owner_thread()


__all__ = ["ThreadPoolBackgroundExecutor"]

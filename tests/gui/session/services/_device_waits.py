"""Controlled driver effects and real Qt completion for device service tests."""

from __future__ import annotations

import threading
from collections.abc import Callable

from qtpy.QtCore import QCoreApplication, QEventLoop, QTimer
from zcu_tools.device.base import BaseDeviceInfo
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.events import (
    DeviceOperationFinishedPayload,
    DeviceSetupFinishedPayload,
)


def drain_until(condition: Callable[[], bool], *, label: str = "completion") -> None:
    """Pump real owner deliveries until condition; the timer only guards deadlock."""
    assert QCoreApplication.instance() is not None
    loop = QEventLoop()
    probe = QTimer()
    probe.timeout.connect(lambda: loop.quit() if condition() else None)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(loop.quit)
    probe.start(0)
    # The caller has scheduled a driver/owner completion that makes condition true.
    watchdog.start(30_000)
    try:
        if not condition():
            loop.exec()
        assert condition(), f"timed out waiting for {label}"
    finally:
        probe.stop()
        watchdog.stop()


class SetupBlock:
    """Driver setup effect: announces entry, holds readback, then returns or raises.

    stop_event is the service's cancellation signal, captured only after entry.
    The owning DeviceWorkers releases every block on teardown before quiescing.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.stop_event: threading.Event | None = None
        self.error = error

    def __call__(
        self, info: BaseDeviceInfo, stop_event: threading.Event | None = None
    ) -> None:
        self.stop_event = stop_event
        self.entered.set()
        # The test (or its finally teardown) releases this admitted driver call.
        assert self.release.wait(30), "driver setup was not released"
        if self.error is not None:
            raise self.error


class DeviceCompletion:
    """Wait for a selected callback or any observed service failure, never forever."""

    def __init__(self, errors: list[str], *, allow_failure: bool = False) -> None:
        self.errors = errors
        self.before = len(errors)
        self.done = False
        self.allow_failure = allow_failure

    def quit(self) -> None:
        """Mark the selected public completion callback as delivered."""
        self.done = True

    def exec(self) -> None:
        """Dispatch owner callbacks; report unexpected worker failure to the caller."""
        drain_until(
            lambda: self.done or len(self.errors) > self.before,
            label="device operation terminal delivery",
        )
        if not self.allow_failure:
            assert self.errors[self.before :] == []


class DeviceWorkers:
    """Own runner lifetimes, service failure observation, and blocked driver cleanup."""

    def __init__(self) -> None:
        self.runners: list[BackgroundRunner] = []
        self.blocks: list[SetupBlock] = []
        self.errors: list[str] = []

    def background(self) -> BackgroundRunner:
        """Create a real Qt executor retained until public quiesce succeeds."""
        bg = BackgroundRunner()
        self.runners.append(bg)
        return bg

    def setup(self, error: Exception | None = None) -> SetupBlock:
        """Create a controlled setup effect owned by this test's teardown."""
        block = SetupBlock(error)
        self.blocks.append(block)
        return block

    def watch(self, bus: BaseEventBus) -> None:
        """Observe failure facts on the caller-injected public event bus."""
        bus.subscribe(DeviceOperationFinishedPayload, self._operation_done)
        bus.subscribe(DeviceSetupFinishedPayload, self._setup_done)

    def _operation_done(self, payload: DeviceOperationFinishedPayload) -> None:
        if not payload.success:
            self.errors.append(payload.error_message or "device operation failed")

    def _setup_done(self, payload: DeviceSetupFinishedPayload) -> None:
        if payload.outcome == "failed":
            self.errors.append(payload.error_message or "device setup failed")

    def completion(self, *, allow_failure: bool = False) -> DeviceCompletion:
        """Observe the next selected terminal, accepting failure only when requested."""
        return DeviceCompletion(self.errors, allow_failure=allow_failure)

    def close(self) -> None:
        """Release every driver before joining workers and queued owner deliveries."""
        for block in self.blocks:
            block.release.set()
        for bg in self.runners:
            # All controlled effects are released; worker + owner ack must drain.
            assert bg.quiesce(timeout_ms=30_000)
        self.blocks.clear()
        self.runners.clear()
        self.errors.clear()

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
import zcu_tools.device.fake as fake_module
from zcu_tools.device import DeviceManager

from ._threads import DeviceThreads


@pytest.fixture
def manager() -> DeviceManager:
    """Create a fresh registry owner for each manager contract test."""
    return DeviceManager()


@pytest.fixture
def workers() -> Iterator[DeviceThreads]:
    callers = DeviceThreads()
    try:
        yield callers
    finally:
        callers.finish()


@pytest.fixture
def paused_ramp(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[threading.Event, threading.Event]]:
    """Pause the external clock after a real ramp step has been applied."""
    entered = threading.Event()
    release = threading.Event()

    def sleep(_seconds: float) -> None:
        entered.set()
        # The caller's finally/fixture teardown releases each paused ramp step.
        if not release.wait(30.0):
            raise TimeoutError("ramp was not released")

    monkeypatch.setattr(fake_module.time, "sleep", sleep)
    try:
        yield entered, release
    finally:
        release.set()

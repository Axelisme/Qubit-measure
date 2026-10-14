from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
import zcu_tools.device.fake as fake_module


@pytest.fixture
def paused_ramp(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[threading.Event, threading.Event]]:
    """Pause the external clock after a real ramp step has been applied."""
    entered = threading.Event()
    release = threading.Event()

    def sleep(_seconds: float) -> None:
        entered.set()
        if not release.wait(5.0):
            raise TimeoutError("ramp was not released")

    monkeypatch.setattr(fake_module.time, "sleep", sleep)
    try:
        yield entered, release
    finally:
        release.set()

"""The production workflow clock preserves UTC and native cancellation."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest
import zcu_tools.experiment.workflows.clock as clock_module
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.experiment.workflows.ports import Clock


@pytest.fixture
def wall_time(monkeypatch: pytest.MonkeyPatch) -> list[datetime]:
    current = [datetime(2026, 1, 1, tzinfo=UTC)]
    monkeypatch.setattr(
        clock_module,
        "datetime",
        SimpleNamespace(now=lambda tz: current[0].astimezone(tz)),
    )
    return current


def test_now_is_current_utc_wall_time(wall_time: list[datetime]) -> None:
    clock: Clock = UtcClock()
    captured = clock.now()
    assert captured.tzinfo is UTC
    assert captured == wall_time[0]


def test_past_target_returns_without_cancelling(wall_time: list[datetime]) -> None:
    signal = StopSignal()
    UtcClock().wait_until(wall_time[0] - timedelta(seconds=1), signal)
    assert not signal.is_set()


def test_future_deadline_in_another_timezone_returns_without_cancelling(
    wall_time: list[datetime],
) -> None:
    clock = UtcClock()
    target = (wall_time[0] + timedelta(seconds=20)).astimezone(
        timezone(timedelta(hours=5, minutes=30))
    )
    timeouts: list[float | None] = []

    class DeadlineEvent(Event):
        def wait(self, timeout: float | None = None) -> bool:
            timeouts.append(timeout)
            wall_time[0] = target.astimezone(UTC)
            return False

    signal = StopSignal(DeadlineEvent())
    clock.wait_until(target, signal)
    assert timeouts == [20.0]
    assert clock.now() == target
    assert not signal.is_set()


def test_future_wait_returns_on_native_cancellation(wall_time: list[datetime]) -> None:
    clock = UtcClock()
    entered = Event()

    class ObservedEvent(Event):
        def wait(self, timeout: float | None = None) -> bool:
            entered.set()  # Arrival at the injected native waiting boundary.
            return super().wait(timeout)

    signal = StopSignal(ObservedEvent())
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            clock.wait_until, wall_time[0] + timedelta(days=1), signal
        )
        try:
            # The scheduled clock caller must reach Event.wait before cancellation.
            assert entered.wait(30.0)
            assert not pending.done()
            signal.set()
            # Native cancellation wakes Event.wait and lets the clock return.
            assert pending.result(timeout=30.0) is None
        finally:
            signal.set()
    assert signal.is_set()


def test_naive_target_fails_at_clock_boundary() -> None:
    signal = StopSignal()
    with pytest.raises(TypeError, match="naive"):
        UtcClock().wait_until(datetime(2000, 1, 1), signal)
    assert not signal.is_set()

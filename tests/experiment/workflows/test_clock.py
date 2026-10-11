"""The production workflow clock preserves UTC and native cancellation."""

from datetime import UTC, datetime, timedelta, timezone
from threading import Event, Thread

import pytest
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.experiment.workflows.clock import UtcClock
from zcu_tools.experiment.workflows.ports import Clock


def test_now_is_current_utc_wall_time() -> None:
    clock: Clock = UtcClock()
    before = datetime.now(UTC)
    captured = clock.now()
    after = datetime.now(UTC)
    assert captured.tzinfo is UTC
    assert before <= captured <= after


def test_past_target_returns_without_cancelling() -> None:
    signal = StopSignal()
    UtcClock().wait_until(datetime(2000, 1, 1, tzinfo=UTC), signal)
    assert not signal.is_set()


def test_future_deadline_in_another_timezone_returns_without_cancelling() -> None:
    clock = UtcClock()
    signal = StopSignal()
    target = (clock.now() + timedelta(milliseconds=20)).astimezone(
        timezone(timedelta(hours=5, minutes=30))
    )
    clock.wait_until(target, signal)
    assert clock.now() >= target
    assert not signal.is_set()


def test_future_wait_returns_on_native_cancellation() -> None:
    clock = UtcClock()
    signal = StopSignal()
    entered, returned = Event(), Event()

    def wait() -> None:
        entered.set()
        clock.wait_until(clock.now() + timedelta(seconds=5), signal)
        returned.set()

    worker = Thread(target=wait)
    worker.start()
    try:
        assert entered.wait(1.0)
        signal.set()
        assert returned.wait(1.0)
    finally:
        signal.set()
        worker.join(6.0)
    assert not worker.is_alive()
    assert signal.is_set()


def test_naive_target_fails_at_clock_boundary() -> None:
    signal = StopSignal()
    with pytest.raises(TypeError, match="naive"):
        UtcClock().wait_until(datetime(2000, 1, 1), signal)
    assert not signal.is_set()

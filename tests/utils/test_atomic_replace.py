"""Retry policy at the public atomic replacement boundary, without real waits."""

import errno
import time
from pathlib import Path

import pytest
import zcu_tools.utils as utils


class _WindowsError(OSError):
    def __init__(self, winerror: int) -> None:
        super().__init__(errno.EACCES, "synthetic Windows file error")
        self.winerror = winerror


class _Clock:
    def __init__(self) -> None:
        self.elapsed = 0.0
        self.waits: list[float] = []

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.elapsed += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    return clock


@pytest.mark.parametrize("winerror", [32, 33])
def test_transient_windows_lock_publishes_the_complete_file(
    tmp_path: Path, clock: _Clock, winerror: int
) -> None:
    assert callable(getattr(utils, "atomic_replace", None))
    source, destination = tmp_path / "pending", tmp_path / "saved"
    source.write_bytes(b"new complete bytes")
    destination.write_bytes(b"original")
    attempts = 0

    def replace_file(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            assert destination.read_bytes() == b"original"
            raise _WindowsError(winerror)
        source.replace(destination)

    utils.atomic_replace(source, destination, replace_fn=replace_file)

    assert destination.read_bytes() == b"new complete bytes"
    assert not source.exists()
    assert attempts == 3
    assert 0 < clock.elapsed < 0.5


@pytest.mark.parametrize("winerror", [32, 33])
def test_persistent_windows_lock_preserves_error_and_stops_near_half_a_second(
    tmp_path: Path, clock: _Clock, winerror: int
) -> None:
    assert callable(getattr(utils, "atomic_replace", None))
    source, destination = tmp_path / "pending", tmp_path / "saved"
    source.write_bytes(b"pending")
    destination.write_bytes(b"original")
    expected = _WindowsError(winerror)
    attempts: list[float] = []

    def replace_file(source: Path, destination: Path) -> None:
        attempts.append(clock.elapsed)
        raise expected

    with pytest.raises(OSError) as raised:
        utils.atomic_replace(source, destination, replace_fn=replace_file)

    assert raised.value is expected
    assert len(attempts) > 1
    assert 0.45 <= clock.elapsed <= 0.55
    assert max(attempts) <= 0.5
    assert source.read_bytes() == b"pending"
    assert destination.read_bytes() == b"original"


@pytest.mark.parametrize(
    "kind", ["posix-permission", "windows-access", "missing", "runtime"]
)
def test_non_lock_errors_propagate_once_without_waiting(
    tmp_path: Path, clock: _Clock, kind: str
) -> None:
    assert callable(getattr(utils, "atomic_replace", None))
    errors = {
        "posix-permission": PermissionError(errno.EACCES, "permission denied"),
        "windows-access": _WindowsError(5),
        "missing": FileNotFoundError(errno.ENOENT, "source missing"),
        "runtime": RuntimeError("replace provider failed"),
    }
    expected = errors[kind]
    attempts = 0

    def replace_file(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise expected

    with pytest.raises(type(expected)) as raised:
        utils.atomic_replace(
            tmp_path / "pending", tmp_path / "saved", replace_fn=replace_file
        )

    assert raised.value is expected
    assert attempts == 1
    assert clock.waits == []


def test_successful_first_attempt_does_not_wait(tmp_path: Path, clock: _Clock) -> None:
    assert callable(getattr(utils, "atomic_replace", None))
    source, destination = tmp_path / "pending", tmp_path / "saved"
    source.write_bytes(b"complete")

    utils.atomic_replace(source, destination)

    assert destination.read_bytes() == b"complete"
    assert not source.exists()
    assert clock.waits == []


def test_replace_time_counts_towards_the_retry_deadline(
    tmp_path: Path, clock: _Clock
) -> None:
    assert callable(getattr(utils, "atomic_replace", None))
    expected = _WindowsError(32)
    attempts = 0

    def replace_file(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        clock.elapsed += 0.6
        raise expected

    with pytest.raises(OSError) as raised:
        utils.atomic_replace(
            tmp_path / "pending", tmp_path / "saved", replace_fn=replace_file
        )

    assert raised.value is expected
    assert attempts == 1
    assert clock.waits == []


def test_late_wakeup_does_not_start_another_replace(
    tmp_path: Path, clock: _Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert callable(getattr(utils, "atomic_replace", None))
    expected = _WindowsError(33)
    attempts = 0

    def late_wakeup(seconds: float) -> None:
        clock.elapsed += seconds + 0.6

    def replace_file(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise expected

    monkeypatch.setattr(time, "sleep", late_wakeup)
    with pytest.raises(OSError) as raised:
        utils.atomic_replace(
            tmp_path / "pending", tmp_path / "saved", replace_fn=replace_file
        )

    assert raised.value is expected
    assert attempts == 1


def test_non_lock_failure_after_a_lock_is_not_retried(
    tmp_path: Path, clock: _Clock
) -> None:
    assert callable(getattr(utils, "atomic_replace", None))
    expected = _WindowsError(5)
    attempts = 0

    def replace_file(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _WindowsError(32)
        raise expected

    with pytest.raises(OSError) as raised:
        utils.atomic_replace(
            tmp_path / "pending", tmp_path / "saved", replace_fn=replace_file
        )

    assert raised.value is expected
    assert attempts == 2
    assert 0 < clock.elapsed < 0.5

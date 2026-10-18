"""Concurrency tests for manager's registry-scoped lock.

The manager _lock must protect only the registry dict (register/drop/get).
Long I/O operations — setup() ramps, get_info() SCPI reads — must run outside
that lock so that an in-progress ramp on device A cannot block a get_info()
call on the independent device B.

These tests use FakeDevice so no real VISA session is required. Driver gates
control I/O and cleanup. The finish-boundary hammer samples a race with no
public arrival point; its terminal close-once assertions are deterministic.
"""

from __future__ import annotations

import threading
from typing import Any, Literal, cast

import pytest
from zcu_tools.device import (
    DeviceCloseFailure,
    DeviceCloseInProgressError,
    DeviceManager,
    FakeDevice,
    FakeDeviceInfo,
)

from ._threads import DeviceThreads


@pytest.fixture
def manager() -> DeviceManager:
    return DeviceManager()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fast_device() -> FakeDevice:
    """FakeDevice that completes setup instantly (for the observer device B)."""
    return FakeDevice(fast_mode=True)


class _CloseProbe:
    """Deterministic handle for a hooked ``device.close()``."""

    def __init__(
        self,
        calls: list[int],
        entered: threading.Event,
        release: threading.Event,
    ) -> None:
        self.calls = calls
        self.entered = entered
        self.release = release


def _hook_close(
    device: FakeDevice,
    *,
    fail: Literal["none", "first", "always"] = "none",
    base_fail: bool = False,
    fail_after_release: Literal["none", "ordinary", "base"] = "none",
) -> _CloseProbe:
    """Replace ``device.close`` with a deterministic probe.

    The probe records every invocation.  With ``base_fail`` it raises
    ``KeyboardInterrupt`` immediately; with ``fail`` it raises ``RuntimeError``
    immediately on the configured ordinary-failure mode.  Otherwise it
    signals ``entered`` (the manager now holds the in-flight claim), blocks
    on ``release``, and then either raises the ``fail_after_release`` failure
    mode or runs the original close.  Events (not sleeps) keep the
    concurrent tests deterministic regardless of scheduler timing.
    """
    calls: list[int] = []
    entered = threading.Event()
    release = threading.Event()
    original = device.close

    def hooked() -> None:
        calls.append(1)
        if base_fail:
            raise KeyboardInterrupt("probe base abort")
        if fail == "first" and len(calls) == 1:
            raise RuntimeError("probe close failure")
        if fail == "always":
            raise RuntimeError("probe close failure")
        entered.set()
        # Test body or worker teardown releases the in-flight close.
        if not release.wait(30.0):
            raise TimeoutError("close was not released")
        if fail_after_release == "ordinary":
            raise RuntimeError("probe close failure after release")
        if fail_after_release == "base":
            raise KeyboardInterrupt("probe base abort after release")
        original()

    device.close = hooked
    return _CloseProbe(calls, entered, release)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_info_not_blocked_by_concurrent_setup_on_other_device(
    manager: DeviceManager,
    paused_ramp: tuple[threading.Event, threading.Event],
    workers: DeviceThreads,
) -> None:
    dev_a = FakeDevice(fast_mode=False)
    dev_b = _make_fast_device()
    manager.register_device("A", dev_a)
    manager.register_device("B", dev_b)
    cfg_a = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.25)
    cfg_b = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    manager.setup_devices({"B": cfg_b}, progress=False)
    entered, release = paused_ramp
    worker = workers.thread(
        release=release,
        target=lambda: manager.setup_devices({"A": cfg_a}, progress=False),
    )

    worker.start()
    try:
        workers.arrived(entered)
        info_b = manager.get_info("B")
        assert isinstance(info_b, FakeDeviceInfo)
        assert info_b.value == cfg_b.value
        assert worker.is_alive(), "get_info('B') waited for A's ramp to finish"
    finally:
        release.set()
        workers.join(worker)

    info_a = manager.get_info("A")
    assert isinstance(info_a, FakeDeviceInfo)
    assert info_a.value == cfg_a.value


def test_register_device_rejects_non_base_device_without_mutating_registry(
    manager: DeviceManager,
) -> None:
    # tolerated(cast): Test registry rejection of a non-BaseDevice through its typed entry.
    bad_device = cast(Any, object())

    with pytest.raises(TypeError, match="register_device expected BaseDevice"):
        manager.register_device("bad", bad_device)

    assert "bad" not in manager.get_all_devices()


@pytest.mark.parametrize("second_name_exists", [False, True])
def test_register_device_rejects_second_name_without_mutating_registry(
    manager: DeviceManager, second_name_exists: bool
) -> None:
    device = _make_fast_device()
    manager.register_device("flux", device)
    if second_name_exists:
        manager.register_device("other", _make_fast_device())
    before = manager.get_all_devices()

    with pytest.raises(ValueError, match="already registered"):
        manager.register_device("other", device)

    assert manager.get_all_devices() == before
    assert manager.get_device("flux") is device


@pytest.mark.parametrize("replace_driver", [False, True])
def test_register_device_same_name_preserves_overwrite_contract(
    manager: DeviceManager, replace_driver: bool
) -> None:
    original = _make_fast_device()
    manager.register_device("flux", original)
    replacement = _make_fast_device() if replace_driver else original

    with pytest.warns(UserWarning, match="already registered, overwriting"):
        manager.register_device("flux", replacement)

    assert manager.get_all_devices() == {"flux": replacement}


def test_managers_isolate_names_setup_and_disconnect(manager: DeviceManager) -> None:
    other = DeviceManager()
    first = _make_fast_device()
    second = _make_fast_device()
    manager.register_device("flux", first)
    other.register_device("flux", second)

    manager.setup_devices(
        {"flux": FakeDeviceInfo(address="none", output="on", value=0.5)},
        progress=False,
    )
    assert manager.get_device("flux") is first
    assert other.get_device("flux") is second
    assert second.get_value() == 0.0

    manager.close_all_devices()
    assert manager.get_all_devices() == {}
    assert other.get_device("flux") is second
    assert second.get_value() == 0.0


def test_get_all_devices_is_an_independent_mapping(manager: DeviceManager) -> None:
    device = _make_fast_device()
    manager.register_device("flux", device)
    snapshot = manager.get_all_devices()
    snapshot.clear()
    assert manager.get_device("flux") is device


def test_register_device_accepts_base_device(manager: DeviceManager) -> None:
    dev = _make_fast_device()

    manager.register_device("ok", dev)

    assert manager.get_device("ok") is dev


def test_get_all_info_returns_correct_snapshot(manager: DeviceManager) -> None:
    """get_all_info() returns the current state of all registered devices.

    This is a functional regression check: the new out-of-lock implementation
    must still return accurate info for every device in the registry.
    """
    dev_a = _make_fast_device()
    dev_b = _make_fast_device()
    manager.register_device("A", dev_a)
    manager.register_device("B", dev_b)

    cfg_a = FakeDeviceInfo(address="none", output="on", value=0.3, rampstep=0.1)
    cfg_b = FakeDeviceInfo(address="none", output="off", value=0.7, rampstep=0.1)
    manager.setup_devices({"A": cfg_a, "B": cfg_b})

    all_info = manager.get_all_info()

    assert set(all_info.keys()) == {"A", "B"}

    # Narrow to FakeDeviceInfo via isinstance so pyright sees the right type.
    info_a = all_info["A"]
    info_b = all_info["B"]
    assert isinstance(info_a, FakeDeviceInfo)
    assert isinstance(info_b, FakeDeviceInfo)
    assert info_a.value == pytest.approx(0.3)
    assert info_b.value == pytest.approx(0.7)
    assert info_a.output == "on"
    assert info_b.output == "off"


def test_setup_devices_fast_fails_on_unknown_name(manager: DeviceManager) -> None:
    """Unknown device name in setup_devices raises ValueError before any setup."""
    dev = _make_fast_device()
    manager.register_device("known", dev)

    cfg = FakeDeviceInfo(address="none", output="on", value=0.0, rampstep=0.1)
    with pytest.raises(ValueError, match="unknown"):
        manager.setup_devices({"known": cfg, "unknown": cfg})


def test_setup_devices_validates_all_names_before_any_setup(
    manager: DeviceManager,
) -> None:
    """If any name is missing the whole batch is rejected — no partial setup."""
    dev_a = _make_fast_device()
    manager.register_device("A", dev_a)
    # "B" is not registered.

    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)

    with pytest.raises(ValueError, match="B"):
        manager.setup_devices({"A": cfg, "B": cfg})

    # A must not have been set up (value stays at the default 0.0).
    assert dev_a.get_value() == 0.0, (
        "Device A was set up despite the batch being rejected; "
        "fast-fail before any I/O is broken."
    )


def test_setup_devices_validates_names_before_cancel_short_circuit(
    manager: DeviceManager,
) -> None:
    dev = _make_fast_device()
    manager.register_device("known", dev)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    cancel_signal.set()

    with pytest.raises(ValueError, match="unknown"):
        manager.setup_devices(
            {"known": cfg, "unknown": cfg},
            cancel_signal=cancel_signal,
        )

    assert dev.get_output() == "off"
    assert dev.get_value() == 0.0


def test_setup_devices_explicit_cancel_signal_skips_before_first_device(
    manager: DeviceManager,
) -> None:
    dev = _make_fast_device()
    manager.register_device("A", dev)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    cancel_signal.set()

    manager.setup_devices({"A": cfg}, cancel_signal=cancel_signal)

    assert dev.get_output() == "off"
    assert dev.get_value() == 0.0


def test_setup_devices_mid_ramp_cancel_stops_before_next_device(
    manager: DeviceManager,
    paused_ramp: tuple[threading.Event, threading.Event],
    workers: DeviceThreads,
) -> None:
    dev_a = FakeDevice(fast_mode=False)
    dev_b = _make_fast_device()
    manager.register_device("A", dev_a)
    manager.register_device("B", dev_b)
    cfg_a = FakeDeviceInfo(address="none", output="on", value=1.0, rampstep=0.01)
    cfg_b = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    entered, release = paused_ramp
    worker = workers.thread(
        target=lambda: manager.setup_devices(
            {"A": cfg_a, "B": cfg_b},
            cancel_signal=cancel_signal,
        ),
        release=release,
    )
    worker.start()
    try:
        workers.arrived(entered)
        cancel_signal.set()
    finally:
        release.set()
        workers.join(worker)

    assert cancel_signal.is_set()
    assert dev_a.get_output() == "on"
    assert 0.0 < dev_a.get_value() < 1.0
    assert dev_b.get_output() == "off"
    assert dev_b.get_value() == 0.0


# ---------------------------------------------------------------------------
# Registry-owned disconnect: close_device / close_all_devices
# ---------------------------------------------------------------------------


def test_close_device_missing_raises_value_error(manager: DeviceManager) -> None:
    with pytest.raises(ValueError, match="not found"):
        manager.close_device("ghost")

    assert "ghost" not in manager.get_all_devices()


def test_close_device_ignore_missing_is_noop(manager: DeviceManager) -> None:
    manager.close_device("ghost", ignore_missing=True)


def test_close_device_failure_keeps_entries_then_retry_succeeds(
    manager: DeviceManager,
) -> None:
    dev = _make_fast_device()
    probe = _hook_close(dev, fail="first")
    manager.register_device("A", dev)

    with pytest.raises(DeviceCloseFailure) as excinfo:
        manager.close_device("A")

    failure = excinfo.value
    assert failure.names == ("A",)
    assert isinstance(failure.cause, RuntimeError)
    # Failure keeps the entry retryable.
    assert manager.get_all_devices() == {"A": dev}

    # The claim was released on failure: the retry reaches the device again.
    probe.release.set()
    manager.close_device("A")

    assert len(probe.calls) == 2
    assert manager.get_all_devices() == {}


def test_close_all_empty_registry_is_noop(manager: DeviceManager) -> None:
    manager.close_all_devices()


def test_close_all_aggregates_named_failures_and_keeps_entries(
    manager: DeviceManager,
) -> None:
    dev1 = _make_fast_device()
    dev2 = _make_fast_device()
    _hook_close(dev1, fail="always")
    _hook_close(dev2, fail="always")
    manager.register_device("A", dev1)
    manager.register_device("B", dev2)

    with pytest.raises(ExceptionGroup) as excinfo:
        manager.close_all_devices()

    failures = [
        e for e in excinfo.value.exceptions if isinstance(e, DeviceCloseFailure)
    ]
    assert len(failures) == 2
    name_sets = {tuple(sorted(e.names)) for e in failures}
    assert name_sets == {("A",), ("B",)}
    assert all(isinstance(e.cause, RuntimeError) for e in failures)

    # Entries stay registered for retry; claims were released (a second
    # close_device call reaches the device instead of raising InProgressError).
    assert set(manager.get_all_devices()) == {"A", "B"}
    with pytest.raises(DeviceCloseFailure):
        manager.close_device("A")


def test_close_all_continues_after_failure_and_removes_successes(
    manager: DeviceManager,
) -> None:
    dev1 = _make_fast_device()
    dev2 = _make_fast_device()
    _hook_close(dev1, fail="always")
    probe2 = _hook_close(dev2)
    probe2.release.set()
    manager.register_device("A", dev1)
    manager.register_device("B", dev2)

    with pytest.raises(ExceptionGroup) as excinfo:
        manager.close_all_devices()

    failures = [
        e for e in excinfo.value.exceptions if isinstance(e, DeviceCloseFailure)
    ]
    assert [e.names for e in failures] == [("A",)]
    assert "A" in manager.get_all_devices()
    assert "B" not in manager.get_all_devices()
    assert len(probe2.calls) == 1


@pytest.mark.parametrize("registry_change", ["keep", "drop", "overwrite"])
@pytest.mark.parametrize("close_all", [False, True])
def test_register_device_rejects_closing_driver_without_mutating_registry(
    manager: DeviceManager,
    workers: DeviceThreads,
    registry_change: str,
    close_all: bool,
) -> None:
    device = _make_fast_device()
    probe = _hook_close(device)
    manager.register_device("flux", device)

    def closer() -> None:
        if close_all:
            manager.close_all_devices()
        else:
            manager.close_device("flux")

    thread = workers.thread(release=probe.release, target=closer)
    thread.start()
    workers.arrived(probe.entered, "close did not enter device.close()")

    if registry_change == "drop":
        manager.drop_device("flux")
    elif registry_change == "overwrite":
        with pytest.warns(UserWarning, match="overwriting"):
            manager.register_device("flux", _make_fast_device())
    before = manager.get_all_devices()
    # Also reject the original name while it still refers to the driver.
    requested_name = "flux" if registry_change == "keep" else "new"
    with pytest.raises(DeviceCloseInProgressError):
        manager.register_device(requested_name, device)
    assert manager.get_all_devices() == before

    probe.release.set()
    workers.join(thread)
    assert len(probe.calls) == 1
    manager.register_device("new", device)
    assert manager.get_device("new") is device


def test_concurrent_same_identity_close_device_follower_fast_fails(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    """Second manager close of an in-flight identity fails fast, close once."""
    dev = _make_fast_device()
    probe = _hook_close(dev)
    manager.register_device("A", dev)

    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            manager.close_device("A")
        except BaseException as exc:  # pragma: no cover - assertion below
            leader_errors.append(exc)

    t_leader = workers.thread(release=probe.release, target=leader)
    t_leader.start()
    workers.arrived(probe.entered, "leader did not enter device.close()")

    with pytest.raises(DeviceCloseInProgressError):
        manager.close_device("A")

    probe.release.set()
    workers.join(t_leader)

    assert leader_errors == []
    assert len(probe.calls) == 1  # the follower never called device.close()
    assert manager.get_all_devices() == {}

    # Success released the claim: closing the same identity again (after a
    # re-registration) works instead of raising DeviceCloseInProgressError.
    manager.register_device("A", dev)
    manager.close_device("A")
    assert len(probe.calls) == 2
    assert manager.get_all_devices() == {}


def test_close_device_success_never_leaves_reclaimable_window(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    """A follower can never re-claim an identity at the finish boundary.

    Regression: a successful close released the claim and removed stale
    aliases in separate lock acquisitions, leaving a window in which a
    follower could reclaim the still-registered identity and call
    ``device.close()`` a second time.  The finish must be atomic -- followers
    observe either the in-flight claim (``DeviceCloseInProgressError``) or a
    fully cleaned registry, never a released claim whose identity is still
    registered.
    """
    dev = _make_fast_device()
    probe = _hook_close(dev)
    manager.register_device("A", dev)

    leader_done = threading.Event()
    follower_observations: list[str] = []

    def leader() -> None:
        try:
            manager.close_device("A")
        finally:
            leader_done.set()

    def follower() -> None:
        # No public finish-arrival seam; sampling only affects bug detection.
        # Hammer the finish boundary: every attempt must see either the
        # in-flight claim or the cleaned-up registry, never a re-claimable
        # still-registered identity after the claim is dropped.
        while not leader_done.is_set():
            try:
                manager.close_device("A")
            except DeviceCloseInProgressError:
                continue
            except ValueError:
                follower_observations.append("gone")
                return
            else:
                follower_observations.append("reclaimed")
                return
        # Leader already finished: the alias must be gone by then.
        try:
            manager.close_device("A")
        except ValueError:
            follower_observations.append("gone")
        else:
            follower_observations.append("reclaimed")

    t_leader = workers.thread(release=probe.release, target=leader)
    t_leader.start()
    workers.arrived(probe.entered, "leader did not enter device.close()")
    with pytest.raises(DeviceCloseInProgressError):
        manager.close_device("A")
    t_follower = workers.thread(release=probe.release, target=follower)
    t_follower.start()
    probe.release.set()
    workers.join(t_leader)
    workers.join(t_follower)
    assert follower_observations == ["gone"], (
        f"follower observed {follower_observations!r}; a reclaimed identity "
        "would mean the already-closed device was closed again"
    )
    assert len(probe.calls) == 1  # close-once identity guarantee
    assert manager.get_all_devices() == {}


def test_close_device_follower_fast_fails_while_leader_fails_ordinarily(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    """Follower fails fast while an in-flight close ends in ordinary failure.

    A3: the claim is held while the leader's close is in flight, so a
    follower raises ``DeviceCloseInProgressError`` and never calls
    ``device.close()``; after the leader's ``DeviceCloseFailure``, the
    released claim lets a retry reach the device again.
    """
    dev = _make_fast_device()
    probe = _hook_close(dev, fail_after_release="ordinary")
    manager.register_device("A", dev)

    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            manager.close_device("A")
        except BaseException as exc:  # pragma: no cover - asserted below
            leader_errors.append(exc)

    t_leader = workers.thread(release=probe.release, target=leader)
    t_leader.start()
    workers.arrived(probe.entered, "leader did not enter device.close()")

    with pytest.raises(DeviceCloseInProgressError):
        manager.close_device("A")
    assert len(probe.calls) == 1  # follower never reached the device

    probe.release.set()
    workers.join(t_leader)

    assert len(leader_errors) == 1
    assert isinstance(leader_errors[0], DeviceCloseFailure)
    assert "A" in manager.get_all_devices()  # entry stays retryable

    # Failure released the claim: the retry re-claims and reaches the device.
    with pytest.raises(DeviceCloseFailure):
        manager.close_device("A")
    assert len(probe.calls) == 2


def test_close_device_follower_fast_fails_while_leader_aborts(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    """Follower fails fast while an in-flight close ends in BaseException.

    A3: the claim is held across the ``BaseException`` too, so the follower
    raises ``DeviceCloseInProgressError``; the leader's ``KeyboardInterrupt``
    propagates unwrapped, and the released claim lets a retry reach the
    device instead of raising ``DeviceCloseInProgressError``.
    """
    dev = _make_fast_device()
    probe = _hook_close(dev, fail_after_release="base")
    manager.register_device("A", dev)

    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            manager.close_device("A")
        except BaseException as exc:  # pragma: no cover - asserted below
            leader_errors.append(exc)

    t_leader = workers.thread(release=probe.release, target=leader)
    t_leader.start()
    workers.arrived(probe.entered, "leader did not enter device.close()")

    with pytest.raises(DeviceCloseInProgressError):
        manager.close_device("A")
    assert len(probe.calls) == 1  # follower never reached the device

    probe.release.set()
    workers.join(t_leader)

    assert len(leader_errors) == 1
    assert isinstance(leader_errors[0], KeyboardInterrupt)  # propagates unwrapped
    assert "A" in manager.get_all_devices()

    # BaseException released the claim: the retry re-claims and reaches the
    # device instead of raising DeviceCloseInProgressError.
    with pytest.raises(KeyboardInterrupt):
        manager.close_device("A")
    assert len(probe.calls) == 2


def test_close_device_base_exception_releases_claim_and_propagates(
    manager: DeviceManager,
) -> None:
    dev = _make_fast_device()
    probe = _hook_close(dev, base_fail=True)
    manager.register_device("A", dev)

    with pytest.raises(KeyboardInterrupt):
        manager.close_device("A")

    assert "A" in manager.get_all_devices()
    # BaseException propagates unwrapped, but the claim is released: a second
    # call re-claims and reaches the device instead of InProgressError.
    with pytest.raises(KeyboardInterrupt):
        manager.close_device("A")

    assert len(probe.calls) == 2
    assert "A" in manager.get_all_devices()


def test_close_all_reports_in_progress_identity_and_closes_rest(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    """Follower close_all fail-fasts the in-flight identity, closes the rest.

    D6: in-progress identities are not silently skipped -- the batch records
    them as ``DeviceCloseInProgressError`` and aggregates them in the
    ``ExceptionGroup`` while still closing every claimable identity.
    """
    dev1 = _make_fast_device()
    dev2 = _make_fast_device()
    probe1 = _hook_close(dev1)
    probe2 = _hook_close(dev2)
    probe2.release.set()
    manager.register_device("A", dev1)
    manager.register_device("B", dev2)

    def leader() -> None:
        manager.close_device("A")

    t_leader = workers.thread(release=probe1.release, target=leader)
    t_leader.start()
    workers.arrived(probe1.entered, "leader did not enter device.close()")

    # Follower close_all: A is in progress, B is claimable and gets closed.
    with pytest.raises(ExceptionGroup) as excinfo:
        manager.close_all_devices()

    in_progress = [
        e for e in excinfo.value.exceptions if isinstance(e, DeviceCloseInProgressError)
    ]
    assert [e.names for e in in_progress] == [("A",)]
    assert len(probe2.calls) == 1
    assert "B" not in manager.get_all_devices()

    probe1.release.set()
    workers.join(t_leader)

    assert len(probe1.calls) == 1  # A's identity was closed exactly once
    assert "A" not in manager.get_all_devices()


def test_close_all_follower_batch_fast_fails_all_claimed_identities(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    """Concurrent close_all: the follower batch fail-fasts every identity.

    A3: while a leader close_all holds the claims, a follower close_all
    snapshots the same identities, records ``DeviceCloseInProgressError`` for
    each, closes nothing, and raises them as an ``ExceptionGroup``; the
    leader closes each identity exactly once.
    """
    dev1 = _make_fast_device()
    dev2 = _make_fast_device()
    probe1 = _hook_close(dev1)
    probe2 = _hook_close(dev2)
    manager.register_device("A", dev1)
    manager.register_device("B", dev2)

    def leader() -> None:
        manager.close_all_devices()

    t_leader = workers.thread(release=probe1.release, target=leader)
    workers.releases.append(probe2.release)
    t_leader.start()
    workers.arrived(probe1.entered, "leader did not enter the first close()")

    with pytest.raises(ExceptionGroup) as excinfo:
        manager.close_all_devices()

    in_progress = [
        e for e in excinfo.value.exceptions if isinstance(e, DeviceCloseInProgressError)
    ]
    assert {tuple(sorted(e.names)) for e in in_progress} == {("A",), ("B",)}
    assert len(probe1.calls) == 1
    assert len(probe2.calls) == 0  # follower closed nothing

    probe1.release.set()
    probe2.release.set()
    workers.join(t_leader)

    assert len(probe1.calls) == 1
    assert len(probe2.calls) == 1
    assert manager.get_all_devices() == {}


def test_close_all_follower_close_device_fast_fails_while_leader_fails_ordinarily(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    """close_device follower fails fast while a close_all leader fails.

    A3 (batch side): the claim is held for the whole batch close, so a
    close_device follower raises ``DeviceCloseInProgressError`` and never
    reaches the device; after the leader's aggregated ``DeviceCloseFailure``
    the released claim lets a retry reach the device.
    """
    dev = _make_fast_device()
    probe = _hook_close(dev, fail_after_release="ordinary")
    manager.register_device("A", dev)

    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            manager.close_all_devices()
        except BaseException as exc:  # pragma: no cover - asserted below
            leader_errors.append(exc)

    t_leader = workers.thread(release=probe.release, target=leader)
    t_leader.start()
    workers.arrived(probe.entered, "leader did not enter device.close()")

    with pytest.raises(DeviceCloseInProgressError):
        manager.close_device("A")
    assert len(probe.calls) == 1  # follower never reached the device

    probe.release.set()
    workers.join(t_leader)

    assert len(leader_errors) == 1
    assert isinstance(leader_errors[0], ExceptionGroup)
    assert "A" in manager.get_all_devices()  # entry stays retryable

    with pytest.raises(DeviceCloseFailure):
        manager.close_device("A")
    assert len(probe.calls) == 2  # retry re-claimed and reached the device


def test_close_all_base_exception_cleans_successes_and_releases_claims(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    """A BaseException in the batch still cleans up earlier successes.

    A3/D6: when a later device raises ``BaseException``, the batch propagates
    it unwrapped but first atomically drops entries already closed
    successfully and releases every owned claim.  A follower
    fast-fails while the batch is in flight; after the abort the success is
    gone and the interrupted identity stays retryable.
    """
    good = _make_fast_device()
    bad = _make_fast_device()
    probe_good = _hook_close(good)
    probe_bad = _hook_close(bad, fail_after_release="base")
    probe_good.release.set()  # good closes as soon as the batch reaches it
    manager.register_device("good", good)
    manager.register_device("bad", bad)

    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            manager.close_all_devices()
        except BaseException as exc:  # pragma: no cover - asserted below
            leader_errors.append(exc)

    t_leader = workers.thread(release=probe_bad.release, target=leader)
    t_leader.start()
    workers.arrived(probe_bad.entered, "leader did not enter the bad close()")
    assert len(probe_good.calls) == 1  # good was already closed

    with pytest.raises(DeviceCloseInProgressError):
        manager.close_device("bad")
    assert len(probe_bad.calls) == 1  # follower never reached the device

    probe_bad.release.set()
    workers.join(t_leader)

    assert len(leader_errors) == 1
    assert isinstance(leader_errors[0], KeyboardInterrupt)  # propagates unwrapped
    devices = manager.get_all_devices()
    assert "good" not in devices  # success cleaned up despite the abort
    assert devices["bad"] is bad  # interrupted identity stays retryable

    # BaseException released the claim: the retry re-claims and reaches the
    # device instead of raising DeviceCloseInProgressError.
    with pytest.raises(KeyboardInterrupt):
        manager.close_device("bad")
    assert len(probe_bad.calls) == 2


def test_close_device_io_not_under_registry_lock(
    manager: DeviceManager, workers: DeviceThreads
) -> None:
    dev_a = _make_fast_device()
    dev_b = _make_fast_device()
    probe = _hook_close(dev_a)
    manager.register_device("A", dev_a)
    manager.register_device("B", dev_b)

    def closer() -> None:
        manager.close_device("A")

    t_closer = workers.thread(release=probe.release, target=closer)
    t_closer.start()
    workers.arrived(probe.entered, "close did not enter device.close()")

    probe_done = threading.Event()
    probe_result: list[object] = []

    def reader() -> None:
        probe_result.append(manager.get_device("B"))
        probe_done.set()

    t_reader = workers.thread(release=probe.release, target=reader)
    t_reader.start()
    workers.arrived(
        probe_done,
        (
            "get_device('B') blocked while close_device I/O was in flight; "
            "the registry lock is held across device.close()."
        ),
    )
    assert probe_result[0] is dev_b

    probe.release.set()
    workers.join(t_closer)


def test_close_all_io_not_under_registry_lock(
    manager: DeviceManager, workers: DeviceThreads
) -> None:
    dev_a = _make_fast_device()
    dev_b = _make_fast_device()
    probe = _hook_close(dev_a)
    manager.register_device("A", dev_a)
    manager.register_device("B", dev_b)

    def closer() -> None:
        manager.close_all_devices()

    t_closer = workers.thread(release=probe.release, target=closer)
    t_closer.start()
    workers.arrived(probe.entered, "close did not enter device.close()")

    probe_done = threading.Event()
    probe_result: list[object] = []

    def reader() -> None:
        probe_result.append(manager.get_device("B"))
        probe_done.set()

    t_reader = workers.thread(release=probe.release, target=reader)
    t_reader.start()
    workers.arrived(
        probe_done,
        (
            "get_device('B') blocked while close_all I/O was in flight; "
            "the registry lock is held across device.close()."
        ),
    )
    assert probe_result[0] is dev_b

    probe.release.set()
    workers.join(t_closer)
    assert manager.get_all_devices() == {}


@pytest.mark.parametrize("close_all", [False, True])
def test_close_success_keeps_replacement_and_new_devices(
    manager: DeviceManager,
    workers: DeviceThreads,
    close_all: bool,
) -> None:
    dev1 = _make_fast_device()
    probe1 = _hook_close(dev1)
    manager.register_device("A", dev1)

    def closer() -> None:
        if close_all:
            manager.close_all_devices()
        else:
            manager.close_device("A")

    t_closer = workers.thread(release=probe1.release, target=closer)
    t_closer.start()
    workers.arrived(probe1.entered, "close did not enter device.close()")

    # A same-name replacement and a distinct new device appear while the
    # close is in flight.
    dev2 = _make_fast_device()
    dev3 = _make_fast_device()
    probe2 = _hook_close(dev2)
    probe3 = _hook_close(dev3)
    probe2.release.set()
    probe3.release.set()
    manager.register_device("A", dev2)  # overwrite warning expected
    manager.register_device("B", dev3)

    probe1.release.set()
    workers.join(t_closer)

    assert len(probe1.calls) == 1
    assert len(probe2.calls) == 0  # replacement was not closed
    assert len(probe3.calls) == 0  # new device was not closed
    devices = manager.get_all_devices()
    assert devices["A"] is dev2  # same-name replacement survives
    assert devices["B"] is dev3  # distinct new device survives


def test_close_all_snapshot_excludes_devices_registered_during_close(
    manager: DeviceManager,
    workers: DeviceThreads,
) -> None:
    dev1 = _make_fast_device()
    probe1 = _hook_close(dev1)
    manager.register_device("A", dev1)

    def closer() -> None:
        manager.close_all_devices()

    t_closer = workers.thread(release=probe1.release, target=closer)
    t_closer.start()
    workers.arrived(probe1.entered, "close did not enter device.close()")

    dev2 = _make_fast_device()
    probe2 = _hook_close(dev2)
    probe2.release.set()
    manager.register_device("B", dev2)

    probe1.release.set()
    workers.join(t_closer)

    assert len(probe1.calls) == 1
    assert len(probe2.calls) == 0  # not part of the batch snapshot
    devices = manager.get_all_devices()
    assert "A" not in devices
    assert devices["B"] is dev2

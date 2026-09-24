"""Concurrency tests for GlobalDeviceManager's registry-scoped lock.

The manager _lock must protect only the registry dict (register/drop/get).
Long I/O operations — setup() ramps, get_info() SCPI reads — must run outside
that lock so that an in-progress ramp on device A cannot block a get_info()
call on the independent device B.

These tests use FakeDevice so no real VISA session is required.  Event objects
(not sleep) are used for thread synchronisation — the test is deterministic
regardless of scheduler timing.
"""

from __future__ import annotations

import threading
from typing import Any, Literal, cast

import pytest
from zcu_tools.device import (
    DeviceCloseFailure,
    DeviceCloseInProgressError,
    FakeDevice,
    FakeDeviceInfo,
    GlobalDeviceManager,
    device_setup_cancel_scope,
)

# ---------------------------------------------------------------------------
# Fixture: clean registry before/after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_registry() -> object:
    """Ensure GlobalDeviceManager._devices is empty before and after each test.

    GlobalDeviceManager is a class-level singleton; without cleanup a test's
    registrations would leak into subsequent tests and cause spurious failures.
    In-flight close claims are cleared too so a failed test cannot leave a
    stale identity claimed for the next one.
    """
    GlobalDeviceManager._devices.clear()
    GlobalDeviceManager._close_claims.clear()
    yield
    GlobalDeviceManager._devices.clear()
    GlobalDeviceManager._close_claims.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_slow_ramp_device() -> FakeDevice:
    """FakeDevice with real sleep ramp and a tiny rampstep.

    The ramp spans many linspace steps, giving other threads a wide window to
    call manager APIs while the ramp holds the per-instance lock.
    """
    dev = FakeDevice(fast_mode=False)
    dev._rampstep = 1e-2  # many steps => long ramp window
    return dev


def _make_fast_device() -> FakeDevice:
    """FakeDevice that completes setup instantly (for the observer device B)."""
    dev = FakeDevice(fast_mode=True)
    dev._rampstep = 0.1
    return dev


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
        release.wait()
        if fail_after_release == "ordinary":
            raise RuntimeError("probe close failure after release")
        if fail_after_release == "base":
            raise KeyboardInterrupt("probe base abort after release")
        original()

    device.close = hooked  # type: ignore[method-assign]
    return _CloseProbe(calls, entered, release)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_info_not_blocked_by_concurrent_setup_on_other_device() -> None:
    """get_info("B") must return before device A's ramp finishes.

    Old implementation: manager._lock was held for the entire setup() call,
    so get_info("B") would block until A's ramp completed.

    New implementation: manager._lock is released before per-device I/O, so
    get_info("B") resolves independently from A's ramp.

    Verification strategy
    ---------------------
    - ramp_started  : set inside A's _setup hook, guarantees thread 2 fires
                      while A definitely holds its per-instance lock.
    - writer_done   : set after A's setup() returns.
    - get_info_returned : set by thread 2 as soon as get_info("B") returns.

    We assert get_info_returned is set while writer_done is *not* yet set,
    proving get_info("B") completed before A's ramp finished.
    """
    dev_a = _make_slow_ramp_device()
    dev_b = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev_a)
    GlobalDeviceManager.register_device("B", dev_b)

    cfg_a = FakeDeviceInfo(
        address="none", output="on", value=1.0, rampstep=dev_a._rampstep
    )
    cfg_b = FakeDeviceInfo(
        address="none", output="on", value=0.5, rampstep=dev_b._rampstep
    )
    # Pre-configure B so get_info() has a meaningful state to return.
    GlobalDeviceManager.setup_devices({"B": cfg_b})

    ramp_started = threading.Event()
    writer_done = threading.Event()
    get_info_returned = threading.Event()
    writer_done_at_get_info_time: list[bool] = []

    original_setup = dev_a._setup

    def hooked_setup(*args, **kwargs):  # type: ignore[no-untyped-def]
        ramp_started.set()  # signal: A's per-instance lock is now held
        original_setup(*args, **kwargs)

    dev_a._setup = hooked_setup  # type: ignore[method-assign]

    def writer() -> None:
        try:
            GlobalDeviceManager.setup_devices({"A": cfg_a})
        finally:
            dev_a._setup = original_setup  # type: ignore[method-assign]
            writer_done.set()

    def reader() -> None:
        ramp_started.wait()  # wait until A's ramp is in progress
        GlobalDeviceManager.get_info("B")
        # Record whether writer_done was already set at the moment get_info returned.
        writer_done_at_get_info_time.append(writer_done.is_set())
        get_info_returned.set()

    t_writer = threading.Thread(target=writer)
    t_reader = threading.Thread(target=reader)
    t_writer.start()
    t_reader.start()

    # get_info("B") must return well before the ramp finishes.
    # Timeout of 5 s is generous; a blocked call would hold until the full ramp
    # (many steps × 0.01 s sleep each) completes — orders of magnitude longer.
    returned_in_time = get_info_returned.wait(timeout=5.0)

    t_writer.join()
    t_reader.join()

    assert returned_in_time, (
        "get_info('B') did not return within 5 s while A's ramp was in progress; "
        "the manager lock is likely still held across the setup() I/O."
    )
    assert not writer_done_at_get_info_time[0], (
        "get_info('B') returned only after device A's ramp finished; "
        "expected it to return before writer_done was set (lock-free path)."
    )


def test_register_device_rejects_non_base_device_without_mutating_registry() -> None:
    bad_device = cast(Any, object())

    with pytest.raises(TypeError, match="register_device expected BaseDevice"):
        GlobalDeviceManager.register_device("bad", bad_device)

    assert "bad" not in GlobalDeviceManager.get_all_devices()


def test_register_device_accepts_base_device() -> None:
    dev = _make_fast_device()

    GlobalDeviceManager.register_device("ok", dev)

    assert GlobalDeviceManager.get_device("ok") is dev


def test_get_all_info_returns_correct_snapshot() -> None:
    """get_all_info() returns the current state of all registered devices.

    This is a functional regression check: the new out-of-lock implementation
    must still return accurate info for every device in the registry.
    """
    dev_a = _make_fast_device()
    dev_b = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev_a)
    GlobalDeviceManager.register_device("B", dev_b)

    cfg_a = FakeDeviceInfo(address="none", output="on", value=0.3, rampstep=0.1)
    cfg_b = FakeDeviceInfo(address="none", output="off", value=0.7, rampstep=0.1)
    GlobalDeviceManager.setup_devices({"A": cfg_a, "B": cfg_b})

    all_info = GlobalDeviceManager.get_all_info()

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


def test_setup_devices_fast_fails_on_unknown_name() -> None:
    """Unknown device name in setup_devices raises ValueError before any setup."""
    dev = _make_fast_device()
    GlobalDeviceManager.register_device("known", dev)

    cfg = FakeDeviceInfo(address="none", output="on", value=0.0, rampstep=0.1)
    with pytest.raises(ValueError, match="unknown"):
        GlobalDeviceManager.setup_devices({"known": cfg, "unknown": cfg})


def test_setup_devices_validates_all_names_before_any_setup() -> None:
    """If any name is missing the whole batch is rejected — no partial setup."""
    dev_a = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev_a)
    # "B" is not registered.

    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)

    with pytest.raises(ValueError, match="B"):
        GlobalDeviceManager.setup_devices({"A": cfg, "B": cfg})

    # A must not have been set up (value stays at the default 0.0).
    assert dev_a.get_value() == 0.0, (
        "Device A was set up despite the batch being rejected; "
        "fast-fail before any I/O is broken."
    )


def test_setup_devices_validates_names_before_cancel_short_circuit() -> None:
    dev = _make_fast_device()
    GlobalDeviceManager.register_device("known", dev)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    cancel_signal.set()

    with pytest.raises(ValueError, match="unknown"):
        GlobalDeviceManager.setup_devices(
            {"known": cfg, "unknown": cfg},
            cancel_signal=cancel_signal,
        )

    assert dev.get_output() == "off"
    assert dev.get_value() == 0.0


def test_setup_devices_explicit_cancel_signal_skips_before_first_device() -> None:
    dev = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    cancel_signal.set()

    GlobalDeviceManager.setup_devices({"A": cfg}, cancel_signal=cancel_signal)

    assert dev.get_output() == "off"
    assert dev.get_value() == 0.0


def test_setup_devices_ambient_cancel_signal_skips_before_first_device() -> None:
    dev = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    cancel_signal.set()

    with device_setup_cancel_scope(cancel_signal):
        GlobalDeviceManager.setup_devices({"A": cfg})

    assert dev.get_output() == "off"
    assert dev.get_value() == 0.0


def test_setup_devices_mid_ramp_cancel_stops_before_next_device() -> None:
    dev_a = _make_slow_ramp_device()
    dev_b = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev_a)
    GlobalDeviceManager.register_device("B", dev_b)
    cfg_a = FakeDeviceInfo(
        address="none", output="on", value=1.0, rampstep=dev_a._rampstep
    )
    cfg_b = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    cancel_signal = threading.Event()
    timer = threading.Timer(0.03, cancel_signal.set)

    timer.start()
    try:
        GlobalDeviceManager.setup_devices(
            {"A": cfg_a, "B": cfg_b},
            cancel_signal=cancel_signal,
        )
    finally:
        timer.cancel()

    assert cancel_signal.is_set()
    assert dev_a.get_output() == "on"
    assert 0.0 < dev_a.get_value() < 1.0
    assert dev_b.get_output() == "off"
    assert dev_b.get_value() == 0.0


def test_setup_devices_explicit_cancel_signal_overrides_ambient_signal() -> None:
    dev = _make_fast_device()
    GlobalDeviceManager.register_device("A", dev)
    cfg = FakeDeviceInfo(address="none", output="on", value=0.5, rampstep=0.1)
    ambient_cancel = threading.Event()
    ambient_cancel.set()
    explicit_cancel = threading.Event()

    with device_setup_cancel_scope(ambient_cancel):
        GlobalDeviceManager.setup_devices(
            {"A": cfg},
            cancel_signal=explicit_cancel,
        )

    assert dev.get_output() == "on"
    assert dev.get_value() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Registry-owned disconnect: close_device / close_all_devices
# ---------------------------------------------------------------------------


def test_close_device_missing_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not found"):
        GlobalDeviceManager.close_device("ghost")

    assert "ghost" not in GlobalDeviceManager.get_all_devices()


def test_close_device_ignore_missing_is_noop() -> None:
    GlobalDeviceManager.close_device("ghost", ignore_missing=True)


def test_close_device_success_removes_all_aliases_of_identity() -> None:
    dev = _make_fast_device()
    probe = _hook_close(dev)
    probe.release.set()  # no blocking needed for the success path
    GlobalDeviceManager.register_device("A", dev)
    GlobalDeviceManager.register_device("alias", dev)

    GlobalDeviceManager.close_device("A")

    assert len(probe.calls) == 1
    assert GlobalDeviceManager.get_all_devices() == {}


def test_close_device_failure_keeps_entries_then_retry_succeeds() -> None:
    dev = _make_fast_device()
    probe = _hook_close(dev, fail="first")
    GlobalDeviceManager.register_device("A", dev)
    GlobalDeviceManager.register_device("alias", dev)

    with pytest.raises(DeviceCloseFailure) as excinfo:
        GlobalDeviceManager.close_device("A")

    failure = excinfo.value
    assert failure.names == ("A",)
    assert isinstance(failure.cause, RuntimeError)
    # Failure must not remove entries or aliases: they stay retryable.
    assert set(GlobalDeviceManager.get_all_devices()) == {"A", "alias"}

    # The claim was released on failure: the retry reaches the device again.
    probe.release.set()
    GlobalDeviceManager.close_device("A")

    assert len(probe.calls) == 2
    assert GlobalDeviceManager.get_all_devices() == {}


def test_close_all_empty_registry_is_noop() -> None:
    GlobalDeviceManager.close_all_devices()


def test_close_all_dedupes_identity_and_closes_each_once() -> None:
    dev1 = _make_fast_device()
    dev2 = _make_fast_device()
    probe1 = _hook_close(dev1)
    probe2 = _hook_close(dev2)
    probe1.release.set()
    probe2.release.set()
    GlobalDeviceManager.register_device("A", dev1)
    GlobalDeviceManager.register_device("alias", dev1)  # same identity
    GlobalDeviceManager.register_device("B", dev2)

    GlobalDeviceManager.close_all_devices()

    assert len(probe1.calls) == 1  # aliases dedupe to one close per identity
    assert len(probe2.calls) == 1
    assert GlobalDeviceManager.get_all_devices() == {}


def test_close_all_aggregates_named_failures_and_keeps_entries() -> None:
    dev1 = _make_fast_device()
    dev2 = _make_fast_device()
    _hook_close(dev1, fail="always")
    _hook_close(dev2, fail="always")
    GlobalDeviceManager.register_device("A", dev1)
    GlobalDeviceManager.register_device("alias", dev1)
    GlobalDeviceManager.register_device("B", dev2)

    with pytest.raises(ExceptionGroup) as excinfo:
        GlobalDeviceManager.close_all_devices()

    failures = [
        e for e in excinfo.value.exceptions if isinstance(e, DeviceCloseFailure)
    ]
    assert len(failures) == 2
    name_sets = {tuple(sorted(e.names)) for e in failures}
    assert name_sets == {("A", "alias"), ("B",)}
    assert all(isinstance(e.cause, RuntimeError) for e in failures)

    # Entries stay registered for retry; claims were released (a second
    # close_device call reaches the device instead of raising InProgressError).
    assert set(GlobalDeviceManager.get_all_devices()) == {"A", "alias", "B"}
    with pytest.raises(DeviceCloseFailure):
        GlobalDeviceManager.close_device("A")


def test_close_all_continues_after_failure_and_removes_successes() -> None:
    dev1 = _make_fast_device()
    dev2 = _make_fast_device()
    _hook_close(dev1, fail="always")
    probe2 = _hook_close(dev2)
    probe2.release.set()
    GlobalDeviceManager.register_device("A", dev1)
    GlobalDeviceManager.register_device("B", dev2)

    with pytest.raises(ExceptionGroup) as excinfo:
        GlobalDeviceManager.close_all_devices()

    failures = [
        e for e in excinfo.value.exceptions if isinstance(e, DeviceCloseFailure)
    ]
    assert [e.names for e in failures] == [("A",)]
    assert "A" in GlobalDeviceManager.get_all_devices()
    assert "B" not in GlobalDeviceManager.get_all_devices()
    assert len(probe2.calls) == 1


def test_concurrent_same_identity_close_device_follower_fast_fails() -> None:
    """Second manager close of an in-flight identity fails fast, close once."""
    dev = _make_fast_device()
    probe = _hook_close(dev)
    GlobalDeviceManager.register_device("A", dev)

    leader_done = threading.Event()
    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            GlobalDeviceManager.close_device("A")
        except BaseException as exc:  # pragma: no cover - assertion below
            leader_errors.append(exc)
        finally:
            leader_done.set()

    t_leader = threading.Thread(target=leader)
    t_leader.start()
    assert probe.entered.wait(5.0), "leader did not enter device.close()"

    with pytest.raises(DeviceCloseInProgressError):
        GlobalDeviceManager.close_device("A")

    probe.release.set()
    t_leader.join(timeout=5.0)

    assert not t_leader.is_alive()
    assert leader_errors == []
    assert len(probe.calls) == 1  # the follower never called device.close()
    assert GlobalDeviceManager.get_all_devices() == {}

    # Success released the claim: closing the same identity again (after a
    # re-registration) works instead of raising DeviceCloseInProgressError.
    GlobalDeviceManager.register_device("A", dev)
    GlobalDeviceManager.close_device("A")
    assert len(probe.calls) == 2
    assert GlobalDeviceManager.get_all_devices() == {}


def test_close_device_success_never_leaves_reclaimable_window() -> None:
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
    GlobalDeviceManager.register_device("A", dev)

    leader_done = threading.Event()
    follower_observations: list[str] = []

    def leader() -> None:
        try:
            GlobalDeviceManager.close_device("A")
        finally:
            leader_done.set()

    def follower() -> None:
        # Hammer the finish boundary: every attempt must see either the
        # in-flight claim or the cleaned-up registry, never a re-claimable
        # still-registered identity after the claim is dropped.
        while not leader_done.is_set():
            try:
                GlobalDeviceManager.close_device("A")
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
            GlobalDeviceManager.close_device("A")
        except ValueError:
            follower_observations.append("gone")
        else:
            follower_observations.append("reclaimed")

    t_leader = threading.Thread(target=leader)
    t_leader.start()
    assert probe.entered.wait(5.0), "leader did not enter device.close()"
    t_follower = threading.Thread(target=follower)
    t_follower.start()
    probe.release.set()

    t_leader.join(timeout=5.0)
    t_follower.join(timeout=5.0)
    assert not t_leader.is_alive()
    assert not t_follower.is_alive()
    assert follower_observations == ["gone"], (
        f"follower observed {follower_observations!r}; a reclaimed identity "
        "would mean the already-closed device was closed again"
    )
    assert len(probe.calls) == 1  # close-once identity guarantee
    assert GlobalDeviceManager.get_all_devices() == {}


def test_close_device_follower_fast_fails_while_leader_fails_ordinarily() -> None:
    """Follower fails fast while an in-flight close ends in ordinary failure.

    A3: the claim is held while the leader's close is in flight, so a
    follower raises ``DeviceCloseInProgressError`` and never calls
    ``device.close()``; after the leader's ``DeviceCloseFailure``, the
    released claim lets a retry reach the device again.
    """
    dev = _make_fast_device()
    probe = _hook_close(dev, fail_after_release="ordinary")
    GlobalDeviceManager.register_device("A", dev)

    leader_done = threading.Event()
    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            GlobalDeviceManager.close_device("A")
        except BaseException as exc:  # pragma: no cover - asserted below
            leader_errors.append(exc)
        finally:
            leader_done.set()

    t_leader = threading.Thread(target=leader)
    t_leader.start()
    assert probe.entered.wait(5.0), "leader did not enter device.close()"

    with pytest.raises(DeviceCloseInProgressError):
        GlobalDeviceManager.close_device("A")
    assert len(probe.calls) == 1  # follower never reached the device

    probe.release.set()
    t_leader.join(timeout=5.0)

    assert not t_leader.is_alive()
    assert len(leader_errors) == 1
    assert isinstance(leader_errors[0], DeviceCloseFailure)
    assert "A" in GlobalDeviceManager.get_all_devices()  # entry stays retryable

    # Failure released the claim: the retry re-claims and reaches the device.
    with pytest.raises(DeviceCloseFailure):
        GlobalDeviceManager.close_device("A")
    assert len(probe.calls) == 2


def test_close_device_follower_fast_fails_while_leader_aborts() -> None:
    """Follower fails fast while an in-flight close ends in BaseException.

    A3: the claim is held across the ``BaseException`` too, so the follower
    raises ``DeviceCloseInProgressError``; the leader's ``KeyboardInterrupt``
    propagates unwrapped, and the released claim lets a retry reach the
    device instead of raising ``DeviceCloseInProgressError``.
    """
    dev = _make_fast_device()
    probe = _hook_close(dev, fail_after_release="base")
    GlobalDeviceManager.register_device("A", dev)

    leader_done = threading.Event()
    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            GlobalDeviceManager.close_device("A")
        except BaseException as exc:  # pragma: no cover - asserted below
            leader_errors.append(exc)
        finally:
            leader_done.set()

    t_leader = threading.Thread(target=leader)
    t_leader.start()
    assert probe.entered.wait(5.0), "leader did not enter device.close()"

    with pytest.raises(DeviceCloseInProgressError):
        GlobalDeviceManager.close_device("A")
    assert len(probe.calls) == 1  # follower never reached the device

    probe.release.set()
    t_leader.join(timeout=5.0)

    assert not t_leader.is_alive()
    assert len(leader_errors) == 1
    assert isinstance(leader_errors[0], KeyboardInterrupt)  # propagates unwrapped
    assert "A" in GlobalDeviceManager.get_all_devices()

    # BaseException released the claim: the retry re-claims and reaches the
    # device instead of raising DeviceCloseInProgressError.
    with pytest.raises(KeyboardInterrupt):
        GlobalDeviceManager.close_device("A")
    assert len(probe.calls) == 2


def test_close_device_base_exception_releases_claim_and_propagates() -> None:
    dev = _make_fast_device()
    probe = _hook_close(dev, base_fail=True)
    GlobalDeviceManager.register_device("A", dev)

    with pytest.raises(KeyboardInterrupt):
        GlobalDeviceManager.close_device("A")

    assert "A" in GlobalDeviceManager.get_all_devices()
    # BaseException propagates unwrapped, but the claim is released: a second
    # call re-claims and reaches the device instead of InProgressError.
    with pytest.raises(KeyboardInterrupt):
        GlobalDeviceManager.close_device("A")

    assert len(probe.calls) == 2
    assert "A" in GlobalDeviceManager.get_all_devices()


def test_close_all_reports_in_progress_identity_and_closes_rest() -> None:
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
    GlobalDeviceManager.register_device("A", dev1)
    GlobalDeviceManager.register_device("B", dev2)

    leader_done = threading.Event()

    def leader() -> None:
        GlobalDeviceManager.close_device("A")
        leader_done.set()

    t_leader = threading.Thread(target=leader)
    t_leader.start()
    assert probe1.entered.wait(5.0), "leader did not enter device.close()"

    # Follower close_all: A is in progress, B is claimable and gets closed.
    with pytest.raises(ExceptionGroup) as excinfo:
        GlobalDeviceManager.close_all_devices()

    in_progress = [
        e for e in excinfo.value.exceptions if isinstance(e, DeviceCloseInProgressError)
    ]
    assert [e.names for e in in_progress] == [("A",)]
    assert len(probe2.calls) == 1
    assert "B" not in GlobalDeviceManager.get_all_devices()

    probe1.release.set()
    t_leader.join(timeout=5.0)

    assert not t_leader.is_alive()
    assert len(probe1.calls) == 1  # A's identity was closed exactly once
    assert "A" not in GlobalDeviceManager.get_all_devices()


def test_close_all_follower_batch_fast_fails_all_claimed_identities() -> None:
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
    GlobalDeviceManager.register_device("A", dev1)
    GlobalDeviceManager.register_device("B", dev2)

    leader_done = threading.Event()

    def leader() -> None:
        try:
            GlobalDeviceManager.close_all_devices()
        finally:
            leader_done.set()

    t_leader = threading.Thread(target=leader)
    t_leader.start()
    assert probe1.entered.wait(5.0), "leader did not enter the first close()"

    with pytest.raises(ExceptionGroup) as excinfo:
        GlobalDeviceManager.close_all_devices()

    in_progress = [
        e for e in excinfo.value.exceptions if isinstance(e, DeviceCloseInProgressError)
    ]
    assert {tuple(sorted(e.names)) for e in in_progress} == {("A",), ("B",)}
    assert len(probe1.calls) == 1
    assert len(probe2.calls) == 0  # follower closed nothing

    probe1.release.set()
    probe2.release.set()
    t_leader.join(timeout=5.0)

    assert not t_leader.is_alive()
    assert len(probe1.calls) == 1
    assert len(probe2.calls) == 1
    assert GlobalDeviceManager.get_all_devices() == {}


def test_close_all_follower_close_device_fast_fails_while_leader_fails_ordinarily() -> (
    None
):
    """close_device follower fails fast while a close_all leader fails.

    A3 (batch side): the claim is held for the whole batch close, so a
    close_device follower raises ``DeviceCloseInProgressError`` and never
    reaches the device; after the leader's aggregated ``DeviceCloseFailure``
    the released claim lets a retry reach the device.
    """
    dev = _make_fast_device()
    probe = _hook_close(dev, fail_after_release="ordinary")
    GlobalDeviceManager.register_device("A", dev)

    leader_done = threading.Event()
    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            GlobalDeviceManager.close_all_devices()
        except BaseException as exc:  # pragma: no cover - asserted below
            leader_errors.append(exc)
        finally:
            leader_done.set()

    t_leader = threading.Thread(target=leader)
    t_leader.start()
    assert probe.entered.wait(5.0), "leader did not enter device.close()"

    with pytest.raises(DeviceCloseInProgressError):
        GlobalDeviceManager.close_device("A")
    assert len(probe.calls) == 1  # follower never reached the device

    probe.release.set()
    t_leader.join(timeout=5.0)

    assert not t_leader.is_alive()
    assert len(leader_errors) == 1
    assert isinstance(leader_errors[0], ExceptionGroup)
    assert "A" in GlobalDeviceManager.get_all_devices()  # entry stays retryable

    with pytest.raises(DeviceCloseFailure):
        GlobalDeviceManager.close_device("A")
    assert len(probe.calls) == 2  # retry re-claimed and reached the device


def test_close_all_base_exception_cleans_successes_and_releases_claims() -> None:
    """A BaseException in the batch still cleans up earlier successes.

    A3/D6: when a later device raises ``BaseException``, the batch propagates
    it unwrapped but first atomically drops aliases of identities already
    closed successfully and releases every owned claim.  A follower
    fast-fails while the batch is in flight; after the abort the success is
    gone and the interrupted identity stays retryable.
    """
    good = _make_fast_device()
    bad = _make_fast_device()
    probe_good = _hook_close(good)
    probe_bad = _hook_close(bad, fail_after_release="base")
    probe_good.release.set()  # good closes as soon as the batch reaches it
    GlobalDeviceManager.register_device("good", good)
    GlobalDeviceManager.register_device("bad", bad)

    leader_done = threading.Event()
    leader_errors: list[BaseException] = []

    def leader() -> None:
        try:
            GlobalDeviceManager.close_all_devices()
        except BaseException as exc:  # pragma: no cover - asserted below
            leader_errors.append(exc)
        finally:
            leader_done.set()

    t_leader = threading.Thread(target=leader)
    t_leader.start()
    assert probe_bad.entered.wait(5.0), "leader did not enter the bad close()"
    assert len(probe_good.calls) == 1  # good was already closed

    with pytest.raises(DeviceCloseInProgressError):
        GlobalDeviceManager.close_device("bad")
    assert len(probe_bad.calls) == 1  # follower never reached the device

    probe_bad.release.set()
    t_leader.join(timeout=5.0)

    assert not t_leader.is_alive()
    assert len(leader_errors) == 1
    assert isinstance(leader_errors[0], KeyboardInterrupt)  # propagates unwrapped
    devices = GlobalDeviceManager.get_all_devices()
    assert "good" not in devices  # success cleaned up despite the abort
    assert devices["bad"] is bad  # interrupted identity stays retryable

    # BaseException released the claim: the retry re-claims and reaches the
    # device instead of raising DeviceCloseInProgressError.
    with pytest.raises(KeyboardInterrupt):
        GlobalDeviceManager.close_device("bad")
    assert len(probe_bad.calls) == 2


def test_close_device_io_not_under_registry_lock() -> None:
    dev_a = _make_fast_device()
    dev_b = _make_fast_device()
    probe = _hook_close(dev_a)
    GlobalDeviceManager.register_device("A", dev_a)
    GlobalDeviceManager.register_device("B", dev_b)

    def closer() -> None:
        GlobalDeviceManager.close_device("A")

    t_closer = threading.Thread(target=closer)
    t_closer.start()
    assert probe.entered.wait(5.0), "close did not enter device.close()"

    probe_done = threading.Event()
    probe_result: list[object] = []

    def reader() -> None:
        probe_result.append(GlobalDeviceManager.get_device("B"))
        probe_done.set()

    t_reader = threading.Thread(target=reader)
    t_reader.start()
    assert probe_done.wait(5.0), (
        "get_device('B') blocked while close_device I/O was in flight; "
        "the registry lock is held across device.close()."
    )
    assert probe_result[0] is dev_b

    probe.release.set()
    t_closer.join(timeout=5.0)
    assert not t_closer.is_alive()


def test_close_all_io_not_under_registry_lock() -> None:
    dev_a = _make_fast_device()
    dev_b = _make_fast_device()
    probe = _hook_close(dev_a)
    GlobalDeviceManager.register_device("A", dev_a)
    GlobalDeviceManager.register_device("B", dev_b)

    def closer() -> None:
        GlobalDeviceManager.close_all_devices()

    t_closer = threading.Thread(target=closer)
    t_closer.start()
    assert probe.entered.wait(5.0), "close did not enter device.close()"

    probe_done = threading.Event()
    probe_result: list[object] = []

    def reader() -> None:
        probe_result.append(GlobalDeviceManager.get_device("B"))
        probe_done.set()

    t_reader = threading.Thread(target=reader)
    t_reader.start()
    assert probe_done.wait(5.0), (
        "get_device('B') blocked while close_all I/O was in flight; "
        "the registry lock is held across device.close()."
    )
    assert probe_result[0] is dev_b

    probe.release.set()
    t_closer.join(timeout=5.0)
    assert not t_closer.is_alive()
    assert GlobalDeviceManager.get_all_devices() == {}


def test_close_success_keeps_replacement_and_new_devices_removes_stale_alias() -> None:
    dev1 = _make_fast_device()
    probe1 = _hook_close(dev1)
    GlobalDeviceManager.register_device("A", dev1)

    def closer() -> None:
        GlobalDeviceManager.close_device("A")

    t_closer = threading.Thread(target=closer)
    t_closer.start()
    assert probe1.entered.wait(5.0), "close did not enter device.close()"

    # Same-name replacement, a distinct new device, and an alias pointing at
    # the closing identity all appear while the close is in flight.
    dev2 = _make_fast_device()
    dev3 = _make_fast_device()
    probe2 = _hook_close(dev2)
    probe3 = _hook_close(dev3)
    probe2.release.set()
    probe3.release.set()
    GlobalDeviceManager.register_device("A", dev2)  # overwrite warning expected
    GlobalDeviceManager.register_device("B", dev3)
    GlobalDeviceManager.register_device("stale", dev1)

    probe1.release.set()
    t_closer.join(timeout=5.0)

    assert not t_closer.is_alive()
    assert len(probe1.calls) == 1
    assert len(probe2.calls) == 0  # replacement was not closed
    assert len(probe3.calls) == 0  # new device was not closed
    devices = GlobalDeviceManager.get_all_devices()
    assert devices["A"] is dev2  # same-name replacement survives
    assert devices["B"] is dev3  # distinct new device survives
    assert "stale" not in devices  # alias of the closed identity is removed


def test_close_all_snapshot_excludes_devices_registered_during_close() -> None:
    dev1 = _make_fast_device()
    probe1 = _hook_close(dev1)
    GlobalDeviceManager.register_device("A", dev1)

    def closer() -> None:
        GlobalDeviceManager.close_all_devices()

    t_closer = threading.Thread(target=closer)
    t_closer.start()
    assert probe1.entered.wait(5.0), "close did not enter device.close()"

    dev2 = _make_fast_device()
    probe2 = _hook_close(dev2)
    probe2.release.set()
    GlobalDeviceManager.register_device("B", dev2)

    probe1.release.set()
    t_closer.join(timeout=5.0)

    assert not t_closer.is_alive()
    assert len(probe1.calls) == 1
    assert len(probe2.calls) == 0  # not part of the batch snapshot
    devices = GlobalDeviceManager.get_all_devices()
    assert "A" not in devices
    assert devices["B"] is dev2

from __future__ import annotations

import threading
import time

import pytest
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationConflictError,
    OperationGate,
    OperationKind,
    OperationOutcome,
)

_FINISHED = OperationOutcome("finished")


@pytest.mark.parametrize(
    ("active", "requested"),
    [
        (OperationKind.RUN, OperationKind.RUN),
        (OperationKind.RUN, OperationKind.SOC_CONNECT),
        (OperationKind.RUN, OperationKind.DEVICE_CONNECT),
        (OperationKind.SOC_CONNECT, OperationKind.RUN),
        (OperationKind.SOC_CONNECT, OperationKind.SOC_CONNECT),
        (OperationKind.DEVICE_CONNECT, OperationKind.RUN),
        (OperationKind.DEVICE_CONNECT, OperationKind.DEVICE_DISCONNECT),
        (OperationKind.DEVICE_SETUP, OperationKind.DEVICE_CONNECT),
    ],
)
def test_operation_gate_rejects_conflicts(
    active: OperationKind, requested: OperationKind
) -> None:
    gate = OperationGate()
    gate.acquire(active, owner_id="first", resource_id="a")

    with pytest.raises(OperationConflictError):
        gate.acquire(requested, owner_id="second", resource_id="b")


def test_operation_gate_allows_soc_connect_during_device_mutation() -> None:
    gate = OperationGate()
    gate.acquire(OperationKind.DEVICE_CONNECT, owner_id="device", resource_id="flux")

    lease = gate.acquire(OperationKind.SOC_CONNECT, owner_id="soc")

    assert gate.has_active(OperationKind.SOC_CONNECT)
    gate.release(lease, _FINISHED)


def test_operation_gate_tracks_device_mutation_by_name() -> None:
    gate = OperationGate()
    lease = gate.acquire(
        OperationKind.DEVICE_SETUP, owner_id="setup", resource_id="flux"
    )

    assert gate.is_device_mutating("flux")
    assert not gate.is_device_mutating("rf")
    gate.release(lease, _FINISHED)
    assert not gate.is_device_mutating("flux")


def test_release_frees_hardware_immediately() -> None:
    # Exclusion is removed on release so a conflicting op can start at once,
    # even though the handle is retained for late awaiters.
    gate = OperationGate()
    lease = gate.acquire(OperationKind.RUN, owner_id="tab")
    gate.release(lease, _FINISHED)
    # No conflict now — RUN exclusion was dropped.
    gate.acquire(OperationKind.RUN, owner_id="tab2")


def test_operation_gate_rejects_double_release() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.RUN, owner_id="tab")
    gate.release(lease, _FINISHED)

    with pytest.raises(RuntimeError, match="not active"):
        gate.release(lease, _FINISHED)


# ---------------------------------------------------------------------------
# await_outcome — thread-safe wait for off-main blocking handlers
# ---------------------------------------------------------------------------


def test_await_outcome_unblocks_on_release_with_outcome() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.DEVICE_SETUP, owner_id="d", resource_id="d")

    got: list[object] = []
    dt: list[float] = []

    def waiter() -> None:
        t0 = time.monotonic()
        got.append(gate.await_outcome(lease.token, timeout=3.0))
        dt.append(time.monotonic() - t0)

    wt = threading.Thread(target=waiter)
    wt.start()
    time.sleep(0.1)
    gate.release(lease, OperationOutcome("failed", "boom"))  # from "main" thread
    wt.join(timeout=2.0)

    assert got and got[0] == OperationOutcome("failed", "boom")
    assert dt[0] < 1.0  # woke promptly, not on timeout


def test_await_outcome_immediate_for_already_released() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.DEVICE_SETUP, owner_id="d")
    gate.release(lease, OperationOutcome("finished"))

    t0 = time.monotonic()
    assert gate.await_outcome(lease.token, timeout=5.0) == OperationOutcome("finished")
    assert time.monotonic() - t0 < 0.5


def test_await_outcome_immediate_for_unknown_token() -> None:
    gate = OperationGate()
    t0 = time.monotonic()
    # Unknown/evicted token is treated as already finished (never hangs).
    assert gate.await_outcome(99999, timeout=5.0) == OperationOutcome("finished")
    assert time.monotonic() - t0 < 0.5


def test_await_outcome_times_out_while_active() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.DEVICE_SETUP, owner_id="d")
    assert gate.await_outcome(lease.token, timeout=0.1) is None


# ---------------------------------------------------------------------------
# poll — non-blocking status
# ---------------------------------------------------------------------------


def test_poll_pending_then_settled() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.DEVICE_SETUP, owner_id="d")
    assert gate.poll(lease.token) is None  # still pending
    gate.release(lease, OperationOutcome("finished"))
    assert gate.poll(lease.token) == OperationOutcome("finished")


def test_poll_unknown_token_is_finished() -> None:
    gate = OperationGate()
    assert gate.poll(99999) == OperationOutcome("finished")


# ---------------------------------------------------------------------------
# cancel — async stop notification (sets the worker's stop_event)
# ---------------------------------------------------------------------------


def test_cancel_sets_the_registered_stop_event() -> None:
    gate = OperationGate()
    stop_event = threading.Event()
    lease = gate.acquire(
        OperationKind.DEVICE_SETUP, owner_id="d", stop_event=stop_event
    )

    assert not stop_event.is_set()
    gate.cancel(lease.token)
    assert stop_event.is_set()
    # cancel is a request, not a settle: the operation stays active until the
    # worker self-judges and the owner releases the lease.
    assert gate.poll(lease.token) is None


def test_cancel_is_noop_for_operation_without_stop_event() -> None:
    # A connect has no cancellation point: acquire passes no stop_event, so
    # cancel does nothing (shutdown falls back to a timeout force-close).
    gate = OperationGate()
    lease = gate.acquire(OperationKind.SOC_CONNECT, owner_id="soc")
    gate.cancel(lease.token)  # must not raise
    assert gate.poll(lease.token) is None


def test_cancel_unknown_token_is_noop() -> None:
    gate = OperationGate()
    gate.cancel(99999)  # must not raise


def test_cancel_all_sets_every_live_stop_event_and_returns_tokens() -> None:
    # DEVICE_SETUP (cancellable, has stop_event) coexists with SOC_CONNECT
    # (no cancellation point, no stop_event) per the conflict rules.
    gate = OperationGate()
    setup_stop = threading.Event()
    setup_lease = gate.acquire(
        OperationKind.DEVICE_SETUP, owner_id="d", stop_event=setup_stop
    )
    # A connect with no stop_event is still returned (its token must be polled
    # for shutdown to know whether it settled), but its flag stays unset.
    connect_lease = gate.acquire(OperationKind.SOC_CONNECT, owner_id="soc")

    tokens = gate.cancel_all()

    assert set(tokens) == {setup_lease.token, connect_lease.token}
    assert setup_stop.is_set()


def test_cancel_all_ignores_already_settled_operations() -> None:
    gate = OperationGate()
    stop_event = threading.Event()
    lease = gate.acquire(OperationKind.RUN, owner_id="tab", stop_event=stop_event)
    gate.release(lease, OperationOutcome("finished"))

    assert gate.cancel_all() == []

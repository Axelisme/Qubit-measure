from __future__ import annotations

import threading
import time

import pytest
from zcu_tools.gui.services.operation_gate import (
    OperationConflictError,
    OperationGate,
    OperationKind,
)


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
        (OperationKind.DEVICE_SETUP, OperationKind.DEVICE_SET_VALUE),
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
    gate.release(lease)


def test_operation_gate_tracks_device_mutation_by_name() -> None:
    gate = OperationGate()
    lease = gate.acquire(
        OperationKind.DEVICE_SETUP, owner_id="setup", resource_id="flux"
    )

    assert gate.is_device_mutating("flux")
    assert not gate.is_device_mutating("rf")
    gate.release(lease)
    assert not gate.is_device_mutating("flux")


def test_operation_gate_rejects_double_release() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.RUN, owner_id="tab")
    gate.release(lease)

    with pytest.raises(RuntimeError, match="not active"):
        gate.release(lease)


# ---------------------------------------------------------------------------
# await_release (Phase 93) — thread-safe wait for off-main blocking handlers
# ---------------------------------------------------------------------------


def test_await_release_unblocks_on_release() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.DEVICE_SETUP, owner_id="d", resource_id="d")

    ok: list[bool] = []
    dt: list[float] = []

    def waiter() -> None:
        t0 = time.monotonic()
        ok.append(gate.await_release(lease.token, timeout=3.0))
        dt.append(time.monotonic() - t0)

    wt = threading.Thread(target=waiter)
    wt.start()
    time.sleep(0.1)
    gate.release(lease)  # from "main" thread
    wt.join(timeout=2.0)

    assert ok == [True]
    assert dt[0] < 1.0  # woke promptly, not on timeout


def test_await_release_returns_immediately_for_already_released() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.DEVICE_SETUP, owner_id="d")
    gate.release(lease)

    t0 = time.monotonic()
    assert gate.await_release(lease.token, timeout=5.0) is True
    assert time.monotonic() - t0 < 0.5


def test_await_release_returns_immediately_for_unknown_token() -> None:
    gate = OperationGate()
    t0 = time.monotonic()
    assert gate.await_release(99999, timeout=5.0) is True
    assert time.monotonic() - t0 < 0.5


def test_await_release_times_out_while_active() -> None:
    gate = OperationGate()
    lease = gate.acquire(OperationKind.DEVICE_SETUP, owner_id="d")
    assert gate.await_release(lease.token, timeout=0.1) is False

from __future__ import annotations

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

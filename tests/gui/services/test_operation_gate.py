"""Tests for OperationGate — the Exclusion facet (ADR-0019).

Pure hardware mutual-exclusion keyed by an externally-minted token: the
fail-fast ``ensure_can_start`` guard, ``register`` / ``release``, and the
device-name / kind queries. The async handle (await / poll / cancel) lives in
``OperationHandles`` — see test_operation_handles.py.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationGate,
)
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationKind as MeasureOpKind,
)
from zcu_tools.gui.session.ports import OperationConflictError, OperationKind


@pytest.mark.parametrize(
    ("active", "requested"),
    [
        (MeasureOpKind.RUN, MeasureOpKind.RUN),
        (MeasureOpKind.RUN, OperationKind.SOC_CONNECT),
        (MeasureOpKind.RUN, OperationKind.DEVICE_CONNECT),
        (OperationKind.SOC_CONNECT, MeasureOpKind.RUN),
        (OperationKind.SOC_CONNECT, OperationKind.SOC_CONNECT),
        (OperationKind.DEVICE_CONNECT, MeasureOpKind.RUN),
        (OperationKind.DEVICE_CONNECT, OperationKind.DEVICE_DISCONNECT),
        (OperationKind.DEVICE_SETUP, OperationKind.DEVICE_CONNECT),
    ],
)
def test_ensure_can_start_rejects_conflicts(
    active: OperationKind, requested: OperationKind
) -> None:
    gate = OperationGate()
    gate.register(1, active, owner_id="first", resource_id="a")

    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(requested)


def test_allows_soc_connect_during_device_mutation() -> None:
    gate = OperationGate()
    gate.register(
        1, OperationKind.DEVICE_CONNECT, owner_id="device", resource_id="flux"
    )

    gate.ensure_can_start(OperationKind.SOC_CONNECT)  # no conflict
    gate.register(2, OperationKind.SOC_CONNECT, owner_id="soc")

    assert gate.has_active(OperationKind.SOC_CONNECT)
    gate.release(2)


def test_device_mutations_of_different_devices_are_concurrent() -> None:
    """Phase C: a device mutation does not block a mutation of a *different*
    device (resource-aware conflict scoped by resource_id)."""
    gate = OperationGate()
    gate.register(1, OperationKind.DEVICE_SETUP, owner_id="a", resource_id="devA")

    # Different device → no conflict; it registers alongside.
    gate.ensure_can_start(OperationKind.DEVICE_SETUP, resource_id="devB")
    gate.register(2, OperationKind.DEVICE_SETUP, owner_id="b", resource_id="devB")

    assert gate.is_device_mutating("devA")
    assert gate.is_device_mutating("devB")
    gate.release(1)
    gate.release(2)


def test_device_mutation_of_same_device_conflicts() -> None:
    """A mutation of the SAME device still conflicts (resource-aware match)."""
    gate = OperationGate()
    gate.register(1, OperationKind.DEVICE_SETUP, owner_id="a", resource_id="devA")

    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(OperationKind.DEVICE_CONNECT, resource_id="devA")
    gate.release(1)


def test_run_blocks_every_device_mutation_regardless_of_resource() -> None:
    """RUN ↔ device-mutation stays a *global* mutual exclusion (resource_id is
    irrelevant): a sweep drives hardware, so no device may be mutated during it
    and vice-versa."""
    gate = OperationGate()
    gate.register(1, MeasureOpKind.RUN, owner_id="tab")
    # Any device, any name → still blocked.
    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(OperationKind.DEVICE_SETUP, resource_id="devA")
    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(OperationKind.DEVICE_SETUP, resource_id="devB")
    gate.release(1)

    # And the reverse: a device mutation blocks RUN.
    gate.register(2, OperationKind.DEVICE_SETUP, owner_id="a", resource_id="devA")
    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(MeasureOpKind.RUN)
    gate.release(2)


def test_tracks_device_mutation_by_name() -> None:
    gate = OperationGate()
    gate.register(1, OperationKind.DEVICE_SETUP, owner_id="setup", resource_id="flux")

    assert gate.is_device_mutating("flux")
    assert not gate.is_device_mutating("rf")
    gate.release(1)
    assert not gate.is_device_mutating("flux")


def test_release_frees_hardware_immediately() -> None:
    # Exclusion is removed on release so a conflicting op can start at once.
    gate = OperationGate()
    gate.register(1, MeasureOpKind.RUN, owner_id="tab")
    gate.release(1)
    # No conflict now — RUN exclusion was dropped.
    gate.ensure_can_start(MeasureOpKind.RUN)
    gate.register(2, MeasureOpKind.RUN, owner_id="tab2")


def test_rejects_double_release() -> None:
    gate = OperationGate()
    gate.register(1, MeasureOpKind.RUN, owner_id="tab")
    gate.release(1)

    with pytest.raises(RuntimeError, match="not active"):
        gate.release(1)


def test_register_rejects_empty_owner() -> None:
    gate = OperationGate()
    with pytest.raises(ValueError, match="owner_id"):
        gate.register(1, MeasureOpKind.RUN, owner_id="")

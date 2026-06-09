"""Tests for OperationGate — the Exclusion facet (ADR-0019).

Pure hardware mutual-exclusion keyed by an externally-minted token: the
fail-fast ``ensure_can_start`` guard, ``register`` / ``release``, and the
device-name / kind queries. The async handle (await / poll / cancel) lives in
``OperationHandles`` — see test_operation_handles.py.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.services.operation_gate import (
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
    gate.register(1, OperationKind.RUN, owner_id="tab")
    gate.release(1)
    # No conflict now — RUN exclusion was dropped.
    gate.ensure_can_start(OperationKind.RUN)
    gate.register(2, OperationKind.RUN, owner_id="tab2")


def test_rejects_double_release() -> None:
    gate = OperationGate()
    gate.register(1, OperationKind.RUN, owner_id="tab")
    gate.release(1)

    with pytest.raises(RuntimeError, match="not active"):
        gate.release(1)


def test_register_rejects_empty_owner() -> None:
    gate = OperationGate()
    with pytest.raises(ValueError, match="owner_id"):
        gate.register(1, OperationKind.RUN, owner_id="")

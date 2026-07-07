"""Tests for OperationGate — the Exclusion facet (ADR-0019).

Pure hardware mutual-exclusion keyed by an externally-minted token: the
fail-fast ``ensure_can_start`` guard, ``register`` / ``release``, and the
device-name / kind queries. The async handle (await / poll / cancel) lives in
``OperationHandles`` — see test_operation_handles.py.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import pytest
from zcu_tools.gui.app.autofluxdep.operation_gate import (
    OperationGate as AutoFluxDepOperationGate,
)
from zcu_tools.gui.app.autofluxdep.operation_gate import (
    OperationKind as AutoFluxDepOpKind,
)
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationGate as MeasureOperationGate,
)
from zcu_tools.gui.app.main.services.operation_gate import (
    OperationKind as MeasureOpKind,
)
from zcu_tools.gui.session.ports import (
    ExclusionGate,
    OperationConflictError,
    OperationKind,
)


class InspectableExclusionGate(ExclusionGate, Protocol):
    def has_active(self, kind: str) -> bool: ...
    def is_device_mutating(self, name: str) -> bool: ...


GateCase = tuple[InspectableExclusionGate, str]
KindSelector = Callable[[str], str]


@pytest.fixture(
    params=[
        pytest.param(
            lambda: (MeasureOperationGate(), MeasureOpKind.RUN),
            id="measure",
        ),
        pytest.param(
            lambda: (AutoFluxDepOperationGate(), AutoFluxDepOpKind.RUN),
            id="autofluxdep",
        ),
    ]
)
def gate_case(request: pytest.FixtureRequest) -> GateCase:
    factory = cast(Callable[[], GateCase], request.param)
    return factory()


@pytest.mark.parametrize(
    ("active_getter", "requested_getter"),
    [
        (lambda run: run, lambda run: run),
        (lambda run: run, lambda _run: OperationKind.SOC_CONNECT),
        (lambda run: run, lambda _run: OperationKind.DEVICE_CONNECT),
        (lambda _run: OperationKind.SOC_CONNECT, lambda run: run),
        (
            lambda _run: OperationKind.SOC_CONNECT,
            lambda _run: OperationKind.SOC_CONNECT,
        ),
        (lambda _run: OperationKind.DEVICE_CONNECT, lambda run: run),
        (
            lambda _run: OperationKind.DEVICE_CONNECT,
            lambda _run: OperationKind.DEVICE_DISCONNECT,
        ),
        (
            lambda _run: OperationKind.DEVICE_SETUP,
            lambda _run: OperationKind.DEVICE_CONNECT,
        ),
    ],
)
def test_ensure_can_start_rejects_conflicts(
    gate_case: GateCase,
    active_getter: KindSelector,
    requested_getter: KindSelector,
) -> None:
    gate, run_kind = gate_case
    active = active_getter(run_kind)
    requested = requested_getter(run_kind)
    gate.register(1, active, owner_id="first", resource_id="a")

    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(requested)


def test_allows_soc_connect_during_device_mutation(
    gate_case: GateCase,
) -> None:
    gate, _run_kind = gate_case
    gate.register(
        1, OperationKind.DEVICE_CONNECT, owner_id="device", resource_id="flux"
    )

    gate.ensure_can_start(OperationKind.SOC_CONNECT)  # no conflict
    gate.register(2, OperationKind.SOC_CONNECT, owner_id="soc")

    assert gate.has_active(OperationKind.SOC_CONNECT)
    gate.release(2)


def test_device_mutations_of_different_devices_are_concurrent(
    gate_case: GateCase,
) -> None:
    """Phase C: a device mutation does not block a mutation of a *different*
    device (resource-aware conflict scoped by resource_id)."""
    gate, _run_kind = gate_case
    gate.register(1, OperationKind.DEVICE_SETUP, owner_id="a", resource_id="devA")

    # Different device → no conflict; it registers alongside.
    gate.ensure_can_start(OperationKind.DEVICE_SETUP, resource_id="devB")
    gate.register(2, OperationKind.DEVICE_SETUP, owner_id="b", resource_id="devB")

    assert gate.is_device_mutating("devA")
    assert gate.is_device_mutating("devB")
    gate.release(1)
    gate.release(2)


def test_device_mutation_of_same_device_conflicts(
    gate_case: GateCase,
) -> None:
    """A mutation of the SAME device still conflicts (resource-aware match)."""
    gate, _run_kind = gate_case
    gate.register(1, OperationKind.DEVICE_SETUP, owner_id="a", resource_id="devA")

    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(OperationKind.DEVICE_CONNECT, resource_id="devA")
    gate.release(1)


def test_run_blocks_every_device_mutation_regardless_of_resource(
    gate_case: GateCase,
) -> None:
    """RUN ↔ device-mutation stays a *global* mutual exclusion (resource_id is
    irrelevant): a sweep drives hardware, so no device may be mutated during it
    and vice-versa."""
    gate, run_kind = gate_case
    gate.register(1, run_kind, owner_id="tab")
    # Any device, any name → still blocked.
    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(OperationKind.DEVICE_SETUP, resource_id="devA")
    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(OperationKind.DEVICE_SETUP, resource_id="devB")
    gate.release(1)

    # And the reverse: a device mutation blocks RUN.
    gate.register(2, OperationKind.DEVICE_SETUP, owner_id="a", resource_id="devA")
    with pytest.raises(OperationConflictError):
        gate.ensure_can_start(run_kind)
    gate.release(2)


def test_tracks_device_mutation_by_name(
    gate_case: GateCase,
) -> None:
    gate, _run_kind = gate_case
    gate.register(1, OperationKind.DEVICE_SETUP, owner_id="setup", resource_id="flux")

    assert gate.is_device_mutating("flux")
    assert not gate.is_device_mutating("rf")
    gate.release(1)
    assert not gate.is_device_mutating("flux")


def test_release_frees_hardware_immediately(
    gate_case: GateCase,
) -> None:
    # Exclusion is removed on release so a conflicting op can start at once.
    gate, run_kind = gate_case
    gate.register(1, run_kind, owner_id="tab")
    gate.release(1)
    # No conflict now — RUN exclusion was dropped.
    gate.ensure_can_start(run_kind)
    gate.register(2, run_kind, owner_id="tab2")


def test_rejects_double_release(
    gate_case: GateCase,
) -> None:
    gate, run_kind = gate_case
    gate.register(1, run_kind, owner_id="tab")
    gate.release(1)

    with pytest.raises(RuntimeError, match="not active"):
        gate.release(1)


def test_register_rejects_empty_owner(
    gate_case: GateCase,
) -> None:
    gate, run_kind = gate_case
    with pytest.raises(ValueError, match="owner_id"):
        gate.register(1, run_kind, owner_id="")

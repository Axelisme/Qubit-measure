"""Node I/O container tests — Snapshot (read-only projection) + Patch contract.

Covers the snapshot-in / patch-out boundary: a Snapshot holds exactly the
declared keys (undeclared → KeyError), and merging a Patch validates every key
is in the Node's ``provides`` (else PatchContractError). Also the orchestrator
end-to-end: a Node producing an undeclared key fast-fails.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.controller import Controller
from zcu_tools.gui.app.autofluxdep.event_bus import EventBus
from zcu_tools.gui.app.autofluxdep.nodes.io import (
    Patch,
    PatchContractError,
    Snapshot,
    validate_patch,
)
from zcu_tools.gui.app.autofluxdep.nodes.spec import NodeInstance, NodeSpec
from zcu_tools.gui.app.autofluxdep.state import AutoFluxDepState

# --- Snapshot: read-only projection of declared keys ---


def test_snapshot_reads_declared_keys():
    s = Snapshot({"a": 1, "b": 2})
    assert s["a"] == 1
    assert s.get("b") == 2
    assert set(s) == {"a", "b"}
    assert len(s) == 2


def test_snapshot_undeclared_key_raises_keyerror():
    s = Snapshot({"a": 1})
    with pytest.raises(KeyError):
        _ = s["nope"]
    assert s.get("nope") is None  # .get still works, returns default


def test_snapshot_is_immutable():
    s = Snapshot({"a": 1})
    with pytest.raises(TypeError):
        s["a"] = 2  # type: ignore[index]


def test_snapshot_equals_mapping_by_value():
    assert Snapshot({"a": 1}) == {"a": 1}
    assert Snapshot({"a": 1}) == Snapshot({"a": 1})
    assert Snapshot({"a": 1}) != {"a": 2}


# --- Patch: produced container + provides contract ---


def test_patch_set_values_and_modules():
    p = Patch()
    p.set("x", 10)
    p.set_module("readout", "RO")
    assert p.values() == {"x": 10}
    assert p.modules() == {"readout": "RO"}


def test_validate_patch_accepts_declared_keys_and_modules():
    validate_patch(
        Patch({"x": 1}, modules={"readout": "RO"}),
        provides=("x", "y"),
        provides_modules=("readout",),
    )  # no raise


def test_validate_patch_rejects_undeclared_value():
    with pytest.raises(PatchContractError, match="info key"):
        validate_patch(Patch({"x": 1, "rogue": 2}), provides=("x",))


def test_validate_patch_rejects_undeclared_module():
    with pytest.raises(PatchContractError, match="module"):
        validate_patch(
            Patch(modules={"rogue_mod": object()}), provides=(), provides_modules=()
        )


# --- Snapshot module side ---


def test_snapshot_module_reads_declared_undeclared_raises():
    s = Snapshot({"v": 1}, modules={"readout": "RO"})
    assert s.module("readout") == "RO"
    assert s.has_module("readout")
    assert not s.has_module("nope")
    with pytest.raises(KeyError):
        s.module("nope")


# --- orchestrator enforces both contracts end-to-end ---


def test_orchestrator_fast_fails_on_undeclared_value():
    spec = NodeSpec(name="bad", provides=("a",))
    state = AutoFluxDepState(flux_values=[0.0])
    ctrl = Controller(state, EventBus())
    ctrl.add_node(spec)

    def run_node(node: NodeInstance, _snap, _tools) -> Patch:
        return Patch({"b": 1})  # "b" not in provides

    with pytest.raises(PatchContractError):
        ctrl.dry_run(run_node)


def test_orchestrator_fast_fails_on_undeclared_module():
    spec = NodeSpec(name="bad", provides=())  # provides_modules empty
    state = AutoFluxDepState(flux_values=[0.0])
    ctrl = Controller(state, EventBus())
    ctrl.add_node(spec)

    def run_node(node: NodeInstance, _snap, _tools) -> Patch:
        return Patch(modules={"readout": "RO"})  # not in provides_modules

    with pytest.raises(PatchContractError):
        ctrl.dry_run(run_node)

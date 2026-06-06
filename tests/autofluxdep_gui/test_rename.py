"""Instance-name tests — placing the same Builder twice under distinct names.

A PlacedNode's ``name`` is its instance identity (display label + the key into
run_results / Plotters + the auto-follow / remove target), distinct from the
Builder's type name. The controller de-dups names within a workflow and renames
on request, so two ``mist`` placements can become ``g_mist`` / ``e_mist`` with
independent Results — while both still provide the flat info key ``success``.
"""

from __future__ import annotations

import numpy as np
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.event_bus import EventType


def test_repeated_placement_auto_dedups_name():
    ctrl = build_core()
    ctrl.add_node_by_type("mist")
    ctrl.add_node_by_type("mist")
    ctrl.add_node_by_type("mist")
    assert ctrl.state.node_names() == ["mist", "mist_2", "mist_3"]


def test_rename_applies_and_keeps_type():
    ctrl = build_core()
    ctrl.add_node_by_type("mist")
    ctrl.add_node_by_type("mist")
    ctrl.rename_node(0, "g_mist")
    ctrl.rename_node(1, "e_mist")
    assert ctrl.state.node_names() == ["g_mist", "e_mist"]
    # the instance name changed; the Builder type name did not
    assert ctrl.state.nodes[0].type_name == "mist"
    assert ctrl.state.nodes[1].type_name == "mist"


def test_rename_to_taken_name_dedups():
    ctrl = build_core()
    ctrl.add_node_by_type("mist")  # mist
    ctrl.add_node_by_type("mist")  # mist_2
    applied = ctrl.rename_node(1, "mist")  # collides with node 0
    assert applied == "mist_2"  # de-duped, not a duplicate
    assert ctrl.state.node_names() == ["mist", "mist_2"]


def test_rename_blank_is_noop():
    ctrl = build_core()
    ctrl.add_node_by_type("mist")
    applied = ctrl.rename_node(0, "   ")
    assert applied == "mist"  # blank rejected, name unchanged


def test_rename_emits_workflow_changed():
    ctrl = build_core()
    ctrl.add_node_by_type("mist")
    seen = []
    ctrl.bus.subscribe(EventType.WORKFLOW_CHANGED, lambda e: seen.append(e.payload))
    ctrl.rename_node(0, "g_mist")
    assert "g_mist" in seen


def test_two_mist_instances_get_independent_results():
    ctrl = build_core()
    ctrl.add_node_by_type("mist")
    ctrl.rename_node(0, "g_mist")
    ctrl.add_node_by_type("mist")
    ctrl.rename_node(1, "e_mist")
    ctrl.setup(use_mock=True)
    ctrl.set_flux_values([0.0, 0.5])
    ctrl.start_run()

    results = ctrl.state.run_results
    assert set(results) == {"g_mist", "e_mist"}  # keyed by instance name
    assert results["g_mist"] is not results["e_mist"]  # independent containers
    for res in results.values():
        assert not np.isnan(res.signal[-1]).any()  # each filled over the sweep


def test_remove_uses_instance_name():
    ctrl = build_core()
    ctrl.add_node_by_type("mist")
    ctrl.rename_node(0, "g_mist")
    ctrl.add_node_by_type("mist")
    ctrl.rename_node(1, "e_mist")
    ctrl.remove_node("g_mist")
    assert ctrl.state.node_names() == ["e_mist"]


def test_node_entered_excludes_predictor_service():
    # the controller forwards NODE_ENTERED only for user-list Nodes; the injected
    # predictor Service has no list row, so it is filtered out.
    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    ctrl.add_node_by_type("mist")
    ctrl.rename_node(1, "g_mist")
    ctrl.setup(use_mock=True)
    ctrl.set_flux_values([0.0, 1.0])

    entered = []
    ctrl.bus.subscribe(EventType.NODE_ENTERED, lambda e: entered.append(e.payload[0]))
    ctrl.start_run()
    assert "predictor" not in entered
    assert set(entered) == {"qubit_freq", "g_mist"}

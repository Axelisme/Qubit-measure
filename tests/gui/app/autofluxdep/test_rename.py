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
from zcu_tools.gui.app.autofluxdep.events.run import NodeEnteredPayload
from zcu_tools.gui.app.autofluxdep.events.workflow import WorkflowChangedPayload
from zcu_tools.gui.app.autofluxdep.experiments._support.result import Sweep1DResult
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch
from zcu_tools.gui.cfg import FloatSpec

from ._helpers import make_builder, run_controller_to_completion, set_node_cfg_knobs


def _make_filling_builder(name: str):
    """A fake measurement Builder: a 1-D Result whose row this point's produce fills.

    Mechanics tests assert per-instance ``run_results`` containers without any
    acquire — the Node just writes a constant signal row so the Result is "filled"."""

    def _result_factory(params, flux):
        del params
        return Sweep1DResult.allocate(
            np.asarray(flux, dtype=float), np.linspace(0.0, 1.0, 4), x_label="x"
        )

    def _produce(env, snapshot):
        del snapshot
        env.result.signal[env.flux_idx] = np.ones(env.result.n_x)
        patch = Patch()
        patch.set("success", 1.0)
        return patch

    return make_builder(
        name, provides=("success",), produce_fn=_produce, result_factory=_result_factory
    )


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
    ctrl.bus.subscribe(WorkflowChangedPayload, lambda p: seen.append(p.name))
    ctrl.rename_node(0, "g_mist")
    assert "g_mist" in seen


def test_cfg_edits_emit_workflow_changed_for_edited_node():
    ctrl = build_core()
    node = ctrl.add_node(
        make_builder(
            "cfg_node",
            schema_fields=(("gain", FloatSpec("Gain"), 1.0),),
        )
    )
    seen = []
    ctrl.bus.subscribe(WorkflowChangedPayload, lambda p: seen.append(p.name))

    set_node_cfg_knobs(ctrl, 0, {"gain": 2.0})
    ctrl.set_node_cfg_value(0, node.schema.schema.value)

    assert seen == ["cfg_node", "cfg_node"]


def test_reorder_emits_whole_workflow_changed():
    ctrl = build_core()
    ctrl.add_node(make_builder("first"))
    ctrl.add_node(make_builder("second"))
    seen = []
    ctrl.bus.subscribe(WorkflowChangedPayload, lambda p: seen.append(p.name))

    ctrl.reorder(0, 1)

    assert seen == [None]


def test_two_mist_instances_get_independent_results():
    # two placements of the same fake measurement Builder, renamed, get independent
    # Result containers keyed by instance name (the rename mechanic, no acquire).
    ctrl = build_core()
    ctrl.add_node(_make_filling_builder("mist"))
    ctrl.rename_node(0, "g_mist")
    ctrl.add_node(_make_filling_builder("mist"))
    ctrl.rename_node(1, "e_mist")
    ctrl.set_flux_values([0.0, 0.5])
    run_controller_to_completion(ctrl)

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
    # predictor Service has no list row, so it is filtered out (no acquire needed).
    from zcu_tools.gui.app.autofluxdep.nodes.spec import Dependency

    ctrl = build_core()
    ctrl.add_node(make_builder("qubit_freq", requires=(Dependency("predict_freq"),)))
    ctrl.add_node(_make_filling_builder("mist"))
    ctrl.rename_node(1, "g_mist")
    ctrl.set_flux_values([0.0, 1.0])

    entered = []
    ctrl.bus.subscribe(NodeEnteredPayload, lambda p: entered.append(p.name))
    run_controller_to_completion(ctrl)
    assert "predictor" not in entered
    assert set(entered) == {"qubit_freq", "g_mist"}

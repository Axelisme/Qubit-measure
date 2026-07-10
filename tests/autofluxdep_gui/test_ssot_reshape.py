"""Phase 160b — node-param SSOT physicalisation into PlacedNode.schema.

The placement no longer carries a sparse ``params`` dict / ``base_params``; it
owns a per-placement ``NodeCfgSchema`` value tree (the SSOT). These tests pin the
reshape's behavioural contract:

- a placement has a ``schema`` (and *no* ``params`` / the Builder *no*
  ``base_params``);
- a write through ``set_field`` / ``set_node_params`` reflects in the lowered
  knobs (``schema.lower``), i.e. the SSOT is the schema, not a side dict;
- two placements of the same Builder hold *independent* schemas;
- the run-time read-only lock keeps the form's values visible but uneditable.

The make_cfg golden equivalence (defaults reproduce the prototype cfg) lives in
``test_cfg_schema.py``; this file pins the *SSOT location* rather than the cfg
values.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, PlacedNode
from zcu_tools.gui.app.autofluxdep.nodes.qubit_freq import QubitFreqBuilder
from zcu_tools.gui.cfg import CenteredSweepValue


def test_placement_holds_schema_not_params():
    node = PlacedNode(builder=QubitFreqBuilder())
    expected_reps = QubitFreqBuilder().make_default_schema().lower(None)["reps"]
    # the SSOT is the schema; the old sparse dict / base_params are gone
    assert hasattr(node, "schema")
    assert not hasattr(node, "params")
    assert not hasattr(Builder, "base_params")
    # the schema is seeded with the Builder's declared defaults
    assert node.schema.lower(None)["reps"] == expected_reps


def test_overrides_seed_the_schema_at_construction():
    node = PlacedNode(builder=QubitFreqBuilder(), overrides={"reps": 250})
    assert node.schema.lower(None)["reps"] == 250
    # an unknown override key fast-fails (a placement param with no declared knob)
    with pytest.raises(KeyError, match="Unknown node param"):
        PlacedNode(builder=QubitFreqBuilder(), overrides={"bogus": 1})


def test_set_field_reflects_in_lowered_knobs():
    node = PlacedNode(builder=QubitFreqBuilder())
    node.schema.set_field("reps", 333)
    node.schema.set_field("qub_gain", 0.7)
    knobs = node.schema.lower(None)
    assert knobs["reps"] == 333
    assert knobs["qub_gain"] == 0.7


def test_two_placements_have_independent_schemas():
    builder = QubitFreqBuilder()
    a = PlacedNode(builder=builder)
    b = PlacedNode(builder=builder)
    expected_reps = builder.make_default_schema().lower(None)["reps"]
    a.schema.set_field("reps", 111)
    # editing one placement does not bleed into the other (cloned default schemas)
    assert a.schema.lower(None)["reps"] == 111
    assert b.schema.lower(None)["reps"] == expected_reps


def test_set_node_params_writes_schema_and_bumps_version():
    ctrl = build_core()
    node = ctrl.add_node_by_type("qubit_freq")
    index = ctrl.state.nodes.index(node)
    before = ctrl.state.version.get("workflow")

    ctrl.set_node_params(index, {"qub_gain": "0.9"})

    assert node.schema.lower(None)["qub_gain"] == 0.9
    assert ctrl.state.version.get("workflow") > before


def test_set_node_params_accepts_sweep_value():
    ctrl = build_core()
    node = ctrl.add_node_by_type("qubit_freq")
    index = ctrl.state.nodes.index(node)

    ctrl.set_node_params(
        index, {"detune_sweep": CenteredSweepValue(center=0.0, span=10.0, expts=11)}
    )
    detune = node.schema.lower(None)["detune_sweep"]
    assert (float(detune.start), float(detune.stop), int(detune.expts)) == (
        -5.0,
        5.0,
        11,
    )

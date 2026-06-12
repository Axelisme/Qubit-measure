"""Phase 160b — NodeCfgForm: typed cfg form over a PlacedNode's schema SSOT.

The node detail pane's edit form is now the measure CfgFormWidget (reused via the
``cfg/form`` seam) bound to a ``SectionLiveField`` over the placement's schema.
These tests drive the LiveModel the form renders (the same surface a user spin /
line-edit drives) and assert:

- the rendered field set == the node's declared knob keys (spec keys);
- editing a scalar (int/float) / a sweep writes back into the placement schema
  SSOT via the controller's typed entry;
- clearing an optional scalar lowers it to None (omitted from the knobs);
- an invalid scalar is rejected (the model marks it invalid, the omitted/last-good
  value stays — no malformed write);
- the run-time read-only lock disables editing while keeping values visible.

No qtbot: the repo drives widgets directly under the autouse ``qapp`` fixture
(mirrors ``tests/gui/ui/test_cfg_form.py``).
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import DirectValue, SweepValue
from zcu_tools.gui.app.autofluxdep.ui.node_cfg_form import NodeCfgForm


@pytest.fixture
def ctrl_node():
    """A controller with one qubit_freq placement (its schema = the SSOT)."""
    ctrl = build_core()
    node = ctrl.add_node_by_type("qubit_freq")
    index = ctrl.state.nodes.index(node)
    yield ctrl, node, index
    ctrl._background_svc.quiesce()


def test_rendered_fields_match_spec_keys(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        # the LiveModel renders exactly the schema's declared knobs
        assert set(form._model.fields.keys()) == set(node.schema.keys)
    finally:
        form.teardown()


def test_edit_scalar_writes_back_to_schema(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        # drive the int knob the way a spin-box edit would (model.set_value →
        # schema_changed → controller.set_node_params → schema SSOT)
        form._model.fields["reps"].set_value(DirectValue(value=321))
        form._model.fields["qub_gain"].set_value(DirectValue(value=0.42))
        knobs = node.schema.lower(None)
        assert knobs["reps"] == 321 and isinstance(knobs["reps"], int)
        assert knobs["qub_gain"] == 0.42
    finally:
        form.teardown()


def test_edit_sweep_writes_back_to_schema(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        form._model.fields["detune_sweep"].set_value(
            SweepValue(start=-15.0, stop=25.0, expts=41)
        )
        detune = node.schema.lower(None)["detune_sweep"]
        assert (float(detune.start), float(detune.stop), int(detune.expts)) == (
            -15.0,
            25.0,
            41,
        )
    finally:
        form.teardown()


def test_optional_scalar_blank_lowers_to_none(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        # earlystop_snr is an optional FloatSpec: set then clear it; an unset
        # optional scalar lowers to an omitted key (None — no early-stop)
        form._model.fields["earlystop_snr"].set_value(DirectValue(value=50.0))
        assert node.schema.lower(None)["earlystop_snr"] == 50.0
        form._model.fields["earlystop_snr"].set_value(DirectValue(value=None))
        assert "earlystop_snr" not in node.schema.lower(None)
    finally:
        form.teardown()


def test_invalid_required_scalar_is_rejected(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        # qub_nqz is a required IntSpec; clearing it (None) is invalid. The model
        # marks the field invalid (the form surfaces it red) and the value tree now
        # carries an unset required leaf, so lowering Fast-Fails rather than
        # fabricating a default — no malformed-but-silent run cfg.
        form._model.fields["qub_nqz"].set_value(DirectValue(value=None))
        assert not form._model.fields["qub_nqz"].is_valid()
        assert not form._form.is_valid()  # the whole form reports invalid
        with pytest.raises(RuntimeError, match="qub_nqz"):
            node.schema.lower(None)
    finally:
        form.teardown()


def test_read_only_lock_keeps_values_visible(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        form.set_read_only(True)
        assert not form._form.isEnabled()  # editing disabled
        # the model (values) is untouched — "what this run used" stays visible
        reps_value = form._model.fields["reps"].get_value()
        assert isinstance(reps_value, DirectValue) and reps_value.value == 1000
        form.set_read_only(False)
        assert form._form.isEnabled()
    finally:
        form.teardown()

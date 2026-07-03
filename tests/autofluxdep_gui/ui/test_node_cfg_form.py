"""Phase 160b — NodeCfgForm: typed cfg form over a PlacedNode's schema SSOT.

The node detail pane's edit form is now the measure CfgFormWidget (reused via the
``cfg/form`` seam) bound to a ``SectionLiveField`` over the placement's schema.
These tests drive the LiveModel the form renders (the same surface a user spin /
line-edit drives) and assert:

- the rendered field set == the node's declared knob keys (spec keys);
- the rendered root field set == the node's declared section keys;
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

from typing import cast

import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import (
    DirectValue,
    FloatSpec,
    IntSpec,
    NodeCfgSchema,
    SweepValue,
    node_field,
    node_section,
    sectioned_node_schema,
)
from zcu_tools.gui.app.autofluxdep.cfg.form import SectionLiveField
from zcu_tools.gui.app.autofluxdep.nodes.builder import (
    Builder,
    Node,
    PlacedNode,
    RunEnv,
)
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot
from zcu_tools.gui.app.autofluxdep.ui.node_cfg_form import (
    NODE_FIELD_LABEL_MAX_WIDTH,
    NodeCfgForm,
)
from zcu_tools.gui.app.main.ui.fields.common import ElidedLabel


class _NoopNode(Node):
    def produce(self, snapshot: Snapshot) -> Patch:
        del snapshot
        return Patch()


class _SectionedBuilder(Builder):
    name = "sectioned"

    def make_default_schema(self) -> NodeCfgSchema:
        return sectioned_node_schema(
            (
                node_section(
                    "acquire",
                    "Acquisition",
                    node_field(
                        "reps",
                        "reps",
                        IntSpec("Reps"),
                        1000,
                    ),
                ),
                node_section(
                    "drive",
                    "Drive",
                    node_field(
                        "qub_gain",
                        "gain",
                        FloatSpec("Gain"),
                        0.05,
                    ),
                ),
            )
        )

    def build_node(self, env: RunEnv) -> Node:
        del env
        return _NoopNode()


@pytest.fixture
def ctrl_node():
    """A controller with one qubit_freq placement (its schema = the SSOT)."""
    ctrl = build_core()
    node = ctrl.add_node_by_type("qubit_freq")
    index = ctrl.state.nodes.index(node)
    yield ctrl, node, index
    ctrl._background_svc.quiesce()


def _section(form: NodeCfgForm, key: str) -> SectionLiveField:
    field = form._default_model.fields[key]
    assert isinstance(field, SectionLiveField)
    return cast(SectionLiveField, field)


def _subsection(section: SectionLiveField, key: str) -> SectionLiveField:
    field = section.fields[key]
    assert isinstance(field, SectionLiveField)
    return cast(SectionLiveField, field)


def _generation(form: NodeCfgForm) -> SectionLiveField:
    field = form._generation_model
    assert isinstance(field, SectionLiveField)
    return field


def test_rendered_fields_match_spec_keys(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        # Default cfg and generation behavior render as separate editor blocks; the
        # schema keeps logical keys.
        assert set(form._default_model.fields.keys()) == {
            "modules",
            "relax_delay",
            "reps",
            "rounds",
            "detune_sweep",
            "task",
        }
        qub_pulse = _subsection(_section(form, "modules"), "qub_pulse")
        assert set(qub_pulse.fields.keys()) == {
            "waveform",
            "ch",
            "nqz",
            "gain",
            "length",
        }
        assert qub_pulse.fields["ch"].spec.optional is False
        assert qub_pulse.fields["nqz"].spec.choices == [1, 2]
        assert "qub_gain" not in form._default_model.fields
        assert set(_generation(form).fields.keys()) == {"drive_gain_mode"}
        assert set(node.schema.keys) == {
            "detune_sweep",
            "reps",
            "rounds",
            "relax_delay",
            "earlystop_snr",
            "qub_waveform",
            "qub_ch",
            "qub_nqz",
            "qub_gain",
            "qub_length",
            "drive_gain_mode",
        }
    finally:
        form.teardown()


def test_field_labels_use_autofluxdep_width(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        labels = form.findChildren(ElidedLabel)
        assert labels
        assert all(
            label.maximumWidth() == NODE_FIELD_LABEL_MAX_WIDTH for label in labels
        )
        assert any(label.toolTip() == "earlystop_snr:" for label in labels)
    finally:
        form.teardown()


def test_edit_scalar_writes_back_to_schema(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        # drive the int knob the way a spin-box edit would (model.set_value →
        # schema_changed → controller.set_node_params → schema SSOT)
        form._default_model.fields["reps"].set_value(DirectValue(value=321))
        qub_pulse = _subsection(_section(form, "modules"), "qub_pulse")
        qub_pulse.fields["gain"].set_value(DirectValue(value=0.42))
        _generation(form).fields["drive_gain_mode"].set_value(
            DirectValue(value="fixed")
        )
        knobs = node.schema.lower(None)
        assert knobs["reps"] == 321 and isinstance(knobs["reps"], int)
        assert knobs["qub_gain"] == 0.42
        assert knobs["drive_gain_mode"] == "fixed"
    finally:
        form.teardown()


def test_edit_sweep_writes_back_to_schema(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        form._default_model.fields["detune_sweep"].set_value(
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


def test_sectioned_form_commit_projects_to_logical_keys(qapp):
    ctrl = build_core()
    node = PlacedNode(builder=_SectionedBuilder())
    ctrl.state.nodes.append(node)
    index = ctrl.state.nodes.index(node)
    form = NodeCfgForm(ctrl, node, index)
    try:
        assert set(form._default_model.fields) == {"acquire", "drive"}
        assert form._generation_model is None
        assert node.schema.keys == ("reps", "qub_gain")

        _section(form, "acquire").fields["reps"].set_value(DirectValue(value=432))
        _section(form, "drive").fields["gain"].set_value(DirectValue(value=0.25))

        knobs = node.schema.lower(None)
        assert knobs["reps"] == 432
        assert knobs["qub_gain"] == 0.25
    finally:
        form.teardown()
        ctrl._background_svc.quiesce()


def test_optional_scalar_blank_lowers_to_none(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        # earlystop_snr is an optional FloatSpec: set then clear it; an unset
        # optional scalar lowers to an omitted key (None — no early-stop)
        earlystop_snr = _section(form, "task").fields["earlystop_snr"]
        earlystop_snr.set_value(DirectValue(value=50.0))
        assert node.schema.lower(None)["earlystop_snr"] == 50.0
        earlystop_snr.set_value(DirectValue(value=None))
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
        nqz = _subsection(_section(form, "modules"), "qub_pulse").fields["nqz"]
        nqz.set_value(DirectValue(value=None))
        assert not nqz.is_valid()
        assert not form._default_form.is_valid()  # the whole default block is invalid
        with pytest.raises(RuntimeError, match="modules\\.qub_pulse\\.nqz"):
            node.schema.lower(None)
    finally:
        form.teardown()


def test_read_only_lock_keeps_values_visible(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        form.set_read_only(True)
        assert not form._default_form.isEnabled()  # editing disabled
        assert (
            form._generation_form is not None and not form._generation_form.isEnabled()
        )
        # the model (values) is untouched — "what this run used" stays visible
        reps_value = form._default_model.fields["reps"].get_value()
        assert isinstance(reps_value, DirectValue) and reps_value.value == 1000
        form.set_read_only(False)
        assert form._default_form.isEnabled()
        assert form._generation_form is not None and form._generation_form.isEnabled()
    finally:
        form.teardown()

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

from typing import Any, cast

import pytest
from qtpy.QtWidgets import QGroupBox, QVBoxLayout  # type: ignore[attr-defined]
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.cfg import (
    CenteredSweepValue,
    DirectValue,
    FloatSpec,
    IntSpec,
    NodeCfgSchema,
    OverridePath,
    OverridePlan,
    ScalarSpec,
)
from zcu_tools.gui.app.autofluxdep.cfg.form import (
    CfgFormWidget,
    FieldDecorationPatch,
    LiveModelEnv,
    SectionLiveField,
)
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
from zcu_tools.gui.session.events import SessionEvent

from .._helpers import node_field, node_section, sectioned_node_schema


class _NoopNode(Node):
    def produce(self, snapshot: Snapshot) -> Patch:
        del snapshot
        return Patch()


class _SectionedBuilder(Builder):
    name = "sectioned"

    def make_default_schema(self, ctx: Any | None = None) -> NodeCfgSchema:
        del ctx
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


class _InitialOverrideBuilder(_SectionedBuilder):
    name = "initial_override"

    def override_plan(self, schema: NodeCfgSchema) -> OverridePlan:
        del schema
        return OverridePlan(
            (
                OverridePath(
                    "acquire.reps",
                    "after_first_point",
                    "generation.synthetic",
                    "later points replace acquisition reps",
                ),
            )
        )


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


def _ref_subsection(section: SectionLiveField, key: str) -> SectionLiveField:
    field = section.fields[key]
    sub_field = getattr(field, "sub_field", None)
    assert isinstance(sub_field, SectionLiveField)
    return sub_field


def _generation(form: NodeCfgForm) -> SectionLiveField:
    field = form._generation_model
    assert isinstance(field, SectionLiveField)
    return field


def _generation_group(form: NodeCfgForm, key: str) -> SectionLiveField:
    return _subsection(_generation(form), key)


def _generation_leaf(form: NodeCfgForm, key: str) -> Any:
    for group in _generation(form).fields.values():
        if isinstance(group, SectionLiveField) and key in group.fields:
            return group.fields[key]
    raise AssertionError(f"generation leaf {key!r} not found")


def _field_labels(section: SectionLiveField) -> dict[str, str]:
    return {key: cast(Any, field).spec.label for key, field in section.fields.items()}


def _rendered_generation_paths(form: NodeCfgForm) -> set[str]:
    generation_form = form._generation_form
    assert generation_form is not None
    return set(generation_form.decoration_paths())


class _DecorationProvider:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def decoration_for(
        self, path: str, spec: object, value: object
    ) -> FieldDecorationPatch | None:
        del spec, value
        self.seen.append(path)
        if path == "acquire.reps":
            return FieldDecorationPatch(
                enabled=False,
                tone="warning",
                badge="runtime",
                tooltip="Generated at run time",
            )
        return None


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
            "sweep",
        }
        assert _field_labels(_section(form, "modules"))["qub_pulse"] == "Probe Pulse"
        qub_pulse = _ref_subsection(_section(form, "modules"), "qub_pulse")
        assert {"waveform", "ch", "nqz", "gain"}.issubset(qub_pulse.fields)
        ch_spec = qub_pulse.fields["ch"].spec
        nqz_spec = qub_pulse.fields["nqz"].spec
        assert isinstance(ch_spec, ScalarSpec)
        assert isinstance(nqz_spec, ScalarSpec)
        assert ch_spec.optional is False
        assert nqz_spec.choices == [1, 2]
        assert "qub_gain" not in form._default_model.fields
        assert set(_generation(form).fields.keys()) == {
            "acquisition",
            "drive_gain",
            "freq_recovery",
            "predictor_correction",
        }
        assert set(_generation_group(form, "acquisition").fields.keys()) == {
            "earlystop_snr",
            "acquire_retry",
        }
        assert set(_generation_group(form, "drive_gain").fields.keys()) == {
            "drive_gain_mode",
            "target_kappa",
            "qf_width_seed",
        }
        assert set(_generation_group(form, "freq_recovery").fields.keys()) == {
            "physical_recovery_mode",
        }
        assert set(_generation_group(form, "predictor_correction").fields.keys()) == {
            "pred_freq_correction_strategy",
            "pred_freq_correction_idw_k",
            "pred_freq_correction_idw_epsilon",
            "pred_freq_correction_decay_points",
        }
        assert _field_labels(_generation_group(form, "acquisition")) == {
            "earlystop_snr": "earlystop_snr",
            "acquire_retry": "retry",
        }
        assert _field_labels(_generation_group(form, "drive_gain")) == {
            "drive_gain_mode": "mode",
            "target_kappa": "target_kappa",
            "qf_width_seed": "initial_linewidth_mhz",
        }
        assert _field_labels(_generation_group(form, "freq_recovery")) == {
            "physical_recovery_mode": "mode",
        }
        assert _field_labels(_generation_group(form, "predictor_correction")) == {
            "pred_freq_correction_strategy": "strategy",
            "pred_freq_correction_idw_k": "idw_k",
            "pred_freq_correction_idw_epsilon": "idw_epsilon",
            "pred_freq_correction_decay_points": "decay_points",
        }
        generation_form = form._generation_form
        assert generation_form is not None
        assert (
            generation_form.decoration_for_path("drive_gain.qf_width_seed").tooltip
            == "Initial linewidth before measured feedback exists."
        )
        assert set(node.schema.keys) == {
            "detune_sweep",
            "reps",
            "rounds",
            "relax_delay",
            "earlystop_snr",
            "acquire_retry",
            "qub_pulse",
            "readout",
            "qub_ch",
            "qub_nqz",
            "qub_gain",
            "qub_length",
            "drive_gain_mode",
            "target_kappa",
            "qf_width_seed",
            "physical_recovery_mode",
            "pred_freq_correction_strategy",
            "pred_freq_correction_idw_k",
            "pred_freq_correction_idw_epsilon",
            "pred_freq_correction_decay_points",
        }
    finally:
        form.teardown()


def test_generation_choices_render_only_active_strategy_fields(ctrl_node, qapp):
    del qapp
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        paths = _rendered_generation_paths(form)
        assert "predictor_correction.pred_freq_correction_strategy" in paths
        assert "predictor_correction.pred_freq_correction_idw_k" in paths
        assert "predictor_correction.pred_freq_correction_idw_epsilon" in paths
        assert "predictor_correction.pred_freq_correction_decay_points" in paths
        assert "freq_recovery.physical_recovery_mode" in paths
        assert "drive_gain.target_kappa" in paths

        strategy = _generation_group(form, "predictor_correction").fields[
            "pred_freq_correction_strategy"
        ]
        strategy.set_value(DirectValue(value="last_good"))

        paths = _rendered_generation_paths(form)
        assert "predictor_correction.pred_freq_correction_strategy" in paths
        assert "predictor_correction.pred_freq_correction_idw_k" not in paths
        assert "predictor_correction.pred_freq_correction_idw_epsilon" not in paths
        assert "predictor_correction.pred_freq_correction_decay_points" in paths

        strategy.set_value(DirectValue(value="off"))

        paths = _rendered_generation_paths(form)
        assert "predictor_correction.pred_freq_correction_strategy" in paths
        assert "predictor_correction.pred_freq_correction_idw_k" not in paths
        assert "predictor_correction.pred_freq_correction_decay_points" not in paths

        recovery_mode = _generation_group(form, "freq_recovery").fields[
            "physical_recovery_mode"
        ]
        recovery_mode.set_value(DirectValue(value="off"))

        paths = _rendered_generation_paths(form)
        assert "freq_recovery.physical_recovery_mode" in paths

        drive_gain_mode = _generation_group(form, "drive_gain").fields[
            "drive_gain_mode"
        ]
        drive_gain_mode.set_value(DirectValue(value="fixed"))

        paths = _rendered_generation_paths(form)
        assert "drive_gain.drive_gain_mode" in paths
        assert "drive_gain.target_kappa" not in paths
        assert "drive_gain.qf_width_seed" not in paths
    finally:
        form.teardown()


def test_generation_choices_render_only_active_readout_search_fields(qapp):
    del qapp
    ctrl = build_core()
    node = ctrl.add_node_by_type("ro_optimize")
    index = ctrl.state.nodes.index(node)
    form = NodeCfgForm(ctrl, node, index)
    try:
        paths = _rendered_generation_paths(form)
        assert "relax.relax_delay_mode" in paths
        assert "relax.t1_seed_us" in paths
        assert "relax.relax_factor" in paths
        assert "freq_search.freq_range_mode" in paths
        assert "freq_search.freq_window_mode" in paths
        assert "freq_search.freq_half_width_mhz" in paths
        assert "gain_search.gain_range_mode" in paths
        assert "gain_search.gain_window_mode" in paths
        assert "gain_search.gain_half_width" in paths

        _generation_group(form, "freq_search").fields["freq_window_mode"].set_value(
            DirectValue(value="from_default_sweep")
        )

        paths = _rendered_generation_paths(form)
        assert "freq_search.freq_window_mode" in paths
        assert "freq_search.freq_half_width_mhz" not in paths

        _generation_group(form, "freq_search").fields["freq_range_mode"].set_value(
            DirectValue(value="fixed")
        )

        paths = _rendered_generation_paths(form)
        assert "freq_search.freq_range_mode" in paths
        assert "freq_search.freq_window_mode" not in paths
        assert "freq_search.freq_half_width_mhz" not in paths

        _generation_group(form, "gain_search").fields["gain_range_mode"].set_value(
            DirectValue(value="fixed")
        )

        paths = _rendered_generation_paths(form)
        assert "gain_search.gain_range_mode" in paths
        assert "gain_search.gain_window_mode" not in paths
        assert "gain_search.gain_half_width" not in paths

        _generation_group(form, "relax").fields["relax_delay_mode"].set_value(
            DirectValue(value="fixed")
        )

        paths = _rendered_generation_paths(form)
        assert "relax.relax_delay_mode" in paths
        assert "relax.t1_seed_us" not in paths
        assert "relax.relax_factor" not in paths
    finally:
        form.teardown()
        ctrl._background_svc.quiesce()


def test_cfg_form_decoration_provider_collects_nested_paths(qapp):
    ctrl = build_core()
    schema = _SectionedBuilder().make_default_schema()
    env = LiveModelEnv(ctrl=ctrl)
    model = SectionLiveField(schema.schema.spec, env, schema.schema.value)
    provider = _DecorationProvider()
    form = CfgFormWidget(
        decoration_provider=provider,
        field_label_max_width=500,
    )
    try:
        form.attach(model)

        assert set(form.decoration_paths()) == {
            "acquire",
            "acquire.reps",
            "drive",
            "drive.gain",
        }
        decoration = form.decoration_for_path("acquire.reps")
        assert decoration.enabled is False
        assert decoration.tone == "warning"
        assert decoration.badge == "runtime"
        assert decoration.tooltip == "Generated at run time"
        assert form.decoration_for_path("drive.gain").enabled is True
        runtime_labels = [
            label
            for label in form.findChildren(ElidedLabel)
            if label.toolTip() == "Generated at run time"
        ]
        assert runtime_labels
        assert getattr(runtime_labels[0], "_full_text") == "Reps [runtime]:"
        assert runtime_labels[0].isEnabled() is False
        with pytest.raises(KeyError, match="Unknown cfg field path"):
            form.decoration_for_path("missing")
    finally:
        form.detach()
        model.teardown()
        ctrl._background_svc.quiesce()


def test_node_cfg_form_renders_override_plan_badges_and_refreshes(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        freq_decoration = form._default_form.decoration_for_path(
            "modules.qub_pulse.freq"
        )
        assert freq_decoration.enabled is False
        assert freq_decoration.badge == "generated"

        gain_decoration = form._default_form.decoration_for_path(
            "modules.qub_pulse.gain"
        )
        assert gain_decoration.enabled is False
        assert gain_decoration.badge == "generated"

        readout_decoration = form._default_form.decoration_for_path("modules.readout")
        assert readout_decoration.enabled is True
        assert readout_decoration.badge == "template"
        assert "pulse_cfg.freq" in readout_decoration.tooltip
        assert "ro_cfg.ro_freq" in readout_decoration.tooltip
        assert "fallback" in readout_decoration.tooltip
        assert "ro_cfg.trig_offset" not in readout_decoration.tooltip

        readout_freq = form._default_form.decoration_for_path(
            "modules.readout.pulse_cfg.freq"
        )
        assert readout_freq.enabled is True
        assert readout_freq.badge == "fallback"

        readout_trigger = form._default_form.decoration_for_path(
            "modules.readout.ro_cfg.trig_offset"
        )
        assert readout_trigger.enabled is True
        assert readout_trigger.badge == ""

        _generation_leaf(form, "drive_gain_mode").set_value(DirectValue(value="fixed"))

        refreshed = form._default_form.decoration_for_path("modules.qub_pulse.gain")
        assert refreshed.enabled is True
        assert refreshed.badge == ""
    finally:
        form.teardown()


def test_node_cfg_form_renders_initial_override_badge(qapp):
    ctrl = build_core()
    node = ctrl.add_node(_InitialOverrideBuilder())
    index = ctrl.state.nodes.index(node)
    form = NodeCfgForm(ctrl, node, index)
    try:
        decoration = form._default_form.decoration_for_path("acquire.reps")
        assert decoration.enabled is True
        assert decoration.tone == "warning"
        assert decoration.badge == "initial"
        assert "Initial value is used at flux point 0" in decoration.tooltip

        initial_labels = [
            label
            for label in form.findChildren(ElidedLabel)
            if label.toolTip() == decoration.tooltip
        ]
        assert initial_labels
        assert getattr(initial_labels[0], "_full_text") == "Reps [initial]:"
        assert initial_labels[0].isEnabled() is True
    finally:
        form.teardown()
        ctrl._background_svc.quiesce()


def test_ro_optimize_previous_best_ranges_are_editable_initial_fields(qapp):
    ctrl = build_core()
    node = ctrl.add_node_by_type("ro_optimize")
    index = ctrl.state.nodes.index(node)
    form = NodeCfgForm(ctrl, node, index)
    try:
        freq_decoration = form._default_form.decoration_for_path("sweep.freq")
        assert freq_decoration.enabled is True
        assert freq_decoration.badge == "initial"
        assert "Initial value is used at flux point 0" in freq_decoration.tooltip

        gain_decoration = form._default_form.decoration_for_path("sweep.gain")
        assert gain_decoration.enabled is True
        assert gain_decoration.badge == "initial"
        assert "Initial value is used at flux point 0" in gain_decoration.tooltip

        _generation_group(form, "freq_search").fields["freq_range_mode"].set_value(
            DirectValue(value="fixed")
        )
        assert form._default_form.decoration_for_path("sweep.freq").badge == ""

        _generation_group(form, "gain_search").fields["gain_range_mode"].set_value(
            DirectValue(value="fixed")
        )
        assert form._default_form.decoration_for_path("sweep.gain").badge == ""
    finally:
        form.teardown()
        ctrl._background_svc.quiesce()


def test_lenrabi_auto_sweep_marks_only_stop_generated(qapp):
    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.app.main.ui.fields.common import SweepWidget

    ctrl = build_core()
    node = ctrl.add_node_by_type("lenrabi")
    index = ctrl.state.nodes.index(node)
    form = NodeCfgForm(ctrl, node, index)
    try:
        length_decoration = form._default_form.decoration_for_path("sweep.length")
        assert length_decoration.enabled is True
        assert length_decoration.badge == ""

        start_decoration = form._default_form.decoration_for_path("sweep.length.start")
        assert start_decoration.enabled is True
        assert start_decoration.badge == ""

        stop_decoration = form._default_form.decoration_for_path("sweep.length.stop")
        assert stop_decoration.enabled is False
        assert stop_decoration.badge == "generated"
        assert "sweep stop is generated" in stop_decoration.tooltip

        sweep_widget = form._default_form.findChild(SweepWidget)
        assert sweep_widget is not None
        assert sweep_widget._start_widget.isEnabled() is True
        assert sweep_widget._stop_widget.isEnabled() is False
        assert sweep_widget._expts.isEnabled() is True
        labels = {
            label.text(): label.toolTip() for label in sweep_widget.findChildren(QLabel)
        }
        assert labels["stop [generated]"] == stop_decoration.tooltip

        _generation_group(form, "sweep").fields["sweep_range_mode"].set_value(
            DirectValue(value="fixed")
        )

        refreshed = form._default_form.decoration_for_path("sweep.length.stop")
        assert refreshed.enabled is True
        assert refreshed.badge == ""
        refreshed_sweep_widget = form._default_form.findChild(SweepWidget)
        assert refreshed_sweep_widget is not None
        assert refreshed_sweep_widget._stop_widget.isEnabled() is True
        refreshed_labels = {
            label.text(): label.toolTip()
            for label in refreshed_sweep_widget.findChildren(QLabel)
        }
        assert "stop" in refreshed_labels
        assert "stop [generated]" not in refreshed_labels
    finally:
        form.teardown()
        ctrl._background_svc.quiesce()


def test_field_labels_use_autofluxdep_width(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        labels = form.findChildren(ElidedLabel)
        assert labels
        constrained = [
            label
            for label in labels
            if label.maximumWidth() == NODE_FIELD_LABEL_MAX_WIDTH
        ]
        assert constrained
        assert any(
            getattr(label, "_full_text") == "earlystop_snr:"
            and "completed-round SNR" in label.toolTip()
            for label in constrained
        )
    finally:
        form.teardown()


def test_default_and_generation_blocks_share_vertical_space(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        layout = cast(QVBoxLayout, form.layout())
        assert layout is not None
        assert layout.count() == 2
        assert layout.stretch(0) == 1
        assert layout.stretch(1) == 1

        groups = form.findChildren(QGroupBox)
        assert {group.title() for group in groups} == {
            "Default cfg",
            "Generation overrides",
        }
        assert all(group.maximumHeight() > 10_000 for group in groups)
        assert all(group.minimumHeight() == 0 for group in groups)
    finally:
        form.teardown()


def test_session_refresh_updates_split_live_models(ctrl_node, qapp, monkeypatch):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    seen: list[tuple[str, object]] = []
    try:
        generation = form._generation_model
        assert generation is not None
        monkeypatch.setattr(
            form._default_model,
            "refresh_external",
            lambda event: seen.append(("default", event)),
        )
        monkeypatch.setattr(
            generation,
            "refresh_external",
            lambda event: seen.append(("generation", event)),
        )

        form.refresh_external(SessionEvent.CONTEXT_SWITCHED)

        assert seen == [
            ("default", SessionEvent.CONTEXT_SWITCHED),
            ("generation", SessionEvent.CONTEXT_SWITCHED),
        ]
    finally:
        form.teardown()


def test_edit_scalar_writes_back_to_schema(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    form = NodeCfgForm(ctrl, node, index)
    try:
        # drive the int knob the way a spin-box edit would (model.set_value →
        # schema_changed → controller.set_node_params → schema SSOT)
        form._default_model.fields["reps"].set_value(DirectValue(value=321))
        qub_pulse = _ref_subsection(_section(form, "modules"), "qub_pulse")
        qub_pulse.fields["gain"].set_value(DirectValue(value=0.42))
        _generation_leaf(form, "drive_gain_mode").set_value(DirectValue(value="fixed"))
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
        _section(form, "sweep").fields["freq"].set_value(
            CenteredSweepValue(center=0.0, span=40.0, expts=41)
        )
        detune = node.schema.lower(None)["detune_sweep"]
        assert (float(detune.start), float(detune.stop), int(detune.expts)) == (
            -20.0,
            20.0,
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
        earlystop_snr = _generation_leaf(form, "earlystop_snr")
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
        nqz = _ref_subsection(_section(form, "modules"), "qub_pulse").fields["nqz"]
        nqz.set_value(DirectValue(value=None))
        assert not nqz.is_valid()
        assert not form._default_form.is_valid()  # the whole default block is invalid
        with pytest.raises(RuntimeError, match="modules\\.qub_pulse\\.nqz"):
            node.schema.lower(None)
    finally:
        form.teardown()


def test_read_only_lock_keeps_values_visible(ctrl_node, qapp):
    ctrl, node, index = ctrl_node
    node.schema.set_field("reps", 321)
    expected_reps = node.schema.lower(None)["reps"]
    form = NodeCfgForm(ctrl, node, index)
    try:
        form.set_read_only(True)
        assert not form._default_form.isEnabled()  # editing disabled
        assert (
            form._generation_form is not None and not form._generation_form.isEnabled()
        )
        # the model (values) is untouched — "what this run used" stays visible
        reps_value = form._default_model.fields["reps"].get_value()
        assert isinstance(reps_value, DirectValue) and reps_value.value == expected_reps
        form.set_read_only(False)
        assert form._default_form.isEnabled()
        assert form._generation_form is not None and form._generation_form.isEnabled()
    finally:
        form.teardown()

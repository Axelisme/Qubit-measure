"""Tests — CfgFormWidget populate / read_values round-trip."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import (
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
)
from zcu_tools.gui.app.main.live_model import SectionLiveField, SweepLiveField
from zcu_tools.gui.event_bus import BaseEventBus as EventBus

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ctrl():
    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = MagicMock()
    c.get_current_ml.return_value = MagicMock()
    c.list_arb_waveforms.return_value = []
    return c


def _schema(spec_fields: dict, value_fields: dict) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields=spec_fields),
        value=CfgSectionValue(fields=value_fields),
    )


def _attach(w, schema: CfgSchema, ctrl):
    """Build a LiveModel from ``schema`` and attach the widget to it.

    Mirrors the production flow where the CfgEditorService owns the model and the
    widget ``attach``es (ADR-0008). The model is returned for tests that drive it
    directly (e.g. external refresh, which the service performs in production).
    """
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, SectionLiveField

    model = SectionLiveField(schema.spec, LiveModelEnv(ctrl=ctrl), schema.value)
    w.attach(model)
    return model


def _make_ctx():
    from zcu_tools.gui.app.main.adapter import ExpContext

    return ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None)


# ---------------------------------------------------------------------------
# schema_to_dict — SweepValue
# ---------------------------------------------------------------------------


def test_sweep_value_uses_expts_as_canonical():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=1.0, stop=2.0, expts=5, step=999.0)},
    )
    result = schema.to_raw_dict(None, ml)
    assert isinstance(result["f"], SweepCfg)
    assert result["f"].expts == 5
    assert result["f"].step == pytest.approx(0.25)


def test_sweep_value_step_mode():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    result = schema.to_raw_dict(None, ml)
    sweep = result["f"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.step == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# make_scalar_widget / read_scalar_widget
# ---------------------------------------------------------------------------


def test_scalar_int_widget_round_trip(qapp):
    from zcu_tools.gui.app.main.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="X", type=int)
    w = make_scalar_widget(spec, 42)
    assert read_scalar_widget(w, spec) == 42


def test_scalar_float_widget_round_trip(qapp):
    from zcu_tools.gui.app.main.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Pi", type=float)
    w = make_scalar_widget(spec, 3.14)
    assert read_scalar_widget(w, spec) == pytest.approx(3.14)


def test_scalar_bool_widget_round_trip(qapp):
    from zcu_tools.gui.app.main.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Flag", type=bool)
    w = make_scalar_widget(spec, True)
    assert read_scalar_widget(w, spec) is True


def test_scalar_choices_widget_round_trip(qapp):
    from zcu_tools.gui.app.main.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Model", type=str, choices=["hm", "t", "auto"])
    w = make_scalar_widget(spec, "hm")
    assert read_scalar_widget(w, spec) == "hm"


def test_dynamic_arb_waveform_data_choices(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox
    from zcu_tools.gui.app.main.live_model import ScalarLiveField
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    ctrl.list_arb_waveforms.return_value = ["asset_a", "asset_b"]
    schema = _schema(
        {
            "data": ScalarSpec(
                label="Data key",
                type=str,
                required=True,
                choices_source="arb_waveforms",
            )
        },
        {"data": DirectValue(None)},
    )
    w = CfgFormWidget()
    model = _attach(w, schema, ctrl)

    combo = w.findChild(QComboBox)
    assert combo is not None
    assert [combo.itemText(i) for i in range(combo.count())] == ["asset_a", "asset_b"]
    assert combo.currentIndex() == -1
    assert not w.is_valid()

    combo.setCurrentIndex(1)

    field = cast(ScalarLiveField, model.fields["data"])
    value = field.get_value()
    assert isinstance(value, DirectValue)
    assert value.value == "asset_b"
    assert w.is_valid()


def test_arb_waveform_data_choice_allows_empty_initial_value(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox
    from zcu_tools.gui.app.main.live_model import ScalarLiveField
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    ctrl.list_arb_waveforms.return_value = ["asset_a"]
    schema = _schema(
        {
            "data": ScalarSpec(
                label="Data key",
                type=str,
                choices_source="arb_waveforms",
            )
        },
        {"data": DirectValue("")},
    )
    w = CfgFormWidget()
    model = _attach(w, schema, ctrl)

    combo = w.findChild(QComboBox)
    assert combo is not None
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "asset_a"]
    assert combo.currentIndex() == 0
    assert w.is_valid()

    field = cast(ScalarLiveField, model.fields["data"])
    value = field.get_value()
    assert isinstance(value, DirectValue)
    assert value.value == ""


def test_scalar_editable_false_widget_disabled(qapp):
    from zcu_tools.gui.app.main.ui.fields import make_scalar_widget

    spec = ScalarSpec(label="RO", type=float, editable=False)
    w = make_scalar_widget(spec, 1.0)
    assert not w.isEnabled()


def test_optional_scalar_widget_is_line_edit_empty_for_none(qapp):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.app.main.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Mixer freq", type=float, optional=True)
    # None → an empty QLineEdit (spinbox cannot show "unset"); reads back as None.
    w = make_scalar_widget(spec, "")
    assert isinstance(w, QLineEdit)
    assert w.text() == ""
    assert read_scalar_widget(w, spec) is None


def test_optional_scalar_widget_round_trips_value(qapp):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.app.main.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Mixer freq", type=float, optional=True)
    w = make_scalar_widget(spec, 5000.0)
    assert isinstance(w, QLineEdit)
    assert read_scalar_widget(w, spec) == pytest.approx(5000.0)
    # Clearing the field reads back as None (unset).
    w.setText("")
    assert read_scalar_widget(w, spec) is None


def test_grouped_field_renders_in_collapsed_subsection(qapp, ctrl):
    from zcu_tools.gui.app.main.adapter import make_default_value
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, SectionLiveField
    from zcu_tools.gui.app.main.ui.fields.containers import (
        SectionWidget,
        _CollapsibleSection,
    )

    spec = CfgSectionSpec(
        fields={
            "reps": ScalarSpec(label="Reps", type=int),
            "mixer_freq": ScalarSpec(
                label="Mixer freq", type=float, optional=True, group="Advanced"
            ),
        }
    )
    field = SectionLiveField(spec, LiveModelEnv(ctrl=ctrl), make_default_value(spec))
    w = SectionWidget(field, top_level=True)

    # Both fields get widgets (grouping is presentation-only, not a value change).
    assert set(w._child_widgets) == {"reps", "mixer_freq"}

    # A collapsed "Advanced" sub-section was created for the grouped field.
    advanced = [
        s
        for s in w.findChildren(_CollapsibleSection)
        if s._header_label is not None and "Advanced" in s._header_label.text()
    ]
    assert len(advanced) == 1
    assert advanced[0]._toggle_btn is not None
    assert not advanced[0]._toggle_btn.isChecked()  # collapsed by default


def test_scalar_widget_minimum_width_reduced(qapp):
    from zcu_tools.gui.app.main.ui.fields import make_scalar_widget

    spec = ScalarSpec(label="Name", type=str)
    w = make_scalar_widget(spec, "demo")
    assert w.minimumWidth() == 20


def test_scalar_widget_eval_mode_shows_resolved_ghost(qapp, ctrl):
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.app.main.ui.fields.common import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    field = ScalarLiveField(
        ScalarSpec(label="Freq", type=float),
        LiveModelEnv(ctrl=ctrl),
        initial_val=EvalValue("r_f"),
    )

    w = ScalarWidget(field)
    assert w._ghost is not None
    assert w._ghost.text() == "= 6000.0"


def test_scalar_widget_eval_mode_marks_unresolved_red(qapp, ctrl):
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.app.main.ui.fields.common import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    ctrl.get_current_md.return_value = MetaDict()
    field = ScalarLiveField(
        ScalarSpec(label="Freq", type=float),
        LiveModelEnv(ctrl=ctrl),
        initial_val=EvalValue("missing"),
    )

    w = ScalarWidget(field)
    assert w._ghost is not None
    assert w._ghost.text() == "= ?"
    assert "red" in w._ghost.styleSheet()


def test_scalar_widget_value_ref_text_resolves_on_space_in_eval_mode(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.app.main.ui.fields.common import ScalarWidget
    from zcu_tools.gui.session.value_lookup import ValueInfo
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    ctrl.read_value_source.return_value = (
        ValueInfo("device.flux.value", float, "device:flux"),
        0.125,
    )
    field = ScalarLiveField(
        ScalarSpec(label="Freq", type=float),
        LiveModelEnv(ctrl=ctrl),
        initial_val=EvalValue("r_f"),
    )
    w = ScalarWidget(field)
    edit = w.findChild(QLineEdit)
    assert edit is not None

    edit.setText("@{device.flux.value} ")
    edit.setCursorPosition(len(edit.text()))
    assert w._source_input is not None
    w._source_input._on_text_edited(edit.text())

    val = field.get_value()
    assert isinstance(val, EvalValue)
    assert val.expr == "0.125"
    assert edit.text() == "0.125"
    assert w._mode == "eval"
    ctrl.read_value_source.assert_called_once_with("device.flux.value")


def test_scalar_widget_eval_menu_extends_standard_line_edit_menu(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.app.main.ui.fields.common import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    field = ScalarLiveField(
        ScalarSpec(label="Freq", type=float),
        LiveModelEnv(ctrl=ctrl),
        initial_val=EvalValue("r_f"),
    )
    w = ScalarWidget(field)
    edit = w.findChild(QLineEdit)
    assert edit is not None

    menu, mode_action = w._build_context_menu(edit)
    action_texts = [action.text() for action in menu.actions()]

    assert mode_action is not None
    assert "Use direct value" in action_texts
    assert len(action_texts) > 1


def test_scalar_widget_unresolved_eval_can_switch_back_to_direct(qapp, ctrl):
    from qtpy.QtWidgets import QDoubleSpinBox  # type: ignore[attr-defined]
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.app.main.ui.fields.common import ScalarWidget
    from zcu_tools.meta_tool import MetaDict

    ctrl.get_current_md.return_value = MetaDict()
    field = ScalarLiveField(
        ScalarSpec(label="Freq", type=float),
        LiveModelEnv(ctrl=ctrl),
        initial_val=EvalValue("missing"),
    )
    w = ScalarWidget(field)

    field.set_value(None)

    value = field.get_value()
    assert isinstance(value, DirectValue)
    # unset scalar is value=None (ADR-0010) — no placeholder default
    assert value.value is None
    assert w._mode == "direct"
    spin = w.findChild(QDoubleSpinBox)
    assert spin is not None
    assert spin.value() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CfgFormWidget — populate and read_values / read_schema
# ---------------------------------------------------------------------------


def test_read_values_before_populate_raises(qapp):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_values()


def test_read_schema_before_populate_raises(qapp):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_schema()


def test_populate_scalar_fields_round_trip(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "reps": ScalarSpec(label="Reps", type=int),
            "freq": ScalarSpec(label="Freq", type=float),
        },
        {
            "reps": DirectValue(100),
            "freq": DirectValue(6.0),
        },
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    assert out.fields["reps"].value == 100  # type: ignore[union-attr]
    assert out.fields["freq"].value == pytest.approx(6.0)  # type: ignore[union-attr]


def test_cfg_form_reflects_model_external_refresh(qapp, ctrl):
    """The widget repaints when the (service-owned) model refreshes an EvalValue.

    Under ADR-0008 the service drives ``refresh_external`` on the model it owns;
    the attached widget reflects it for free via the model's bubbling on_change.
    Here we drive the model directly (the service-bus path is covered in
    test_cfg_editor) and assert the widget's read-back + schema_changed fire.
    """
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.session.events import SessionEvent
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    ctrl.get_current_md.return_value = md
    schema = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": EvalValue("r_f")},
    )
    w = CfgFormWidget()
    emitted = []
    w.schema_changed.connect(emitted.append)
    model = _attach(w, schema, ctrl)

    md.r_f = 6100.0
    model.refresh_external(SessionEvent.MD_CHANGED)  # the service does this in prod

    val = w.read_values().fields["freq"]
    assert isinstance(val, EvalValue)
    assert val.resolved == 6100.0
    assert emitted


def test_cfg_form_does_not_subscribe_bus(qapp, ctrl):
    """The widget no longer touches the EventBus (ADR-0008 moved refresh to the
    service). attach/detach must not register any bus subscription."""
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": DirectValue(6000.0)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    _attach(w, schema, ctrl)  # re-attach swaps models cleanly

    bus = ctrl.get_bus.return_value
    assert bus._subs == {} or all(not subs for subs in bus._subs.values())


def test_read_schema_returns_cfg_schema(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(10)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_schema()
    assert isinstance(out, CfgSchema)
    assert out.spec is schema.spec


def test_read_values_does_not_mutate_original(qapp, ctrl):
    from qtpy.QtWidgets import QSpinBox  # type: ignore[attr-defined]
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    spin = w.findChild(QSpinBox)
    assert spin is not None
    spin.setValue(999)

    out = w.read_values()
    assert out.fields["reps"].value == 999  # type: ignore[union-attr]
    assert schema.value.fields["reps"].value == 100  # type: ignore[union-attr]


def test_populate_sweep_field_round_trip(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=5.8, stop=6.2, expts=201)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.start == pytest.approx(5.8)
    assert sv.stop == pytest.approx(6.2)
    assert sv.expts == 201
    assert sv.step == pytest.approx(0.002)


def test_populate_sweep_field_step_preserved(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.1)


def test_sweep_widget_step_change_recomputes_expts_and_stop(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.common import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    sweep_widget._step.setValue(0.2)
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.expts == 6
    assert sv.stop == pytest.approx(1.0)
    assert sv.step == pytest.approx(0.2)


def test_sweep_widget_non_step_change_recomputes_step(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.common import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    sweep_widget._expts.setValue(5)
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.25)


def test_sweep_widget_start_supports_eval_mode(qapp, ctrl):
    from zcu_tools.gui.app.main.adapter import EvalValue
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.common import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    cast(SweepLiveField, sweep_widget._field).start_field.set_value(
        EvalValue(expr="r_f - 1", resolved=5999.0)
    )
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert isinstance(sv.start, EvalValue)
    assert sv.start.expr == "r_f - 1"


def test_populate_nested_section_round_trip(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "inner": CfgSectionSpec(
                fields={"gain": ScalarSpec(label="Gain", type=float)}
            )
        },
        {"inner": CfgSectionValue(fields={"gain": DirectValue(0.05)})},
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    inner = out.fields["inner"]
    assert isinstance(inner, CfgSectionValue)
    assert inner.fields["gain"].value == pytest.approx(0.05)  # type: ignore[union-attr]


def test_choice_section_renders_only_active_choice_fields(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half_width": ScalarSpec(label="Half width", type=float),
        "decay": ScalarSpec(label="Decay", type=float),
        "manual_value": ScalarSpec(label="Manual", type=float),
    }
    schema = _schema(
        {
            "search": ChoiceSectionSpec(
                label="Search",
                fields=fields,
                bindings=(
                    ChoiceBinding(
                        "mode",
                        {
                            "auto": CfgSectionSpec(
                                fields={
                                    "half_width": fields["half_width"],
                                    "decay": fields["decay"],
                                }
                            ),
                            "fixed": CfgSectionSpec(
                                fields={"manual_value": fields["manual_value"]}
                            ),
                        },
                    ),
                ),
            )
        },
        {
            "search": CfgSectionValue(
                fields={
                    "mode": DirectValue("auto"),
                    "half_width": DirectValue(1.0),
                    "decay": DirectValue(3.0),
                    "manual_value": DirectValue(2.0),
                }
            )
        },
    )
    w = CfgFormWidget()
    model = _attach(w, schema, ctrl)

    paths = set(w.decoration_paths())
    assert "search.mode" in paths
    assert "search.half_width" in paths
    assert "search.decay" in paths
    assert "search.manual_value" not in paths

    search = model.fields["search"]
    assert isinstance(search, SectionLiveField)
    search.fields["mode"].set_value(DirectValue("fixed"))

    paths = set(w.decoration_paths())
    assert "search.mode" in paths
    assert "search.half_width" not in paths
    assert "search.decay" not in paths
    assert "search.manual_value" in paths

    out = w.read_values().fields["search"]
    assert isinstance(out, CfgSectionValue)
    assert set(out.fields) == {"mode", "half_width", "decay", "manual_value"}


def test_choice_section_rejects_unknown_choice_fields():
    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto"]),
    }

    with pytest.raises(RuntimeError, match="unknown field"):
        ChoiceSectionSpec(
            fields=fields,
            bindings=(
                ChoiceBinding(
                    "mode",
                    {
                        "auto": CfgSectionSpec(
                            fields={"missing": ScalarSpec(label="Missing", type=float)}
                        )
                    },
                ),
            ),
        )


def test_literal_rows_are_hidden_regardless_of_key(qapp, ctrl):
    """All LiteralSpec fields render no widget — discriminators (type/style) and
    adapter lock_literal'd fields (e.g. a sweep-driven freq) alike."""
    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "type": LiteralSpec("pulse", label="Type"),
            # a non-type/style LiteralSpec: the lock_literal scenario
            "freq": LiteralSpec(0.0, label="Freq"),
            "waveform": CfgSectionSpec(
                label="Waveform",
                fields={
                    "style": LiteralSpec("gauss", label="Style"),
                    "sigma": ScalarSpec(label="Sigma", type=float),
                },
            ),
        },
        {
            "waveform": CfgSectionValue(fields={"sigma": DirectValue(1.2)}),
        },
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    labels = [label.text() for label in w.findChildren(QLabel)]
    assert "Type:" not in labels
    assert "Style:" not in labels
    assert "Freq:" not in labels  # locked field hidden too
    assert "Sigma:" in labels


def test_module_ref_toggle_sits_left_of_combo_and_controls_subsection(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox, QHBoxLayout, QToolButton
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget

    custom_spec = CfgSectionSpec(
        label="Pulse Shape",
        fields={
            "type": LiteralSpec("pulse"),
            "gain": ScalarSpec(label="Gain", type=float),
        },
    )
    schema = _schema(
        {"pulse": ModuleRefSpec(label="Pulse", allowed=[custom_spec])},
        {
            "pulse": ModuleRefValue(
                chosen_key="<Custom:Pulse Shape>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.25)}),
            )
        },
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    w.show()

    ref_widget = w.findChild(ModuleRefWidget)
    assert ref_widget is not None

    root_layout = ref_widget.layout()
    assert root_layout is not None
    header_item = root_layout.itemAt(0)
    assert header_item is not None
    header = cast(QHBoxLayout, header_item.layout())
    expand_item = header.itemAt(0)
    combo_item = header.itemAt(1)
    assert expand_item is not None
    assert combo_item is not None
    assert isinstance(expand_item.widget(), QToolButton)
    assert isinstance(combo_item.widget(), QComboBox)
    assert ref_widget._sub_container.isVisible() is True

    ref_widget._expand_btn.click()
    assert ref_widget._sub_container.isVisible() is False

    ref_widget._expand_btn.click()
    assert ref_widget._sub_container.isVisible() is True


def test_waveform_ref_toggle_sits_left_of_combo(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox, QHBoxLayout, QToolButton
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget

    custom_spec = CfgSectionSpec(
        label="Gaussian",
        fields={
            "style": LiteralSpec("gauss"),
            "sigma": ScalarSpec(label="Sigma", type=float),
        },
    )
    schema = _schema(
        {"waveform": WaveformRefSpec(label="Waveform", allowed=[custom_spec])},
        {
            "waveform": WaveformRefValue(
                chosen_key="<Custom:Gaussian>",
                value=CfgSectionValue(fields={"sigma": DirectValue(0.5)}),
            )
        },
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    ref_widget = w.findChild(ModuleRefWidget)
    assert ref_widget is not None
    root_layout = ref_widget.layout()
    assert root_layout is not None
    header_item = root_layout.itemAt(0)
    assert header_item is not None
    header = cast(QHBoxLayout, header_item.layout())
    expand_item = header.itemAt(0)
    combo_item = header.itemAt(1)
    assert expand_item is not None
    assert combo_item is not None
    assert isinstance(expand_item.widget(), QToolButton)
    assert isinstance(combo_item.widget(), QComboBox)


def test_cfg_form_does_not_wrap_module_ref_row(qapp, ctrl):
    from qtpy.QtWidgets import QFormLayout
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.containers import SectionWidget

    custom_spec = CfgSectionSpec(
        label="Long Custom Module Name",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    schema = _schema(
        {"pulse": ModuleRefSpec(label="Pulse", allowed=[custom_spec])},
        {
            "pulse": ModuleRefValue(
                chosen_key="<Custom:Long Custom Module Name>",
                value=CfgSectionValue(fields={"gain": DirectValue(0.25)}),
            )
        },
    )
    w = CfgFormWidget()
    w.resize(520, 480)
    _attach(w, schema, ctrl)

    section = w.findChild(SectionWidget)
    assert section is not None
    assert (
        section._container.form.rowWrapPolicy()
        == QFormLayout.RowWrapPolicy.DontWrapRows
    )


def test_populate_module_ref_field_round_trip(qapp, ctrl):
    from zcu_tools.gui.app.main.adapter import ModuleRefSpec, ModuleRefValue
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    allowed_spec = CfgSectionSpec(
        label="Pulse",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"mod": ModuleRefSpec(allowed=[allowed_spec], label="Module")}
        ),
        value=CfgSectionValue(
            fields={
                "mod": ModuleRefValue(
                    chosen_key="<Custom:Pulse>",
                    value=CfgSectionValue(fields={"gain": DirectValue(0.5)}),
                )
            }
        ),
    )
    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    mod = out.fields["mod"]
    assert isinstance(mod, ModuleRefValue)
    assert mod.chosen_key == "<Custom:Pulse>"
    assert mod.value.fields["gain"].value == pytest.approx(0.5)  # type: ignore[union-attr]


def test_populate_full_fake_freq_schema(qapp, ctrl):
    """Smoke test: FakeFreqAdapter default schema populates and round-trips."""
    from zcu_tools.experiment.v2_gui.adapters.fake.freq import FakeFreqAdapter
    from zcu_tools.gui.app.main.adapter import ModuleRefSpec
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget

    ctx = _make_ctx()
    schema = FakeFreqAdapter().make_default_cfg(ctx)

    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    out = w.read_values()

    for key in ("reps", "rounds", "sweep", "modules"):
        assert key in out.fields, f"missing key: {key}"
    # The simulated resonance moved to the adapter __init__ — no 'model' in cfg.
    assert "model" not in out.fields

    assert isinstance(out.fields["sweep"], CfgSectionValue)
    assert isinstance(out.fields["sweep"].fields["freq"], SweepValue)
    # modules is a CfgSectionValue with readout as ModuleRefValue
    modules_val = out.fields["modules"]
    assert isinstance(modules_val, CfgSectionValue)
    # Verify spec has ModuleRefSpec for readout
    modules_spec = schema.spec.fields["modules"]
    assert hasattr(modules_spec, "fields")
    readout_spec = modules_spec.fields["readout"]  # type: ignore[union-attr]
    assert isinstance(readout_spec, ModuleRefSpec)


def test_section_widget_no_header(qapp, ctrl):
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, SectionLiveField
    from zcu_tools.gui.app.main.ui.fields.containers import SectionWidget

    spec = CfgSectionSpec(
        label="TestSection",
        fields={"val": ScalarSpec(label="Val", type=int)},
    )
    val = CfgSectionValue(fields={"val": DirectValue(10)})
    field = SectionLiveField(spec, LiveModelEnv(ctrl=ctrl), val)

    # 1. no_header=False (default)
    w1 = SectionWidget(field, top_level=False, no_header=False)
    assert w1._container._toggle_btn is not None
    assert w1._container._header_label is not None

    # 2. no_header=True
    w2 = SectionWidget(field, top_level=False, no_header=True)
    assert w2._container._toggle_btn is None
    assert w2._container._header_label is None


def test_module_ref_widget_modified_label_and_no_overwrite(qapp, ctrl):
    from typing import Any, cast

    from qtpy.QtWidgets import QDoubleSpinBox
    from zcu_tools.gui.app.main.live_model import ModuleRefLiveField
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget
    from zcu_tools.meta_tool import ModuleLibrary

    ml = ModuleLibrary()
    ml.modules["my_pulse"] = cast(Any, {"type": "readout/direct", "ro_freq": 7000.0})
    ctrl.get_current_ml.return_value = ml

    from zcu_tools.gui.app.main.cfg_schemas import module_cfg_to_value

    lib_spec, lib_val = module_cfg_to_value(
        {"type": "readout/direct", "ro_freq": 7000.0}
    )
    schema = CfgSchema(
        spec=CfgSectionSpec(
            fields={"mod": ModuleRefSpec(allowed=[lib_spec], label="Module")}
        ),
        value=CfgSectionValue(
            fields={
                "mod": ModuleRefValue(
                    chosen_key="my_pulse",
                    value=lib_val,
                )
            }
        ),
    )

    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    w.show()

    ref_widget = w.findChild(ModuleRefWidget)
    assert ref_widget is not None
    assert ref_widget._expand_btn.isChecked() is False
    assert ref_widget._sub_container.isVisible() is False
    # 1. Initially unmodified
    assert ref_widget._combo.currentText() == "Lib: my_pulse"
    assert cast(ModuleRefLiveField, ref_widget._field).is_modified() is False

    # 2. Simulate user edits the inner value via spinbox
    spin = ref_widget.findChild(QDoubleSpinBox)
    assert spin is not None
    spin.setValue(8000.0)

    # Verify is_modified is True and combobox text has (modified) suffix
    assert cast(ModuleRefLiveField, ref_widget._field).is_modified() is True
    assert ref_widget._combo.currentText() == "Lib: my_pulse (modified)"

    # 3. Trigger MD_CHANGED and verify it does not overwrite modified value
    from zcu_tools.gui.session.events import MdChangedPayload
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    ctrl.get_bus.return_value.emit(MdChangedPayload(md=md))

    # Should stay as user modified (8000.0), not library default (7000.0)
    mod_val = w.read_values().fields["mod"]
    assert isinstance(mod_val, ModuleRefValue)
    freq_val = mod_val.value.fields["ro_freq"]
    assert isinstance(freq_val, DirectValue)
    assert freq_val.value == 8000.0

    # Verify both modified and clean items are present in combo box
    items_list = [
        ref_widget._combo.itemText(i) for i in range(ref_widget._combo.count())
    ]
    assert "Lib: my_pulse (modified)" in items_list
    assert "Revert to Lib: my_pulse" in items_list

    # 4. Select the clean item to revert modifications
    clean_idx = -1
    for i in range(ref_widget._combo.count()):
        if ref_widget._combo.itemText(i) == "Revert to Lib: my_pulse":
            clean_idx = i
            break
    assert clean_idx >= 0
    ref_widget._combo.setCurrentIndex(clean_idx)

    assert cast(ModuleRefLiveField, ref_widget._field).is_modified() is False
    mod_val2 = w.read_values().fields["mod"]
    assert isinstance(mod_val2, ModuleRefValue)
    freq_val2 = mod_val2.value.fields["ro_freq"]
    assert isinstance(freq_val2, DirectValue)
    assert freq_val2.value == 7000.0


# ---------------------------------------------------------------------------
# optional ModuleRefSpec UI (None option in combo)
# ---------------------------------------------------------------------------


def _make_optional_module_ref_schema(enabled: bool = True) -> CfgSchema:
    from zcu_tools.gui.app.main.adapter import ModuleRefSpec, ModuleRefValue

    inner_spec = CfgSectionSpec(
        label="Pulse",
        fields={"ch": ScalarSpec(label="Ch", type=int)},
    )
    outer_spec = CfgSectionSpec(
        fields={
            "module": ModuleRefSpec(
                allowed=[inner_spec], label="Module", optional=True
            ),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )
    if enabled:
        inner_val = CfgSectionValue(fields={"ch": DirectValue(0)})
        outer_val = CfgSectionValue(
            fields={
                "module": ModuleRefValue(chosen_key="<Custom:Pulse>", value=inner_val),
                "reps": DirectValue(10),
            }
        )
    else:
        outer_val = CfgSectionValue(fields={"reps": DirectValue(10)})
    return CfgSchema(spec=outer_spec, value=outer_val)


def test_optional_module_ref_renders_none_option(qapp, ctrl):
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget

    schema = _make_optional_module_ref_schema(enabled=True)
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    module_widgets = w.findChildren(ModuleRefWidget)
    assert len(module_widgets) >= 1
    mw = module_widgets[0]

    # None option should be at index 0
    assert mw._combo.itemData(0) == ModuleRefWidget._NONE_KEY


def test_optional_module_ref_select_none_disables_sub(qapp, ctrl):
    from zcu_tools.gui.app.main.live_model import ModuleRefLiveField
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget

    schema = _make_optional_module_ref_schema(enabled=True)
    w = CfgFormWidget()
    _attach(w, schema, ctrl)

    module_widgets = w.findChildren(ModuleRefWidget)
    mw = module_widgets[0]
    field = cast(ModuleRefLiveField, mw._field)

    assert field.is_enabled is True

    # Select the None option
    none_idx = mw._combo.findData(ModuleRefWidget._NONE_KEY)
    assert none_idx == 0
    mw._combo.setCurrentIndex(none_idx)

    assert field.is_enabled is False
    assert not mw._sub_container.isEnabled()


def test_module_ref_missing_library_shows_red_badge_and_invalid(qapp, ctrl):
    """A LINKED ref to an absent library key shows the red missing-ref badge and
    is invalid (recoverable — re-adding the name re-links it)."""
    from zcu_tools.gui.app.main.adapter import LiteralSpec
    from zcu_tools.gui.app.main.live_model import ModuleRefLiveField
    from zcu_tools.gui.app.main.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.app.main.ui.fields.containers import ModuleRefWidget
    from zcu_tools.meta_tool import ModuleLibrary

    pulse_spec = CfgSectionSpec(
        label="Pulse",
        fields={
            "type": LiteralSpec("pulse"),
            "gain": ScalarSpec(label="Gain", type=float),
        },
    )
    schema = _schema(
        {"pulse": ModuleRefSpec(label="Pulse", allowed=[pulse_spec])},
        {
            "pulse": ModuleRefValue(
                chosen_key="missing_pulse",
                value=CfgSectionValue(
                    fields={"type": DirectValue("pulse"), "gain": DirectValue(0.2)}
                ),
            )
        },
    )
    ctrl.get_current_ml.return_value = ModuleLibrary()

    w = CfgFormWidget()
    _attach(w, schema, ctrl)
    w.show()

    ref_widget = w.findChild(ModuleRefWidget)
    assert ref_widget is not None
    field = cast(ModuleRefLiveField, ref_widget._field)
    assert field.has_missing_library_ref() is True
    assert field.is_valid() is False
    assert ref_widget._missing_ref_hint.isVisible() is True
    assert "missing_pulse" in ref_widget._missing_ref_hint.text()

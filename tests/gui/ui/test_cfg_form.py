"""Tests — CfgFormWidget populate / read_values round-trip (Phase 19)."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    MultiSweepSpec,
    MultiSweepValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    schema_to_dict,
)
from zcu_tools.gui.event_bus import EventBus, MdChangedPayload
from zcu_tools.gui.live_model import SweepLiveField

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ctrl():
    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = MagicMock()
    c.get_current_ml.return_value = MagicMock()
    return c


def _schema(spec_fields: dict, value_fields: dict) -> CfgSchema:
    return CfgSchema(
        spec=CfgSectionSpec(fields=spec_fields),
        value=CfgSectionValue(fields=value_fields),
    )


def _make_ctx():
    from zcu_tools.gui.adapter import ExpContext

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
    result = schema_to_dict(schema, ml)
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
    result = schema_to_dict(schema, ml)
    sweep = result["f"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.step == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# make_scalar_widget / read_scalar_widget
# ---------------------------------------------------------------------------


def test_scalar_int_widget_round_trip(qapp):
    from zcu_tools.gui.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="X", type=int)
    w = make_scalar_widget(spec, 42)
    assert read_scalar_widget(w, spec) == 42


def test_scalar_float_widget_round_trip(qapp):
    from zcu_tools.gui.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Pi", type=float)
    w = make_scalar_widget(spec, 3.14)
    assert read_scalar_widget(w, spec) == pytest.approx(3.14)


def test_scalar_bool_widget_round_trip(qapp):
    from zcu_tools.gui.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Flag", type=bool)
    w = make_scalar_widget(spec, True)
    assert read_scalar_widget(w, spec) is True


def test_scalar_choices_widget_round_trip(qapp):
    from zcu_tools.gui.ui.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Model", type=str, choices=["hm", "t", "auto"])
    w = make_scalar_widget(spec, "hm")
    assert read_scalar_widget(w, spec) == "hm"


def test_scalar_editable_false_widget_disabled(qapp):
    from zcu_tools.gui.ui.fields import make_scalar_widget

    spec = ScalarSpec(label="RO", type=float, editable=False)
    w = make_scalar_widget(spec, 1.0)
    assert not w.isEnabled()


def test_scalar_widget_minimum_width_reduced(qapp):
    from zcu_tools.gui.ui.fields import make_scalar_widget

    spec = ScalarSpec(label="Name", type=str)
    w = make_scalar_widget(spec, "demo")
    assert w.minimumWidth() == 20


def test_scalar_widget_eval_mode_shows_resolved_ghost(qapp, ctrl):
    from zcu_tools.gui.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.ui.fields.common import ScalarWidget
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
    from zcu_tools.gui.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.ui.fields.common import ScalarWidget
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


def test_scalar_widget_eval_menu_extends_standard_line_edit_menu(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit  # type: ignore[attr-defined]
    from zcu_tools.gui.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.ui.fields.common import ScalarWidget
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
    from zcu_tools.gui.live_model import LiveModelEnv, ScalarLiveField
    from zcu_tools.gui.ui.fields.common import ScalarWidget
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
    assert value.is_unset is True
    assert value.value == pytest.approx(0.0)
    assert w._mode == "direct"
    spin = w.findChild(QDoubleSpinBox)
    assert spin is not None
    assert spin.value() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CfgFormWidget — populate and read_values / read_schema
# ---------------------------------------------------------------------------


def test_read_values_before_populate_raises(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_values()


def test_read_schema_before_populate_raises(qapp):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    w = CfgFormWidget()
    with pytest.raises(RuntimeError):
        w.read_schema()


def test_populate_scalar_fields_round_trip(qapp, ctrl):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

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
    w.populate(schema, ctrl)
    out = w.read_values()

    assert out.fields["reps"].value == 100  # type: ignore[union-attr]
    assert out.fields["freq"].value == pytest.approx(6.0)  # type: ignore[union-attr]


def test_cfg_form_refreshes_eval_field_from_bus(qapp, ctrl):
    from zcu_tools.gui.event_bus import GuiEvent
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
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
    w.populate(schema, ctrl)

    md.r_f = 6100.0
    ctrl.get_bus.return_value.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=md))

    val = w.read_values().fields["freq"]
    assert isinstance(val, EvalValue)
    assert val.resolved == 6100.0
    assert emitted


def test_cfg_form_unsubscribes_bus_on_repopulate(qapp, ctrl):
    from zcu_tools.gui.event_bus import GuiEvent
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"freq": ScalarSpec(label="Freq", type=float)},
        {"freq": DirectValue(6000.0)},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
    w.populate(schema, ctrl)

    bus = ctrl.get_bus.return_value
    assert len(bus._subs[GuiEvent.MD_CHANGED]) == 1
    assert len(bus._subs[GuiEvent.CONTEXT_SWITCHED]) == 1
    assert len(bus._subs[GuiEvent.ML_CHANGED]) == 1


def test_read_schema_returns_cfg_schema(qapp, ctrl):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(10)},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
    out = w.read_schema()
    assert isinstance(out, CfgSchema)
    assert out.spec is schema.spec


def test_read_values_does_not_mutate_original(qapp, ctrl):
    from qtpy.QtWidgets import QSpinBox  # type: ignore[attr-defined]
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)

    spin = w.findChild(QSpinBox)
    assert spin is not None
    spin.setValue(999)

    out = w.read_values()
    assert out.fields["reps"].value == 999  # type: ignore[union-attr]
    assert schema.value.fields["reps"].value == 100  # type: ignore[union-attr]


def test_populate_sweep_field_round_trip(qapp, ctrl):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=5.8, stop=6.2, expts=201)},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.start == pytest.approx(5.8)
    assert sv.stop == pytest.approx(6.2)
    assert sv.expts == 201
    assert sv.step == pytest.approx(0.1)


def test_populate_sweep_field_step_preserved(qapp, ctrl):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.1)


def test_sweep_widget_step_change_recomputes_expts_and_stop(qapp, ctrl):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.common import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
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
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.common import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    sweep_widget._expts.setValue(5)
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.25)


def test_sweep_widget_start_supports_eval_mode(qapp, ctrl):
    from zcu_tools.gui.adapter import EvalValue
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.common import SweepWidget

    schema = _schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
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
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "inner": CfgSectionSpec(
                fields={"gain": ScalarSpec(label="Gain", type=float)}
            )
        },
        {"inner": CfgSectionValue(fields={"gain": DirectValue(0.05)})},
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
    out = w.read_values()

    inner = out.fields["inner"]
    assert isinstance(inner, CfgSectionValue)
    assert inner.fields["gain"].value == pytest.approx(0.05)  # type: ignore[union-attr]


def test_literal_type_and_style_rows_are_hidden(qapp, ctrl):
    from qtpy.QtWidgets import QLabel
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "type": LiteralSpec("pulse", label="Type"),
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
    w.populate(schema, ctrl)

    labels = [label.text() for label in w.findChildren(QLabel)]
    assert "Type:" not in labels
    assert "Style:" not in labels
    assert "Sigma:" in labels


def test_module_ref_toggle_sits_left_of_combo_and_controls_subsection(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox, QHBoxLayout, QToolButton
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.containers import ModuleRefWidget

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
    w.populate(schema, ctrl)
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
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.containers import ModuleRefWidget

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
    w.populate(schema, ctrl)

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
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.containers import SectionWidget

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
    w.populate(schema, ctrl)

    section = w.findChild(SectionWidget)
    assert section is not None
    assert (
        section._container.form.rowWrapPolicy()
        == QFormLayout.RowWrapPolicy.DontWrapRows
    )


def test_populate_multi_sweep_round_trip(qapp, ctrl):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    schema = _schema(
        {
            "ms": MultiSweepSpec(
                axes={"x": SweepSpec(label="X"), "y": SweepSpec(label="Y")},
                label="Multi",
            )
        },
        {
            "ms": MultiSweepValue(
                axes={
                    "x": SweepValue(start=0.0, stop=1.0, expts=5),
                    "y": SweepValue(start=2.0, stop=3.0, expts=3),
                }
            )
        },
    )
    w = CfgFormWidget()
    w.populate(schema, ctrl)
    out = w.read_values()

    ms = out.fields["ms"]
    assert isinstance(ms, MultiSweepValue)
    assert ms.axes["x"].expts == 5
    assert ms.axes["y"].start == pytest.approx(2.0)


def test_populate_module_ref_field_round_trip(qapp, ctrl):
    from zcu_tools.gui.adapter import ModuleRefSpec, ModuleRefValue
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

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
    w.populate(schema, ctrl)
    out = w.read_values()

    mod = out.fields["mod"]
    assert isinstance(mod, ModuleRefValue)
    assert mod.chosen_key == "<Custom:Pulse>"
    assert mod.value.fields["gain"].value == pytest.approx(0.5)  # type: ignore[union-attr]


def test_populate_full_fake_freq_schema(qapp, ctrl):
    """Smoke test: FakeFreqAdapter default schema populates and round-trips."""
    from zcu_tools.experiment.v2_gui.adapters.onetone.fakefreq import FakeFreqAdapter
    from zcu_tools.gui.adapter import ModuleRefSpec
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget

    ctx = _make_ctx()
    schema = FakeFreqAdapter().make_default_cfg(ctx)

    w = CfgFormWidget()
    w.populate(schema, ctrl)
    out = w.read_values()

    for key in ("reps", "rounds", "sweep", "model", "modules"):
        assert key in out.fields, f"missing key: {key}"

    assert isinstance(out.fields["sweep"], CfgSectionValue)
    assert isinstance(out.fields["sweep"].fields["freq"], SweepValue)
    assert isinstance(out.fields["model"], CfgSectionValue)
    for model_key in ("freq", "Ql", "Qc_abs", "phi", "a0_abs", "edelay", "noise_scale"):
        assert model_key in out.fields["model"].fields, (
            f"missing model key: {model_key}"
        )
    # modules is a CfgSectionValue with readout as ModuleRefValue
    modules_val = out.fields["modules"]
    assert isinstance(modules_val, CfgSectionValue)
    # Verify spec has ModuleRefSpec for readout
    modules_spec = schema.spec.fields["modules"]
    assert hasattr(modules_spec, "fields")
    readout_spec = modules_spec.fields["readout"]  # type: ignore[union-attr]
    assert isinstance(readout_spec, ModuleRefSpec)


def test_section_widget_no_header(qapp, ctrl):
    from zcu_tools.gui.live_model import LiveModelEnv, SectionLiveField
    from zcu_tools.gui.ui.fields.containers import SectionWidget

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
    from zcu_tools.gui.live_model import ModuleRefLiveField
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.containers import ModuleRefWidget
    from zcu_tools.meta_tool import ModuleLibrary

    ml = ModuleLibrary()
    ml.modules["my_pulse"] = cast(Any, {"type": "readout/direct", "ro_freq": 7000.0})
    ctrl.get_current_ml.return_value = ml

    from zcu_tools.gui.cfg_schemas import module_cfg_to_value

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
    w.populate(schema, ctrl)
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
    from zcu_tools.gui.event_bus import GuiEvent, MdChangedPayload
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    ctrl.get_bus.return_value.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=md))

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
# Phase 55 — optional ModuleRefSpec UI (None option in combo)
# ---------------------------------------------------------------------------


def _make_optional_module_ref_schema(enabled: bool = True) -> CfgSchema:
    from zcu_tools.gui.adapter import ModuleRefSpec, ModuleRefValue

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
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.containers import ModuleRefWidget

    schema = _make_optional_module_ref_schema(enabled=True)
    w = CfgFormWidget()
    w.populate(schema, ctrl)

    module_widgets = w.findChildren(ModuleRefWidget)
    assert len(module_widgets) >= 1
    mw = module_widgets[0]

    # None option should be at index 0
    assert mw._combo.itemData(0) == ModuleRefWidget._NONE_KEY


def test_optional_module_ref_select_none_disables_sub(qapp, ctrl):
    from zcu_tools.gui.live_model import ModuleRefLiveField
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.containers import ModuleRefWidget

    schema = _make_optional_module_ref_schema(enabled=True)
    w = CfgFormWidget()
    w.populate(schema, ctrl)

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


def test_module_ref_missing_library_hint_visible_and_clear_on_switch(qapp, ctrl):
    from zcu_tools.gui.ui.cfg_form import CfgFormWidget
    from zcu_tools.gui.ui.fields.containers import ModuleRefWidget
    from zcu_tools.meta_tool import ModuleLibrary

    pulse_spec = CfgSectionSpec(
        label="Pulse",
        fields={"gain": ScalarSpec(label="Gain", type=float)},
    )
    schema = _schema(
        {"pulse": ModuleRefSpec(label="Pulse", allowed=[pulse_spec])},
        {
            "pulse": ModuleRefValue(
                chosen_key="missing_pulse",
                value=CfgSectionValue(fields={"gain": DirectValue(0.2)}),
            )
        },
    )
    ctrl.get_current_ml.return_value = ModuleLibrary()

    w = CfgFormWidget()
    w.populate(schema, ctrl)
    w.show()

    ref_widget = w.findChild(ModuleRefWidget)
    assert ref_widget is not None
    assert ref_widget._combo.currentText() == "Missing: missing_pulse"
    assert ref_widget._missing_ref_hint.isVisible() is True
    assert "missing_pulse" in ref_widget._missing_ref_hint.text()

    custom_idx = ref_widget._combo.findData("<Custom:Pulse>")
    assert custom_idx >= 0
    ref_widget._combo.setCurrentIndex(custom_idx)
    assert ref_widget._missing_ref_hint.isVisible() is False

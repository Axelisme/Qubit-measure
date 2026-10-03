"""Tests for shared CfgFormWidget populate and read-values behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QComboBox, QLabel, QLineEdit
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgNodeSpec,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChoiceBinding,
    ChoiceSectionSpec,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.cfg.binding import SectionField
from zcu_tools.gui.widgets.cfg import CfgFormWidget
from zcu_tools.gui.widgets.cfg.fields import CenteredSweepWidget

from tests.gui.widgets.cfg._form_support import (
    attach_draft,
    scalar_field,
    section_schema,
)
from tests.gui.widgets.cfg._tree_support import tree_item, tree_widget

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ctx():
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    return SessionEnv(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None)


# ---------------------------------------------------------------------------
# schema_to_dict — SweepValue
# ---------------------------------------------------------------------------


def test_sweep_value_uses_expts_as_canonical():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=1.0, stop=2.0, expts=5, step=999.0)},
    )
    result = schema_to_raw_dict(schema, None, ml)
    assert isinstance(result["f"], SweepCfg)
    assert result["f"].expts == 5
    assert result["f"].step == pytest.approx(0.25)


def test_sweep_value_step_mode():
    from zcu_tools.program.v2 import SweepCfg

    ml = MagicMock()
    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    result = schema_to_raw_dict(schema, None, ml)
    sweep = result["f"]
    assert isinstance(sweep, SweepCfg)
    assert sweep.step == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# make_scalar_widget / read_scalar_widget
# ---------------------------------------------------------------------------


def test_scalar_int_widget_round_trip(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="X", type=int)
    w = make_scalar_widget(spec, 42)
    assert read_scalar_widget(w, spec) == 42


def test_scalar_float_widget_round_trip(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Pi", type=float)
    w = make_scalar_widget(spec, 3.14)
    assert read_scalar_widget(w, spec) == pytest.approx(3.14)


def test_scalar_bool_widget_round_trip(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Flag", type=bool)
    w = make_scalar_widget(spec, True)
    assert read_scalar_widget(w, spec) is True


def test_scalar_choices_widget_round_trip(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Model", type=str, choices=["hm", "t", "auto"])
    w = make_scalar_widget(spec, "hm")
    assert read_scalar_widget(w, spec) == "hm"


def test_dynamic_arb_waveform_data_choices(qapp, ctrl):
    ctrl.arb_waveforms.list_data_keys.return_value = ["asset_a", "asset_b"]
    schema = section_schema(
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
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)

        combo = form.findChild(QComboBox)
        assert combo is not None
        choices = [combo.itemText(i) for i in range(combo.count())]
        assert choices == ["asset_a", "asset_b"]
        assert combo.currentIndex() == -1
        assert not form.is_valid()

        combo.setCurrentIndex(1)

        value = form.read_values().fields["data"]
        assert isinstance(value, DirectValue) and value.value == "asset_b"
        assert form.is_valid()
    finally:
        form.detach()
        form.close()
        draft.close()


def test_arb_waveform_data_choice_allows_empty_initial_value(qapp, ctrl):
    ctrl.arb_waveforms.list_data_keys.return_value = ["asset_a"]
    schema = section_schema(
        {
            "data": ScalarSpec(
                label="Data key",
                type=str,
                choices_source="arb_waveforms",
            )
        },
        {"data": DirectValue("")},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)

        combo = form.findChild(QComboBox)
        assert combo is not None
        assert [combo.itemText(i) for i in range(combo.count())] == ["", "asset_a"]
        assert combo.currentIndex() == 0
        assert form.is_valid()

        value = form.read_values().fields["data"]
        assert isinstance(value, DirectValue) and value.value == ""
    finally:
        form.detach()
        form.close()
        draft.close()


def test_dynamic_choice_renders_inactive_current_value_but_remains_invalid(qapp, ctrl):
    from qtpy.QtWidgets import QComboBox
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctrl.arb_waveforms.list_data_keys.return_value = []
    schema = section_schema(
        {
            "data": ScalarSpec(
                label="Data key",
                type=str,
                required=True,
                choices_source="arb_waveforms",
            )
        },
        {"data": DirectValue("retired_asset")},
    )
    form = CfgFormWidget()
    attach_draft(form, schema, ctrl)

    combo = form.findChild(QComboBox)
    assert combo is not None
    assert [combo.itemText(index) for index in range(combo.count())] == [
        "retired_asset"
    ]
    assert combo.currentText() == "retired_asset"
    assert not form.is_valid()


def test_scalar_editable_false_widget_disabled(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget

    spec = ScalarSpec(label="RO", type=float, editable=False)
    w = make_scalar_widget(spec, 1.0)
    assert not w.isEnabled()


def test_optional_scalar_widget_is_line_edit_empty_for_none(qapp):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Mixer freq", type=float, optional=True)
    # None → an empty QLineEdit (spinbox cannot show "unset"); reads back as None.
    w = make_scalar_widget(spec, "")
    assert isinstance(w, QLineEdit)
    assert w.text() == ""
    assert read_scalar_widget(w, spec) is None


def test_optional_scalar_widget_round_trips_value(qapp):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget, read_scalar_widget

    spec = ScalarSpec(label="Mixer freq", type=float, optional=True)
    w = make_scalar_widget(spec, 5000.0)
    assert isinstance(w, QLineEdit)
    assert read_scalar_widget(w, spec) == pytest.approx(5000.0)
    # Clearing the field reads back as None (unset).
    w.setText("")
    assert read_scalar_widget(w, spec) is None


def test_scalar_widget_minimum_width_reduced(qapp):
    from zcu_tools.gui.widgets.cfg.fields import make_scalar_widget

    spec = ScalarSpec(label="Name", type=str)
    w = make_scalar_widget(spec, "demo")
    assert w.minimumWidth() == 20


def test_scalar_widget_unresolved_eval_can_switch_back_to_direct(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg.fields import ScalarWidget
    from zcu_tools.resources.context import MetaDict

    ctrl.get_current_md.return_value = MetaDict()
    field = scalar_field(
        ctrl,
        ScalarSpec(label="Freq", type=float),
        EvalValue("missing"),
    )
    w = ScalarWidget(field)

    field.set_value(None)

    value = field.get_value()
    assert isinstance(value, DirectValue)
    # unset scalar is value=None (ADR-0010) — no placeholder default
    assert value.value is None
    entry = w.findChild(QLineEdit)
    assert entry is not None
    assert entry.text() == ""
    assert not field.is_valid()


# ---------------------------------------------------------------------------
# CfgFormWidget — populate and read_values / read_schema
# ---------------------------------------------------------------------------


def test_populate_scalar_fields_round_trip(qapp, ctrl):
    schema = section_schema(
        {
            "reps": ScalarSpec(label="Reps", type=int),
            "freq": ScalarSpec(label="Freq", type=float),
        },
        {"reps": DirectValue(100), "freq": DirectValue(6.0)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        values = form.read_values()
        reps = values.fields["reps"]
        freq = values.fields["freq"]
        assert isinstance(reps, DirectValue) and reps.value == 100
        assert isinstance(freq, DirectValue) and freq.value == pytest.approx(6.0)
    finally:
        form.detach()
        form.close()
        draft.close()


def test_set_editing_enabled_keeps_scroll_area_enabled(qapp, ctrl):
    from qtpy.QtCore import QEvent
    from qtpy.QtWidgets import QLineEdit, QScrollArea
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    w = CfgFormWidget()
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w.attach(draft)
    try:
        scroll = w.findChild(QScrollArea)
        assert scroll is not None
        assert w.isEnabled()
        assert scroll.isEnabled()

        tree = tree_widget(w)
        spin = tree.findChild(QLineEdit)
        assert spin is not None
        assert spin.isEnabled()

        w.set_editing_enabled(False)

        assert w.isEnabled()
        assert scroll.isEnabled()
        assert not tree.isEnabled()
        assert not spin.isEnabled()

        w.detach()
        qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        w.attach(draft)

        assert w.isEnabled()
        assert scroll.isEnabled()
        reattached_tree = tree_widget(w)
        assert not reattached_tree.isEnabled()
        reattached_spin = reattached_tree.findChild(QLineEdit)
        assert reattached_spin is not None
        assert not reattached_spin.isEnabled()

        w.set_editing_enabled(True)

        assert reattached_tree.isEnabled()
        assert reattached_spin.isEnabled()
    finally:
        w.detach()
        draft.close()


def test_read_schema_returns_cfg_schema(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(10)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    out = w.read_schema()
    assert isinstance(out, CfgSchema)
    assert out.spec is schema.spec


def test_read_values_does_not_mutate_original(qapp, ctrl):
    schema = section_schema(
        {"reps": ScalarSpec(label="Reps", type=int)},
        {"reps": DirectValue(100)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)

        entry = form.findChild(QLineEdit)
        assert entry is not None
        entry.setText("999")

        reps = form.read_values().fields["reps"]
        original_reps = schema.value.fields["reps"]
        assert isinstance(reps, DirectValue) and reps.value == 999
        assert isinstance(original_reps, DirectValue) and original_reps.value == 100
    finally:
        form.detach()
        form.close()
        draft.close()


def test_populate_sweep_field_round_trip(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=5.8, stop=6.2, expts=201)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.start == pytest.approx(5.8)
    assert sv.stop == pytest.approx(6.2)
    assert sv.expts == 201
    assert sv.step == pytest.approx(0.002)


def test_populate_centered_sweep_field_round_trip(qapp, ctrl):
    schema = section_schema(
        {
            "f": CenteredSweepSpec(
                label="Freq",
                center_editable=False,
                center_badge="generated",
                center_tooltip="Generated center",
            )
        },
        {"f": CenteredSweepValue(center=0.0, span=100.0, expts=201)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        sweep_widget = form.findChild(CenteredSweepWidget)
        assert sweep_widget is not None

        span_input = sweep_widget.findChild(QLineEdit, "span")
        points_input = sweep_widget.findChild(QLineEdit, "expts")
        assert span_input is not None and points_input is not None
        center_input = sweep_widget.findChild(QLineEdit)
        assert center_input is not None
        assert not center_input.isEnabled()
        labels = {label.text(): label for label in sweep_widget.findChildren(QLabel)}
        center_label = labels["center [generated]"]
        assert center_label.toolTip() == "Generated center"
        center_cell = center_label.parentWidget()
        span_cell = labels["span"].parentWidget()
        assert center_cell is not None and span_cell is not None
        pair_row = center_cell.parentWidget()
        assert pair_row is not None and pair_row is span_cell.parentWidget()
        form.show()
        qapp.processEvents()
        pair_row.resize(801, pair_row.sizeHint().height())
        qapp.processEvents()
        assert abs(center_cell.width() - span_cell.width()) <= 1
        assert center_cell.width() + span_cell.width() + 4 == pair_row.width()
        assert center_cell.geometry().united(span_cell.geometry()) == pair_row.rect()

        span_input.setText("120.0")
        points_input.setText("121")
        sv = form.read_values().fields["f"]
        assert isinstance(sv, CenteredSweepValue)
        assert sv.center == pytest.approx(0.0)
        assert sv.span == DirectValue(120.0, raw="120.0")
        assert sv.expts == DirectValue(121, raw="121")
        assert sv.step == pytest.approx(1.0)
        assert form.is_valid()

        span_input.setText("0.0")
        sv = form.read_values().fields["f"]
        assert isinstance(sv, CenteredSweepValue)
        assert isinstance(sv.span, DirectValue)
        assert (sv.span.value, sv.span.raw) == (None, "0.0")
        assert sv.span.error is not None
        assert span_input.text() == "0.0"
        assert not form.is_valid()
    finally:
        form.detach()
        form.close()
        draft.close()


def test_populate_sweep_field_step_preserved(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    out = w.read_values()

    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.1)


def test_sweep_widget_step_change_recomputes_expts_and_stop(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    entry = sweep_widget.findChild(QLineEdit, "step")
    assert entry is not None
    entry.setText("0.2")
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.expts == 6
    assert sv.stop == pytest.approx(1.0)
    assert sv.step == DirectValue(0.2, raw="0.2")


def test_sweep_widget_non_step_change_recomputes_step(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg import CfgFormWidget
    from zcu_tools.gui.widgets.cfg.fields import SweepWidget

    schema = section_schema(
        {"f": SweepSpec(label="Freq")},
        {"f": SweepValue(start=0.0, stop=1.0, expts=11, step=0.1)},
    )
    w = CfgFormWidget()
    attach_draft(w, schema, ctrl)
    sweep_widget = w.findChild(SweepWidget)
    assert sweep_widget is not None

    entry = sweep_widget.findChild(QLineEdit, "expts")
    assert entry is not None
    entry.setText("5")
    out = w.read_values()
    sv = out.fields["f"]
    assert isinstance(sv, SweepValue)
    assert sv.step == pytest.approx(0.25)


def test_populate_nested_section_round_trip(qapp, ctrl):
    schema = section_schema(
        {
            "inner": CfgSectionSpec(
                fields={"gain": ScalarSpec(label="Gain", type=float)}
            )
        },
        {"inner": CfgSectionValue(fields={"gain": DirectValue(0.05)})},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        inner = form.read_values().fields["inner"]
        assert isinstance(inner, CfgSectionValue)
        gain = inner.fields["gain"]
        assert isinstance(gain, DirectValue) and gain.value == pytest.approx(0.05)
    finally:
        form.detach()
        form.close()
        draft.close()


def test_choice_section_renders_only_active_choice_fields(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "half_width": ScalarSpec(label="Half width", type=float),
        "decay": ScalarSpec(label="Decay", type=float),
        "manual_value": ScalarSpec(label="Manual", type=float),
    }
    schema = section_schema(
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
    model = attach_draft(w, schema, ctrl)

    paths = set(w.decoration_paths())
    assert "search.mode" in paths
    assert "search.half_width" in paths
    assert "search.decay" in paths
    assert "search.manual_value" not in paths

    search = model.fields["search"]
    assert isinstance(search, SectionField)
    search.fields["mode"].set_value(DirectValue("fixed"))

    paths = set(w.decoration_paths())
    assert "search.mode" in paths
    assert "search.half_width" not in paths
    assert "search.decay" not in paths
    assert "search.manual_value" in paths

    out = w.read_values().fields["search"]
    assert isinstance(out, CfgSectionValue)
    assert set(out.fields) == {"mode", "half_width", "decay", "manual_value"}


def test_spec_tooltip_populates_decoration_and_provider_can_override(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )

    class TooltipProvider:
        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            del spec, value
            if path == "gain":
                return FieldDecorationPatch(tooltip="Provider tooltip")
            return None

    schema = section_schema(
        {
            "gain": ScalarSpec(
                label="Gain",
                type=float,
                tooltip="Spec tooltip",
            ),
            "window": SweepSpec(label="Window", tooltip="Sweep tooltip"),
        },
        {
            "gain": DirectValue(1.0),
            "window": SweepValue(start=0.0, stop=1.0, expts=11),
        },
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget(decoration_provider=TooltipProvider())
    try:
        form.attach(draft)
        before = form.read_values()
        assert tree_item(form, "gain").toolTip(0) == "Provider tooltip"
        assert tree_item(form, "window").toolTip(0) == "Sweep tooltip"

        form.set_decoration_provider(None)
        qapp.processEvents()

        assert tree_item(form, "gain").toolTip(0) == "Spec tooltip"
        assert tree_item(form, "window").toolTip(0) == "Sweep tooltip"
        assert form.read_values() == before
    finally:
        form.detach()
        form.close()
        draft.close()


def test_sweep_edge_decoration_disables_only_that_edge(qapp, ctrl):
    from qtpy.QtWidgets import QWidget
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )

    class StopGeneratedProvider:
        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            del spec, value
            if path == "window.stop":
                return FieldDecorationPatch(
                    enabled=False,
                    tone="muted",
                    badge="generated",
                    tooltip="Stop is generated",
                )
            return None

    schema = section_schema(
        {"window": SweepSpec(label="Window")},
        {"window": SweepValue(start=0.0, stop=10.0, expts=21)},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget(decoration_provider=StopGeneratedProvider())
    try:
        form.attach(draft)
        sweep = tree_widget(form).itemWidget(tree_item(form, "window"), 1)
        assert sweep is not None
        start = sweep.findChild(QWidget, "start")
        stop = sweep.findChild(QWidget, "stop")
        points = sweep.findChild(QLineEdit, "expts")
        assert start is not None and start.isEnabled()
        assert stop is not None and not stop.isEnabled()
        assert points is not None and points.isEnabled()
        labels = {label.text(): label.toolTip() for label in sweep.findChildren(QLabel)}
        assert labels["stop [generated]"] == "Stop is generated"

        editor = start.findChild(QLineEdit)
        assert editor is not None and editor.isEnabled()
        editor.setText("2.5")

        value = form.read_values().fields["window"]
        assert isinstance(value, SweepValue)
        assert isinstance(value.start, DirectValue) and value.start.value == 2.5
        assert value.stop == 10.0
        assert value.expts == 21
    finally:
        form.detach()
        form.close()
        draft.close()


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


def test_choice_section_unknown_selector_value_fast_fails(qapp, ctrl):
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    fields: dict[str, CfgNodeSpec] = {
        "mode": ScalarSpec(label="Mode", type=str, choices=["auto", "fixed"]),
        "auto_gain": ScalarSpec(label="Auto gain", type=float),
        "manual_gain": ScalarSpec(label="Manual gain", type=float),
    }
    schema = section_schema(
        {
            "search": ChoiceSectionSpec(
                label="Search",
                fields=fields,
                bindings=(
                    ChoiceBinding(
                        "mode",
                        {
                            "auto": CfgSectionSpec(
                                fields={"auto_gain": fields["auto_gain"]}
                            ),
                            "fixed": CfgSectionSpec(
                                fields={"manual_gain": fields["manual_gain"]}
                            ),
                        },
                    ),
                ),
            )
        },
        {
            "search": CfgSectionValue(
                fields={
                    "mode": DirectValue("unknown"),
                    "auto_gain": DirectValue(0.1),
                    "manual_gain": DirectValue(0.2),
                }
            )
        },
    )

    with pytest.raises(ValueError, match="unknown value 'unknown'"):
        attach_draft(CfgFormWidget(), schema, ctrl)


def test_literal_rows_are_hidden_regardless_of_key(qapp, ctrl):
    """Hidden literals stay in the snapshot while visible scalar inputs remain editable."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QTreeWidgetItemIterator
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    schema = section_schema(
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
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        tree = tree_widget(form)
        items = QTreeWidgetItemIterator(tree)
        paths: set[str] = set()
        while (item := items.value()) is not None:
            paths.add(item.data(0, Qt.ItemDataRole.UserRole))
            items += 1
        assert {"type", "freq", "waveform.style"}.isdisjoint(paths)

        widget = tree.itemWidget(tree_item(form, "waveform.sigma"), 1)
        assert widget is not None
        editor = widget.findChild(QLineEdit)
        assert editor is not None and editor.isEnabled()
        editor.setText("2.4")

        value = form.read_values()
        assert value.fields["type"] == DirectValue("pulse")
        assert value.fields["freq"] == DirectValue(0.0)
        waveform = value.fields["waveform"]
        assert isinstance(waveform, CfgSectionValue)
        assert waveform.fields["style"] == DirectValue("gauss")
        sigma = waveform.fields["sigma"]
        assert isinstance(sigma, DirectValue) and sigma.value == 2.4
    finally:
        form.detach()
        form.close()
        draft.close()


def test_literal_decoration_reveals_read_only_value(qapp, ctrl):
    from qtpy.QtWidgets import QLineEdit
    from zcu_tools.gui.widgets.cfg import (
        CfgFormWidget,
        FieldDecorationPatch,
    )

    class RevealLiteralProvider:
        def decoration_for(
            self, path: str, spec: object, value: object
        ) -> FieldDecorationPatch | None:
            del spec, value
            if path == "freq":
                return FieldDecorationPatch(
                    hidden=False,
                    enabled=False,
                    badge="generated",
                    tooltip="Generated at run time",
                )
            return None

    schema = section_schema(
        {"freq": LiteralSpec(0.0, label="Freq")},
        {},
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        form.attach(draft)
        assert form.read_values().fields["freq"] == DirectValue(0.0)

        form.set_decoration_provider(RevealLiteralProvider())
        qapp.processEvents()

        item = tree_item(form, "freq")
        assert "generated" in item.text(0)
        assert item.toolTip(0) == "Generated at run time"
        assert item.isDisabled()
        editor = tree_widget(form).itemWidget(item, 1)
        assert isinstance(editor, QLineEdit)
        assert editor.text() == "0.0"
        assert editor.isReadOnly()
        assert not editor.isEnabled()
        assert form.read_values().fields["freq"] == DirectValue(0.0)
    finally:
        form.detach()
        form.close()
        draft.close()


def test_populate_full_fake_freq_schema(qapp, ctrl):
    """The adapter's bound schema survives form attachment and readback."""
    from zcu_tools.experiment.v2_gui.measure.adapters.fake.freq import FakeFreqAdapter

    schema = FakeFreqAdapter().make_default_cfg(_make_ctx())
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    form = CfgFormWidget()
    try:
        # Binding resolves reference labels and expression errors before rendering.
        expected = draft.snapshot()
        form.attach(draft)

        assert form.read_values() == expected.value
        snapshot = form.read_schema()
        assert snapshot.spec is schema.spec
        assert snapshot.value == expected.value
    finally:
        form.detach()
        form.close()
        draft.close()


# ---------------------------------------------------------------------------
# optional ReferenceSpec UI (None option in combo)
# ---------------------------------------------------------------------------

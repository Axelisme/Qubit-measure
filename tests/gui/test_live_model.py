"""Tests for LiveModel reactive data layer."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
)
from zcu_tools.gui.event_bus import EventBus, GuiEvent
from zcu_tools.gui.live_model import (
    CallbackList,
    LiveModelEnv,
    ModuleRefLiveField,
    ScalarLiveField,
    SectionLiveField,
    SweepLiveField,
)
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.program.v2 import WaveformCfgFactory


@pytest.fixture()
def bus():
    return EventBus()


@pytest.fixture()
def ctrl(bus):
    c = MagicMock()
    c.get_bus.return_value = bus
    c.get_current_md.return_value = MagicMock()
    c.get_current_ml.return_value = MagicMock()
    return c


@pytest.fixture()
def env(bus, ctrl):
    return LiveModelEnv(ctrl=ctrl)


def test_scalar_field_reactivity(env):
    spec = ScalarSpec(label="Test", type=int)
    field = ScalarLiveField(spec, env, initial_val=DirectValue(10))

    cb = MagicMock()
    field.on_change.connect(cb)

    field.set_value(20)
    value = field.get_value()
    assert isinstance(value, DirectValue)
    assert value.value == 20
    assert cb.called


def test_section_field_propagation(env):
    spec = CfgSectionSpec(
        fields={
            "f1": ScalarSpec(label="F1", type=int),
            "f2": ScalarSpec(label="F2", type=float),
        }
    )
    initial_val = CfgSectionValue(
        fields={
            "f1": DirectValue(1),
            "f2": DirectValue(0.5),
        }
    )
    section = SectionLiveField(spec, env, initial_val=initial_val)

    cb = MagicMock()
    section.on_change.connect(cb)

    section.fields["f1"].set_value(10)

    # Check that section emitted on_change
    assert cb.called
    val = section.get_value()
    f1_val = cast(ScalarValue, val.fields["f1"])
    f2_val = cast(ScalarValue, val.fields["f2"])
    assert isinstance(f1_val, DirectValue)
    assert isinstance(f2_val, DirectValue)
    assert f1_val.value == 10
    assert f2_val.value == 0.5


def test_scalar_eval_field_resolves_from_md(env):
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    md.rf_w = 2.0
    env.ctrl.get_current_md.return_value = md

    spec = ScalarSpec(label="Freq", type=float)
    field = ScalarLiveField(spec, env, initial_val=EvalValue("r_f - 1.5 * rf_w"))

    val = field.get_value()
    assert isinstance(val, EvalValue)
    assert val.resolved == 5997.0
    assert field.is_valid() is True


def test_scalar_eval_field_refresh_updates_snapshot(env):
    from zcu_tools.meta_tool import MetaDict

    md = MetaDict()
    md.r_f = 6000.0
    env.ctrl.get_current_md.return_value = md

    spec = ScalarSpec(label="Freq", type=float)
    field = ScalarLiveField(spec, env, initial_val=EvalValue("r_f"))
    cb = MagicMock()
    field.on_change.connect(cb)

    md.r_f = 6100.0
    field.refresh_external(GuiEvent.MD_CHANGED)

    val = field.get_value()
    assert isinstance(val, EvalValue)
    assert val.resolved == 6100.0
    assert cb.called


def test_scalar_eval_field_invalid_expression_marks_invalid(env):
    from zcu_tools.meta_tool import MetaDict

    env.ctrl.get_current_md.return_value = MetaDict()
    field = ScalarLiveField(
        ScalarSpec(label="Freq", type=float),
        env,
        initial_val=EvalValue("missing_name"),
    )

    val = field.get_value()
    assert isinstance(val, EvalValue)
    assert val.resolved is None
    assert val.error
    assert field.is_valid() is False


def test_scalar_eval_unresolved_marks_invalid(env):
    field = ScalarLiveField(
        ScalarSpec(label="Freq", type=float),
        env,
        initial_val=EvalValue("missing_name"),
    )

    val = field.get_value()
    assert isinstance(val, EvalValue)
    assert val.resolved is None
    assert field.is_valid() is False


def test_callback_list_propagates_callback_exceptions():
    callbacks = CallbackList()
    callbacks.connect(MagicMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        callbacks.emit()


def test_sweep_live_field_rejects_wrong_value_type(env):
    field = SweepLiveField(SweepSpec(), env, initial_val=SweepValue(0.0, 1.0, 11))

    with pytest.raises(TypeError, match="SweepValue"):
        field.set_value(DirectValue(1))


def test_sweep_live_field_canonicalizes_stale_initial_step(env):
    field = SweepLiveField(
        SweepSpec(), env, initial_val=SweepValue(0.0, 1.0, 5, step=999.0)
    )

    assert field.get_value().step == pytest.approx(0.25)


def test_sweep_live_field_updates_step_through_pure_model(env):
    field = SweepLiveField(
        SweepSpec(), env, initial_val=SweepValue(0.0, 1.0, 11, step=0.1)
    )

    field.update_step(0.2)

    assert field.get_value().expts == 6
    assert field.get_value().step == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# optional ModuleRefSpec
# ---------------------------------------------------------------------------


def _make_optional_module_ref_spec() -> CfgSectionSpec:
    inner = CfgSectionSpec(
        label="Pulse",
        fields={"ch": ScalarSpec(label="Ch", type=int)},
    )
    return CfgSectionSpec(
        fields={
            "module": ModuleRefSpec(allowed=[inner], label="Module", optional=True),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )


def test_optional_module_ref_disabled_is_valid(env):
    spec = _make_optional_module_ref_spec()
    # Value omits "module" key → field starts disabled
    initial = CfgSectionValue(fields={"reps": DirectValue(10)})
    parent = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ModuleRefLiveField, parent.fields["module"])

    assert not field.is_enabled
    assert field.is_valid()
    assert parent.is_valid()


def test_optional_module_ref_set_enabled_emits_signals(env):
    spec = _make_optional_module_ref_spec()
    initial = CfgSectionValue(fields={"reps": DirectValue(10)})
    parent = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ModuleRefLiveField, parent.fields["module"])

    on_enabled = MagicMock()
    on_change = MagicMock()
    field.on_enabled_changed.connect(on_enabled)
    field.on_change.connect(on_change)

    field.set_enabled(True)

    on_enabled.assert_called_once_with(True)
    on_change.assert_called_once()


def test_optional_module_ref_noop_for_non_optional(env):
    inner = CfgSectionSpec(
        label="Pulse", fields={"ch": ScalarSpec(label="Ch", type=int)}
    )
    spec = CfgSectionSpec(
        fields={
            "module": ModuleRefSpec(allowed=[inner], label="Module", optional=False)
        }
    )
    initial = CfgSectionValue(
        fields={
            "module": ModuleRefValue(
                chosen_key="<Custom:Pulse>",
                value=CfgSectionValue(fields={"ch": DirectValue(0)}),
            )
        }
    )
    parent = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ModuleRefLiveField, parent.fields["module"])

    assert field.is_enabled is True
    field.set_enabled(False)  # noop for non-optional
    assert field.is_enabled is True


def test_optional_module_ref_parent_skips_key_when_disabled(env):
    spec = _make_optional_module_ref_spec()
    initial = CfgSectionValue(fields={"reps": DirectValue(10)})
    parent = SectionLiveField(spec, env, initial_val=initial)

    val = parent.get_value()
    assert "module" not in val.fields
    assert val.fields["reps"] == DirectValue(10)


def test_waveform_ref_missing_library_key_waits_for_ml_update(env):
    wav_spec = CfgSectionSpec(
        label="Const",
        fields={
            "style": ScalarSpec(label="Style", type=str),
            "length": ScalarSpec(label="Length", type=float),
        },
    )
    spec = CfgSectionSpec(
        fields={"waveform": WaveformRefSpec(allowed=[wav_spec], label="Waveform")}
    )
    initial = CfgSectionValue(
        fields={
            "waveform": WaveformRefValue(
                chosen_key="ro_waveform",
                value=CfgSectionValue(fields={"length": DirectValue(5.0)}),
            )
        }
    )
    ml = ModuleLibrary()
    env.ctrl.get_current_ml.return_value = ml

    section = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ModuleRefLiveField, section.fields["waveform"])
    assert field.get_chosen_key() == "ro_waveform"
    assert field.sub_field is None
    assert field.is_valid() is False

    ml.register_waveform(
        ro_waveform=WaveformCfgFactory.from_raw(
            {"style": "const", "length": 1.0}, ml=ml
        )
    )
    field.refresh_external(GuiEvent.ML_CHANGED)

    assert field.sub_field is not None
    assert field.is_valid() is True

"""Tests for LiveModel reactive data layer."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    make_default_value,
)
from zcu_tools.gui.app.main.live_model import (
    CallbackList,
    CenteredSweepLiveField,
    LibraryBindingState,
    LiveModelEnv,
    ReferenceLiveField,
    ScalarLiveField,
    SectionLiveField,
    SweepLiveField,
)
from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec
from zcu_tools.gui.app.main.specs.readout import make_pulse_readout_spec
from zcu_tools.gui.app.main.specs.waveform import make_const_waveform_spec
from zcu_tools.gui.app.main.ui.fields.utils import _spec_value_for_chosen
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.events import SessionEvent
from zcu_tools.gui.session.value_lookup import ValueInfo, ValueRef, ValueTypeError
from zcu_tools.meta_tool import ModuleLibrary
from zcu_tools.program.v2 import ModuleCfgFactory


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


def test_same_name_module_and_waveform_references_use_isolated_stores() -> None:
    ml = ModuleLibrary()
    ml.register_module(
        shared={
            "type": "pulse",
            "waveform": {"style": "const", "length": 0.25},
            "ch": 3,
            "nqz": 2,
            "freq": 5000.0,
            "gain": 0.4,
            "phase": 0.0,
            "pre_delay": 0.0,
            "post_delay": 0.0,
        }
    )
    ml.register_waveform(shared={"style": "const", "length": 1.5})
    module_ref = ReferenceSpec(kind="module", allowed=[make_pulse_spec()])
    waveform_ref = ReferenceSpec(
        kind="waveform",
        allowed=[make_const_waveform_spec()],
    )

    module_spec, module_value = _spec_value_for_chosen("shared", module_ref, ml)
    waveform_spec, waveform_value = _spec_value_for_chosen("shared", waveform_ref, ml)

    assert module_spec is not None and module_spec.label == "Pulse"
    assert module_value is not None
    assert module_value.fields["type"] == DirectValue("pulse")
    assert waveform_spec is not None and waveform_spec.label == "Const"
    assert waveform_value is not None
    assert waveform_value.fields["style"] == DirectValue("const")
    assert waveform_value.fields["length"] == DirectValue(1.5)


def test_optional_scalar_unset_is_valid(env):
    spec = ScalarSpec(label="Mixer freq", type=float, optional=True)
    field = ScalarLiveField(spec, env, initial_val=DirectValue(value=None))
    assert field.is_valid()  # optional unset (None) is valid
    field.set_value(5000.0)
    assert field.is_valid()
    field.set_value(None)  # clear back to unset
    assert field.is_valid()


def test_non_optional_scalar_unset_is_invalid(env):
    spec = ScalarSpec(label="Reps", type=int)
    field = ScalarLiveField(spec, env, initial_val=DirectValue(value=None))
    assert not field.is_valid()  # non-optional unset is invalid


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
    field.refresh_external(SessionEvent.MD_CHANGED)

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


def test_scalar_value_ref_resolves_once_to_direct_value(env):
    env.ctrl.read_value_source.return_value = (
        ValueInfo("device.flux.value", float, "device:flux"),
        0.125,
    )
    field = ScalarLiveField(ScalarSpec(label="Flux", type=float), env)

    field.set_value(ValueRef("device.flux.value"))
    field.refresh_external(SessionEvent.DEVICE_CHANGED)

    val = field.get_value()
    assert isinstance(val, DirectValue)
    assert val.value == 0.125
    env.ctrl.read_value_source.assert_called_once_with("device.flux.value", "float")


def test_scalar_value_ref_type_mismatch_fails_before_lookup(env):
    field = ScalarLiveField(ScalarSpec(label="Flux", type=float), env)

    with pytest.raises(ValueTypeError, match="target field"):
        field.set_value(ValueRef("predictor.loaded", "bool"))

    env.ctrl.read_value_source.assert_not_called()


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


def test_centered_sweep_live_field_rejects_wrong_value_type(env):
    field = CenteredSweepLiveField(
        CenteredSweepSpec(), env, initial_val=CenteredSweepValue(0.0, 1.0, 11)
    )

    with pytest.raises(TypeError, match="CenteredSweepValue"):
        field.set_value(DirectValue(1))


def test_centered_sweep_live_field_canonicalizes_stale_initial_step(env):
    field = CenteredSweepLiveField(
        CenteredSweepSpec(),
        env,
        initial_val=CenteredSweepValue(0.0, 10.0, 5, step=999.0),
    )

    assert field.get_value().step == pytest.approx(2.5)


def test_centered_sweep_live_field_updates_step_through_pure_model(env):
    field = CenteredSweepLiveField(
        CenteredSweepSpec(),
        env,
        initial_val=CenteredSweepValue(0.0, 10.0, 11, step=1.0),
    )

    field.update_step(2.0)

    assert field.get_value().expts == 6
    assert field.get_value().span == pytest.approx(10.0)
    assert field.get_value().step == pytest.approx(2.0)


def test_centered_sweep_live_field_rejects_locked_center_mismatch(env):
    field = CenteredSweepLiveField(
        CenteredSweepSpec(center_editable=False, locked_center=0.0),
        env,
        initial_val=CenteredSweepValue(0.0, 10.0, 11),
    )

    with pytest.raises(ValueError, match="locked to 0.0"):
        field.set_value(CenteredSweepValue(5.0, 10.0, 11))

    assert field.get_value().center == pytest.approx(0.0)
    assert field.get_value().span == pytest.approx(10.0)


def test_centered_sweep_live_field_rejects_zero_span_multi_point(env):
    field = CenteredSweepLiveField(
        CenteredSweepSpec(),
        env,
        initial_val=CenteredSweepValue(0.0, 10.0, 11),
    )

    with pytest.raises(ValueError, match="span must be > 0"):
        field.update_span(0.0)

    assert field.get_value().span == pytest.approx(10.0)


def test_centered_sweep_live_field_rejects_promoting_zero_span_to_multi_point(env):
    field = CenteredSweepLiveField(
        CenteredSweepSpec(),
        env,
        initial_val=CenteredSweepValue(0.0, 0.0, 1),
    )

    with pytest.raises(ValueError, match="span must be > 0"):
        field.update_expts(2)

    assert field.get_value().span == pytest.approx(0.0)
    assert field.get_value().expts == 1


def test_centered_sweep_live_field_can_lock_center_only(env):
    field = CenteredSweepLiveField(
        CenteredSweepSpec(center_editable=False),
        env,
        initial_val=CenteredSweepValue(0.0, 10.0, 11),
    )

    assert field.center_field.spec.editable is False
    field.update_span(20.0)
    assert field.get_value().center == pytest.approx(0.0)
    assert field.get_value().span == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# optional ReferenceSpec
# ---------------------------------------------------------------------------


def _make_optional_module_ref_spec() -> CfgSectionSpec:
    inner = CfgSectionSpec(
        label="Pulse",
        fields={"ch": ScalarSpec(label="Ch", type=int)},
    )
    return CfgSectionSpec(
        fields={
            "module": ReferenceSpec(
                kind="module", allowed=[inner], label="Module", optional=True
            ),
            "reps": ScalarSpec(label="Reps", type=int),
        }
    )


def test_optional_module_ref_disabled_is_valid(env):
    spec = _make_optional_module_ref_spec()
    # Value omits "module" key → field starts disabled
    initial = CfgSectionValue(fields={"reps": DirectValue(10)})
    parent = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ReferenceLiveField, parent.fields["module"])

    assert not field.is_enabled
    assert field.is_valid()
    assert parent.is_valid()


def test_optional_module_ref_set_enabled_emits_signals(env):
    spec = _make_optional_module_ref_spec()
    initial = CfgSectionValue(fields={"reps": DirectValue(10)})
    parent = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ReferenceLiveField, parent.fields["module"])

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
            "module": ReferenceSpec(
                kind="module", allowed=[inner], label="Module", optional=False
            )
        }
    )
    initial = CfgSectionValue(
        fields={
            "module": ReferenceValue(
                chosen_key="<Custom:Pulse>",
                value=CfgSectionValue(fields={"ch": DirectValue(0)}),
            )
        }
    )
    parent = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ReferenceLiveField, parent.fields["module"])

    assert field.is_enabled is True
    field.set_enabled(False)  # noop for non-optional
    assert field.is_enabled is True


def test_optional_module_ref_disabled_is_none_in_value(env):
    """A disabled optional ModuleRef self-reports None — the key is present in
    the (complete) value tree with value None, not omitted (ADR-0010)."""
    spec = _make_optional_module_ref_spec()
    initial = CfgSectionValue(fields={"reps": DirectValue(10)})
    parent = SectionLiveField(spec, env, initial_val=initial)

    val = parent.get_value()
    assert val.fields["module"] is None
    assert val.fields["reps"] == DirectValue(10)


def test_module_ref_set_value_none_disables(env):
    """set_value(None) disables an optional ref; get_value round-trips to None
    (ADR-0010) — symmetric set/get, no TypeError."""
    spec = _make_optional_module_ref_spec()
    inner_val = ReferenceValue(
        "<Custom:Pulse>", CfgSectionValue(fields={"ch": DirectValue(3)})
    )
    parent = SectionLiveField(
        spec, env, initial_val=CfgSectionValue(fields={"module": inner_val})
    )
    field = cast(ReferenceLiveField, parent.fields["module"])
    assert field.is_enabled is True
    assert field.get_value() is not None

    field.set_value(None)
    assert field.is_enabled is False
    assert field.get_value() is None


def test_module_ref_disabled_then_reenabled_round_trips(env):
    """Disable then re-enable an optional ref: re-enabling reveals the default
    shape (chosen_key set), not a crash."""
    spec = _make_optional_module_ref_spec()
    parent = SectionLiveField(spec, env, initial_val=CfgSectionValue(fields={}))
    field = cast(ReferenceLiveField, parent.fields["module"])
    assert field.is_enabled is False
    assert field.get_value() is None

    field.set_enabled(True)
    out = field.get_value()
    assert isinstance(out, ReferenceValue)


def test_module_ref_sub_edit_marks_overridden_in_get_value(env, monkeypatch):
    """Editing a sub-field of a LINKED library ref flips is_overridden in get_value."""
    inner = CfgSectionSpec(
        label="Pulse", fields={"ch": ScalarSpec(label="Ch", type=int)}
    )
    spec = CfgSectionSpec(
        fields={"module": ReferenceSpec(kind="module", allowed=[inner])}
    )

    # The library HAS lib_mod (so the ref is a genuine modified library ref, not
    # dangling). Stub the lookup to return the synthetic spec (avoids a real cfg
    # round-trip — this test is about the is_overridden flag, not conversion).
    import zcu_tools.gui.app.main.ui.fields.utils as _utils

    _orig = _utils._spec_value_for_chosen

    def _fake_lookup(chosen_key, allowed, ml):
        if chosen_key == "lib_mod":
            return inner, CfgSectionValue(fields={"ch": DirectValue(0)})
        return _orig(chosen_key, allowed, ml)

    monkeypatch.setattr(_utils, "_spec_value_for_chosen", _fake_lookup)

    initial = CfgSectionValue(
        fields={
            "module": ReferenceValue(
                chosen_key="lib_mod",
                value=CfgSectionValue(fields={"ch": DirectValue(1)}),
                is_overridden=True,
            )
        }
    )
    env.ctrl.get_current_ml.return_value = ModuleLibrary()
    section = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ReferenceLiveField, section.fields["module"])

    out = field.get_value()
    assert out is not None
    assert out.chosen_key == "lib_mod"
    assert out.is_overridden is True


def test_linked_ref_normalizes_locked_literals_without_override(env):
    locked_readout = make_pulse_readout_spec().lock_literal("pulse_cfg.freq", 0.0)
    spec = CfgSectionSpec(
        fields={
            "readout": ReferenceSpec(
                kind="module", allowed=[locked_readout], label="Readout"
            )
        }
    )
    ml = ModuleLibrary()
    ml.modules["readout_rf"] = ModuleCfgFactory.from_raw(
        {
            "type": "readout/pulse",
            "pulse_cfg": {
                "type": "pulse",
                "ch": 0,
                "nqz": 2,
                "freq": 5998.382171589952,
                "phase": 0.0,
                "gain": 0.1,
                "pre_delay": 0.0,
                "post_delay": 0.0,
                "waveform": {"style": "const", "length": 1.0},
            },
            "ro_cfg": {
                "type": "readout/direct",
                "ro_ch": 0,
                "ro_freq": 5998.382171589952,
                "ro_length": 0.9,
                "trig_offset": 0.5,
            },
        }
    )
    env.ctrl.get_current_ml.return_value = ml
    section = SectionLiveField(
        spec,
        env,
        initial_val=CfgSectionValue(
            fields={
                "readout": ReferenceValue(
                    "<Custom:Pulse Readout>", make_default_value(locked_readout)
                )
            }
        ),
    )
    field = cast(ReferenceLiveField, section.fields["readout"])

    field.set_chosen_key("readout_rf")

    out = field.get_value()
    assert out is not None
    assert out.chosen_key == "readout_rf"
    assert out.is_overridden is False
    assert field._binding_state is LibraryBindingState.LINKED
    pulse_cfg = out.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert pulse_cfg.fields["freq"] == DirectValue(0.0)


def test_module_ref_overridden_dangling_stays_relinkable(env, monkeypatch):
    """A restored overridden library ref whose key is absent keeps its key.

    The field is invalid while missing, but re-adding the same key relinks it
    instead of losing the relink target by converting to Custom.
    """
    from zcu_tools.gui.session.events import SessionEvent

    inner = CfgSectionSpec(
        label="Pulse",
        fields={"type": LiteralSpec("pulse"), "ch": ScalarSpec(label="Ch", type=int)},
    )
    spec = CfgSectionSpec(
        fields={"module": ReferenceSpec(kind="module", allowed=[inner])}
    )

    present = {"yes": False}
    import zcu_tools.gui.app.main.ui.fields.utils as _utils

    _orig = _utils._spec_value_for_chosen

    def _fake_lookup(chosen_key, allowed, ml):
        if chosen_key == "some_lib_module" and present["yes"]:
            return inner, CfgSectionValue(
                fields={"type": DirectValue("pulse"), "ch": DirectValue(11)}
            )
        return _orig(chosen_key, allowed, ml)

    monkeypatch.setattr(_utils, "_spec_value_for_chosen", _fake_lookup)
    initial = CfgSectionValue(
        fields={
            "module": ReferenceValue(
                chosen_key="some_lib_module",
                value=CfgSectionValue(
                    fields={"type": DirectValue("pulse"), "ch": DirectValue(7)}
                ),
                is_overridden=True,
            )
        }
    )
    env.ctrl.get_current_ml.return_value = ModuleLibrary()

    section = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ReferenceLiveField, section.fields["module"])

    assert field.get_chosen_key() == "some_lib_module"
    assert field.has_missing_library_ref() is True
    assert field.is_valid() is False
    out = field.get_value()
    assert isinstance(out, ReferenceValue)
    assert out.chosen_key == "some_lib_module"
    assert out.is_overridden is True
    ch = out.value.fields["ch"]
    assert isinstance(ch, DirectValue) and ch.value == 7

    present["yes"] = True
    field.refresh_external(SessionEvent.ML_CHANGED)

    assert field.get_chosen_key() == "some_lib_module"
    assert field._binding_state is LibraryBindingState.LINKED
    assert field.has_missing_library_ref() is False
    assert field.is_valid() is True
    out = field.get_value()
    assert isinstance(out, ReferenceValue)
    assert out.is_overridden is False
    ch = out.value.fields["ch"]
    assert isinstance(ch, DirectValue) and ch.value == 11


def test_modified_ref_self_heals_when_library_key_deleted_at_runtime(env, monkeypatch):
    """A MODIFIED (edited) library ref whose key is deleted at runtime must also
    self-heal to Custom on ML_CHANGED — _refresh_library_binding's old LINKED-only
    guard skipped MODIFIED refs, leaving a dangling chosen_key that broke lowering."""
    from zcu_tools.gui.session.events import SessionEvent

    inner = CfgSectionSpec(
        label="Const",
        fields={
            "style": LiteralSpec("const"),
            "length": ScalarSpec(label="Length", type=float),
        },
    )
    spec = CfgSectionSpec(
        fields={
            "wav": ReferenceSpec(kind="waveform", allowed=[inner], label="Waveform")
        }
    )

    present = {"yes": True}
    import zcu_tools.gui.app.main.ui.fields.utils as _utils

    _orig = _utils._spec_value_for_chosen

    def _fake_lookup(chosen_key, allowed, ml):
        if chosen_key == "lib_wav" and present["yes"]:
            return inner, CfgSectionValue(
                fields={"style": DirectValue("const"), "length": DirectValue(1.0)}
            )
        return _orig(chosen_key, allowed, ml)

    monkeypatch.setattr(_utils, "_spec_value_for_chosen", _fake_lookup)
    env.ctrl.get_current_ml.return_value = ModuleLibrary()

    initial = CfgSectionValue(
        fields={
            "wav": ReferenceValue(
                chosen_key="lib_wav",
                value=CfgSectionValue(
                    fields={"style": DirectValue("const"), "length": DirectValue(99.9)}
                ),
                is_overridden=True,  # MODIFIED
            )
        }
    )
    section = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ReferenceLiveField, section.fields["wav"])
    assert field.get_chosen_key() == "lib_wav"
    assert field.is_modified() is True

    # Delete the library key, then notify: the MODIFIED ref must heal to Custom,
    # keeping the user's edited length (99.9).
    present["yes"] = False
    field.refresh_external(SessionEvent.ML_CHANGED)

    assert field.get_chosen_key() == "<Custom:Const>"
    assert field.is_valid() is True
    out = field.get_value()
    assert out is not None
    length = out.value.fields["length"]
    assert isinstance(length, DirectValue)
    assert length.value == 99.9


def test_module_ref_custom_key_is_never_overridden(env):
    """A <Custom:> ref restored with is_overridden=True stays not-overridden."""
    inner = CfgSectionSpec(
        label="Pulse", fields={"ch": ScalarSpec(label="Ch", type=int)}
    )
    spec = CfgSectionSpec(
        fields={"module": ReferenceSpec(kind="module", allowed=[inner])}
    )
    initial = CfgSectionValue(
        fields={
            "module": ReferenceValue(
                chosen_key="<Custom:Pulse>",
                value=CfgSectionValue(fields={"ch": DirectValue(3)}),
                is_overridden=True,  # nonsensical for Custom; must be ignored
            )
        }
    )
    section = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ReferenceLiveField, section.fields["module"])

    assert field.is_modified() is False
    out = field.get_value()
    assert out is not None
    assert out.is_overridden is False


def test_linked_ref_missing_key_is_invalid_and_relinks_on_readd(env, monkeypatch):
    """A LINKED ref to an absent key stays LINKED + missing + invalid (red badge),
    NOT Custom — so re-adding an entry of that name re-links it (recoverable)."""
    wav_spec = CfgSectionSpec(
        label="Const",
        fields={
            "style": LiteralSpec("const"),
            "length": ScalarSpec(label="Length", type=float),
        },
    )
    spec = CfgSectionSpec(
        fields={
            "waveform": ReferenceSpec(
                kind="waveform", allowed=[wav_spec], label="Waveform"
            )
        }
    )

    present = {"yes": False}
    import zcu_tools.gui.app.main.ui.fields.utils as _utils

    _orig = _utils._spec_value_for_chosen

    def _fake_lookup(chosen_key, allowed, ml):
        if chosen_key == "ro_waveform" and present["yes"]:
            return wav_spec, CfgSectionValue(
                fields={"style": DirectValue("const"), "length": DirectValue(1.0)}
            )
        return _orig(chosen_key, allowed, ml)

    monkeypatch.setattr(_utils, "_spec_value_for_chosen", _fake_lookup)
    env.ctrl.get_current_ml.return_value = ModuleLibrary()

    initial = CfgSectionValue(
        fields={
            "waveform": ReferenceValue(
                chosen_key="ro_waveform",
                value=CfgSectionValue(
                    fields={
                        "style": DirectValue("const"),
                        "length": DirectValue(5.0),
                    }
                ),
            )
        }
    )
    section = SectionLiveField(spec, env, initial_val=initial)
    field = cast(ReferenceLiveField, section.fields["waveform"])
    # Absent key → LINKED + missing + invalid (NOT healed to Custom).
    assert field.get_chosen_key() == "ro_waveform"
    assert field._binding_state is LibraryBindingState.LINKED
    assert field.has_missing_library_ref() is True
    assert field.is_valid() is False
    out = field.get_value()
    assert isinstance(out, ReferenceValue)
    assert out.value.fields["length"] == DirectValue(5.0)

    # Other session refreshes keep the embedded snapshot instead of replacing it
    # with an empty section while the library key is still missing.
    field.refresh_external(SessionEvent.CONTEXT_SWITCHED)
    out = field.get_value()
    assert isinstance(out, ReferenceValue)
    assert out.value.fields["length"] == DirectValue(5.0)

    # Re-add an entry of that name → re-links to a valid LINKED ref.
    present["yes"] = True
    field.refresh_external(SessionEvent.ML_CHANGED)
    assert field.get_chosen_key() == "ro_waveform"
    assert field._binding_state is LibraryBindingState.LINKED
    assert field.has_missing_library_ref() is False
    assert field.is_valid() is True
    out = field.get_value()
    assert isinstance(out, ReferenceValue)
    assert out.value.fields["length"] == DirectValue(1.0)

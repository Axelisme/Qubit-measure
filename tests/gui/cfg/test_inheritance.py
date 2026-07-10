"""Tests for inherit_from — field value inheritance when switching Ref combos."""

from __future__ import annotations

from zcu_tools.gui.app.main.specs.readout import (
    make_direct_readout_spec,
    make_pulse_readout_spec,
)
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    align_locked_literals,
    inherit_from,
    make_default_value,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scalar_spec(type_: type, label: str = "x") -> ScalarSpec:
    return ScalarSpec(label=label, type=type_)


def _section(fields: dict) -> CfgSectionSpec:
    return CfgSectionSpec(fields=fields)


def _val(fields: dict) -> CfgSectionValue:
    return CfgSectionValue(fields=fields)


# ---------------------------------------------------------------------------
# ScalarSpec
# ---------------------------------------------------------------------------


def test_scalar_same_type_inherits():
    old_spec = _section({"x": _scalar_spec(float)})
    new_spec = _section({"x": _scalar_spec(float)})
    old_val = _val({"x": DirectValue(3.14)})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["x"] == DirectValue(3.14)


def test_scalar_type_mismatch_uses_default():
    old_spec = _section({"x": _scalar_spec(int)})
    new_spec = _section({"x": _scalar_spec(float)})
    old_val = _val({"x": DirectValue(5)})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["x"] == DirectValue(0.0)


def test_scalar_missing_old_key_uses_default():
    old_spec = _section({})
    new_spec = _section({"x": _scalar_spec(int)})
    old_val = _val({})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["x"] == DirectValue(0)


# ---------------------------------------------------------------------------
# LiteralSpec
# ---------------------------------------------------------------------------


def test_literal_never_inherits():
    old_spec = _section({"t": LiteralSpec(value="gauss")})
    new_spec = _section({"t": LiteralSpec(value="cosine")})
    old_val = _val({"t": DirectValue("gauss")})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["t"] == DirectValue("cosine")


def test_literal_new_value_used_even_when_old_missing():
    old_spec = _section({})
    new_spec = _section({"t": LiteralSpec(value="flat_top")})
    old_val = _val({})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["t"] == DirectValue("flat_top")


# ---------------------------------------------------------------------------
# SweepSpec
# ---------------------------------------------------------------------------


def test_sweep_inherits_whole_value():
    old_spec = _section({"s": SweepSpec()})
    new_spec = _section({"s": SweepSpec()})
    old_val = _val({"s": SweepValue(start=1.0, stop=5.0, expts=21, step=0.2)})

    result = inherit_from(old_val, old_spec, new_spec)

    sv = result.fields["s"]
    assert isinstance(sv, SweepValue)
    assert sv.start == 1.0
    assert sv.stop == 5.0
    assert sv.expts == 21
    assert sv.step == 0.2


def test_sweep_type_mismatch_uses_default():
    old_spec = _section({"s": _scalar_spec(float)})
    new_spec = _section({"s": SweepSpec()})
    old_val = _val({"s": DirectValue(1.0)})

    result = inherit_from(old_val, old_spec, new_spec)

    sv = result.fields["s"]
    assert isinstance(sv, SweepValue)
    assert sv.start == 0.0


def test_centered_sweep_inherits_whole_value():
    old_spec = _section({"s": CenteredSweepSpec()})
    new_spec = _section({"s": CenteredSweepSpec()})
    old_val = _val({"s": CenteredSweepValue(center=5.0, span=20.0, expts=21)})

    result = inherit_from(old_val, old_spec, new_spec)

    sv = result.fields["s"]
    assert isinstance(sv, CenteredSweepValue)
    assert sv.center == 5.0
    assert sv.span == 20.0
    assert sv.expts == 21
    assert sv.step == 1.0


def test_centered_sweep_type_mismatch_uses_default():
    old_spec = _section({"s": SweepSpec()})
    new_spec = _section({"s": CenteredSweepSpec()})
    old_val = _val({"s": SweepValue(start=1.0, stop=5.0, expts=21)})

    result = inherit_from(old_val, old_spec, new_spec)

    sv = result.fields["s"]
    assert isinstance(sv, CenteredSweepValue)
    assert sv.center == 0.5
    assert sv.span == 1.0


# ---------------------------------------------------------------------------
# ModuleRefSpec / WaveformRefSpec
# ---------------------------------------------------------------------------


def test_moduleref_inherits_chosen_key_and_value():
    inner = _section({"ch": _scalar_spec(int)})
    ref_spec = ModuleRefSpec(allowed=[inner])
    old_spec = _section({"m": ref_spec})
    new_spec = _section({"m": ref_spec})
    old_inner_val = _val({"ch": DirectValue(3)})
    old_val = _val({"m": ModuleRefValue(chosen_key="readout_rf", value=old_inner_val)})

    result = inherit_from(old_val, old_spec, new_spec)

    mrv = result.fields["m"]
    assert isinstance(mrv, ModuleRefValue)
    assert mrv.chosen_key == "readout_rf"
    assert mrv.value.fields["ch"] == DirectValue(3)


def test_waveformref_inherits():
    inner = _section({"amp": _scalar_spec(float)})
    ref_spec = WaveformRefSpec(allowed=[inner])
    old_spec = _section({"w": ref_spec})
    new_spec = _section({"w": ref_spec})
    old_inner_val = _val({"amp": DirectValue(0.9)})
    old_val = _val(
        {"w": WaveformRefValue(chosen_key="ro_waveform", value=old_inner_val)}
    )

    result = inherit_from(old_val, old_spec, new_spec)

    wrv = result.fields["w"]
    assert isinstance(wrv, WaveformRefValue)
    assert wrv.chosen_key == "ro_waveform"
    assert wrv.value.fields["amp"] == DirectValue(0.9)


def test_moduleref_type_mismatch_uses_default():
    inner = _section({"ch": _scalar_spec(int)})
    ref_spec = ModuleRefSpec(allowed=[inner])
    old_spec = _section({"m": _scalar_spec(float)})
    new_spec = _section({"m": ref_spec})
    old_val = _val({"m": DirectValue(1.0)})

    result = inherit_from(old_val, old_spec, new_spec)

    mrv = result.fields["m"]
    assert isinstance(mrv, ModuleRefValue)
    assert mrv.chosen_key.startswith("<Custom:")


# ---------------------------------------------------------------------------
# CfgSectionSpec — recursive
# ---------------------------------------------------------------------------


def test_section_recurses():
    inner_old = _section({"a": _scalar_spec(float), "b": _scalar_spec(int)})
    inner_new = _section({"a": _scalar_spec(float), "c": _scalar_spec(str)})
    old_spec = _section({"sub": inner_old})
    new_spec = _section({"sub": inner_new})
    old_val = _val({"sub": _val({"a": DirectValue(9.9), "b": DirectValue(7)})})

    result = inherit_from(old_val, old_spec, new_spec)

    sub = result.fields["sub"]
    assert isinstance(sub, CfgSectionValue)
    assert sub.fields["a"] == DirectValue(9.9)  # inherited
    assert sub.fields["c"] == DirectValue("")  # new field → default


def test_section_type_mismatch_uses_default():
    inner = _section({"a": _scalar_spec(float)})
    old_spec = _section({"sub": _scalar_spec(float)})
    new_spec = _section({"sub": inner})
    old_val = _val({"sub": DirectValue(1.0)})

    result = inherit_from(old_val, old_spec, new_spec)

    sub = result.fields["sub"]
    assert isinstance(sub, CfgSectionValue)
    assert sub.fields["a"] == DirectValue(0.0)


# ---------------------------------------------------------------------------
# Mixed / edge cases
# ---------------------------------------------------------------------------


def test_extra_old_keys_ignored():
    old_spec = _section({"x": _scalar_spec(float), "y": _scalar_spec(float)})
    new_spec = _section({"x": _scalar_spec(float)})
    old_val = _val({"x": DirectValue(1.1), "y": DirectValue(2.2)})

    result = inherit_from(old_val, old_spec, new_spec)

    assert set(result.fields.keys()) == {"x"}
    assert result.fields["x"] == DirectValue(1.1)


def test_empty_old_all_defaults():
    new_spec = _section(
        {
            "a": _scalar_spec(float),
            "b": _scalar_spec(int),
            "s": SweepSpec(),
        }
    )
    old_val = _val({})
    old_spec = _section({})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["a"] == DirectValue(0.0)
    assert result.fields["b"] == DirectValue(0)
    assert isinstance(result.fields["s"], SweepValue)


# ---------------------------------------------------------------------------
# Hardcoded cross-spec: Direct Readout ↔ Pulse Readout
# ---------------------------------------------------------------------------


def _direct_readout_val() -> CfgSectionValue:
    return _val(
        {
            "type": DirectValue("readout/direct"),
            "ro_ch": DirectValue(0),
            "ro_freq": DirectValue(6500.0),
            "ro_length": DirectValue(2.0),
            "trig_offset": DirectValue(0.5),
        }
    )


def test_direct_to_pulse_injects_into_ro_cfg():
    old_val = _direct_readout_val()

    result = inherit_from(
        old_val, make_direct_readout_spec(), make_pulse_readout_spec()
    )

    # ro_cfg should be the old DirectReadout value verbatim
    assert result.fields["ro_cfg"] is old_val
    # pulse_cfg should be default (CfgSectionValue present)
    assert isinstance(result.fields["pulse_cfg"], CfgSectionValue)


def test_direct_to_pulse_ro_cfg_carries_field_values():
    old_val = _direct_readout_val()

    result = inherit_from(
        old_val, make_direct_readout_spec(), make_pulse_readout_spec()
    )

    ro = result.fields["ro_cfg"]
    assert isinstance(ro, CfgSectionValue)
    assert ro.fields["ro_ch"] == DirectValue(0)
    assert ro.fields["ro_freq"] == DirectValue(6500.0)
    assert ro.fields["ro_length"] == DirectValue(2.0)
    assert ro.fields["trig_offset"] == DirectValue(0.5)


def test_pulse_to_direct_extracts_ro_cfg():
    ro_val = _direct_readout_val()
    old_val = _val(
        {
            "type": DirectValue("readout/pulse"),
            "pulse_cfg": _val({}),
            "ro_cfg": ro_val,
        }
    )

    result = inherit_from(
        old_val, make_pulse_readout_spec(), make_direct_readout_spec()
    )

    assert result is ro_val


def test_pulse_to_direct_missing_ro_cfg_uses_default():
    old_val = _val(
        {
            "type": DirectValue("readout/pulse"),
            "pulse_cfg": _val({}),
            # ro_cfg missing
        }
    )

    result = inherit_from(
        old_val, make_pulse_readout_spec(), make_direct_readout_spec()
    )

    assert isinstance(result, CfgSectionValue)
    ro_ch = result.fields.get("ro_ch")
    assert isinstance(ro_ch, DirectValue)
    assert ro_ch.value == 0


# ---------------------------------------------------------------------------
# make_default_value: complete value tree + optional ref → None (ADR-0010)
# ---------------------------------------------------------------------------


def test_make_default_value_is_complete_and_optional_ref_is_none():
    """The default value tree has an entry for every spec field (no missing
    keys); an optional ModuleRef/WaveformRef defaults to None (disabled), a
    non-optional one to an enabled ref (ADR-0010)."""
    inner = CfgSectionSpec(label="Pulse", fields={"ch": ScalarSpec("Ch", int)})
    spec = CfgSectionSpec(
        fields={
            "reps": ScalarSpec("Reps", int),
            "opt_mod": ModuleRefSpec(allowed=[inner], label="Opt", optional=True),
            "req_mod": ModuleRefSpec(allowed=[inner], label="Req", optional=False),
            "opt_wf": WaveformRefSpec(allowed=[inner], label="OptWf", optional=True),
        }
    )
    val = make_default_value(spec)

    # complete: every spec field has an entry
    assert set(val.fields) == set(spec.fields)
    # optional refs default to None (disabled)
    assert val.fields["opt_mod"] is None
    assert val.fields["opt_wf"] is None
    # non-optional ref defaults to an enabled ModuleRefValue
    assert isinstance(val.fields["req_mod"], ModuleRefValue)


def test_inherit_from_preserves_disabled_optional_ref():
    """Inheriting from an old value whose optional ref was disabled (None) keeps
    it disabled in the new value (ADR-0010)."""
    inner = CfgSectionSpec(label="Pulse", fields={"ch": ScalarSpec("Ch", int)})
    spec = CfgSectionSpec(
        fields={"reset": ModuleRefSpec(allowed=[inner], label="Reset", optional=True)}
    )
    old_val = CfgSectionValue(fields={"reset": None})
    result = inherit_from(old_val, spec, spec)
    assert result.fields["reset"] is None


def test_align_locked_literals_projects_linked_ref_into_caller_spec():
    locked_readout = make_pulse_readout_spec().lock_literal("pulse_cfg.freq", 0.0)
    spec = CfgSectionSpec(
        fields={"readout": ModuleRefSpec(allowed=[locked_readout], label="Readout")}
    )
    readout_value = make_default_value(make_pulse_readout_spec())
    pulse_cfg = readout_value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    pulse_cfg.fields["freq"] = DirectValue(5998.0)
    value = CfgSectionValue(
        fields={"readout": ModuleRefValue("readout_rf", readout_value)}
    )

    result = align_locked_literals(spec, value)

    assert result is value
    readout = result.fields["readout"]
    assert isinstance(readout, ModuleRefValue)
    pulse_cfg = readout.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert pulse_cfg.fields["freq"] == DirectValue(0.0)

"""Tests for inherit_from — field value inheritance when switching Ref combos."""

from __future__ import annotations

from zcu_tools.gui.specs.readout import DIRECT_READOUT_SPEC, PULSE_READOUT_SPEC
from zcu_tools.gui.adapter import (
    CfgSectionSpec,
    CfgSectionValue,
    LiteralSpec,
    ModuleRefSpec,
    ModuleRefValue,
    MultiSweepSpec,
    MultiSweepValue,
    ScalarSpec,
    ScalarValue,
    SweepSpec,
    SweepValue,
    WaveformRefSpec,
    WaveformRefValue,
    inherit_from,
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
    old_val = _val({"x": ScalarValue(3.14)})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["x"] == ScalarValue(3.14)


def test_scalar_type_mismatch_uses_default():
    old_spec = _section({"x": _scalar_spec(int)})
    new_spec = _section({"x": _scalar_spec(float)})
    old_val = _val({"x": ScalarValue(5)})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["x"] == ScalarValue(0.0)


def test_scalar_missing_old_key_uses_default():
    old_spec = _section({})
    new_spec = _section({"x": _scalar_spec(int)})
    old_val = _val({})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["x"] == ScalarValue(0)


# ---------------------------------------------------------------------------
# LiteralSpec
# ---------------------------------------------------------------------------


def test_literal_never_inherits():
    old_spec = _section({"t": LiteralSpec(value="gauss")})
    new_spec = _section({"t": LiteralSpec(value="cosine")})
    old_val = _val({"t": ScalarValue("gauss")})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["t"] == ScalarValue("cosine")


def test_literal_new_value_used_even_when_old_missing():
    old_spec = _section({})
    new_spec = _section({"t": LiteralSpec(value="flat_top")})
    old_val = _val({})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["t"] == ScalarValue("flat_top")


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
    old_val = _val({"s": ScalarValue(1.0)})

    result = inherit_from(old_val, old_spec, new_spec)

    sv = result.fields["s"]
    assert isinstance(sv, SweepValue)
    assert sv.start == 0.0


# ---------------------------------------------------------------------------
# MultiSweepSpec
# ---------------------------------------------------------------------------


def test_multisweep_per_axis_inheritance():
    shared_ax = SweepSpec()
    old_spec = _section({"ms": MultiSweepSpec(axes={"x": shared_ax, "y": shared_ax})})
    new_spec = _section({"ms": MultiSweepSpec(axes={"x": shared_ax, "z": shared_ax})})
    old_val = _val(
        {
            "ms": MultiSweepValue(
                axes={
                    "x": SweepValue(1.0, 2.0, 5),
                    "y": SweepValue(0.0, 1.0, 3),
                }
            )
        }
    )

    result = inherit_from(old_val, old_spec, new_spec)

    msv = result.fields["ms"]
    assert isinstance(msv, MultiSweepValue)
    assert msv.axes["x"] == SweepValue(1.0, 2.0, 5)  # inherited
    assert msv.axes["z"] == SweepValue(0.0, 1.0, 11)  # new axis → default


# ---------------------------------------------------------------------------
# ModuleRefSpec / WaveformRefSpec
# ---------------------------------------------------------------------------


def test_moduleref_inherits_chosen_key_and_value():
    inner = _section({"ch": _scalar_spec(int)})
    ref_spec = ModuleRefSpec(allowed=[inner])
    old_spec = _section({"m": ref_spec})
    new_spec = _section({"m": ref_spec})
    old_inner_val = _val({"ch": ScalarValue(3)})
    old_val = _val({"m": ModuleRefValue(chosen_key="readout_rf", value=old_inner_val)})

    result = inherit_from(old_val, old_spec, new_spec)

    mrv = result.fields["m"]
    assert isinstance(mrv, ModuleRefValue)
    assert mrv.chosen_key == "readout_rf"
    assert mrv.value.fields["ch"] == ScalarValue(3)


def test_waveformref_inherits():
    inner = _section({"amp": _scalar_spec(float)})
    ref_spec = WaveformRefSpec(allowed=[inner])
    old_spec = _section({"w": ref_spec})
    new_spec = _section({"w": ref_spec})
    old_inner_val = _val({"amp": ScalarValue(0.9)})
    old_val = _val(
        {"w": WaveformRefValue(chosen_key="ro_waveform", value=old_inner_val)}
    )

    result = inherit_from(old_val, old_spec, new_spec)

    wrv = result.fields["w"]
    assert isinstance(wrv, WaveformRefValue)
    assert wrv.chosen_key == "ro_waveform"
    assert wrv.value.fields["amp"] == ScalarValue(0.9)


def test_moduleref_type_mismatch_uses_default():
    inner = _section({"ch": _scalar_spec(int)})
    ref_spec = ModuleRefSpec(allowed=[inner])
    old_spec = _section({"m": _scalar_spec(float)})
    new_spec = _section({"m": ref_spec})
    old_val = _val({"m": ScalarValue(1.0)})

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
    old_val = _val({"sub": _val({"a": ScalarValue(9.9), "b": ScalarValue(7)})})

    result = inherit_from(old_val, old_spec, new_spec)

    sub = result.fields["sub"]
    assert isinstance(sub, CfgSectionValue)
    assert sub.fields["a"] == ScalarValue(9.9)  # inherited
    assert sub.fields["c"] == ScalarValue("")  # new field → default


def test_section_type_mismatch_uses_default():
    inner = _section({"a": _scalar_spec(float)})
    old_spec = _section({"sub": _scalar_spec(float)})
    new_spec = _section({"sub": inner})
    old_val = _val({"sub": ScalarValue(1.0)})

    result = inherit_from(old_val, old_spec, new_spec)

    sub = result.fields["sub"]
    assert isinstance(sub, CfgSectionValue)
    assert sub.fields["a"] == ScalarValue(0.0)


# ---------------------------------------------------------------------------
# Mixed / edge cases
# ---------------------------------------------------------------------------


def test_extra_old_keys_ignored():
    old_spec = _section({"x": _scalar_spec(float), "y": _scalar_spec(float)})
    new_spec = _section({"x": _scalar_spec(float)})
    old_val = _val({"x": ScalarValue(1.1), "y": ScalarValue(2.2)})

    result = inherit_from(old_val, old_spec, new_spec)

    assert set(result.fields.keys()) == {"x"}
    assert result.fields["x"] == ScalarValue(1.1)


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

    assert result.fields["a"] == ScalarValue(0.0)
    assert result.fields["b"] == ScalarValue(0)
    assert isinstance(result.fields["s"], SweepValue)


# ---------------------------------------------------------------------------
# Hardcoded cross-spec: Direct Readout ↔ Pulse Readout
# ---------------------------------------------------------------------------


def _direct_readout_val() -> CfgSectionValue:
    return _val(
        {
            "type": ScalarValue("readout/direct"),
            "ro_ch": ScalarValue(0),
            "ro_freq": ScalarValue(6500.0),
            "ro_length": ScalarValue(2.0),
            "trig_offset": ScalarValue(0.5),
        }
    )


def test_direct_to_pulse_injects_into_ro_cfg():
    old_val = _direct_readout_val()

    result = inherit_from(old_val, DIRECT_READOUT_SPEC, PULSE_READOUT_SPEC)

    # ro_cfg should be the old DirectReadout value verbatim
    assert result.fields["ro_cfg"] is old_val
    # pulse_cfg should be default (CfgSectionValue present)
    assert isinstance(result.fields["pulse_cfg"], CfgSectionValue)


def test_direct_to_pulse_ro_cfg_carries_field_values():
    old_val = _direct_readout_val()

    result = inherit_from(old_val, DIRECT_READOUT_SPEC, PULSE_READOUT_SPEC)

    ro = result.fields["ro_cfg"]
    assert isinstance(ro, CfgSectionValue)
    assert ro.fields["ro_ch"] == ScalarValue(0)
    assert ro.fields["ro_freq"] == ScalarValue(6500.0)
    assert ro.fields["ro_length"] == ScalarValue(2.0)
    assert ro.fields["trig_offset"] == ScalarValue(0.5)


def test_pulse_to_direct_extracts_ro_cfg():
    ro_val = _direct_readout_val()
    old_val = _val(
        {
            "type": ScalarValue("readout/pulse"),
            "pulse_cfg": _val({}),
            "ro_cfg": ro_val,
        }
    )

    result = inherit_from(old_val, PULSE_READOUT_SPEC, DIRECT_READOUT_SPEC)

    assert result is ro_val


def test_pulse_to_direct_missing_ro_cfg_uses_default():
    old_val = _val(
        {
            "type": ScalarValue("readout/pulse"),
            "pulse_cfg": _val({}),
            # ro_cfg missing
        }
    )

    result = inherit_from(old_val, PULSE_READOUT_SPEC, DIRECT_READOUT_SPEC)

    from zcu_tools.gui.adapter import ChannelValue

    assert isinstance(result, CfgSectionValue)
    assert isinstance(result.fields.get("ro_ch"), ChannelValue)

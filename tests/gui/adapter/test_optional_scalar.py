"""Tests for optional scalar fields (ScalarSpec.optional).

An optional scalar may be left empty (value None) and is *valid* while empty; at
lowering it is omitted so the config-model default applies (e.g. mixer_freq=None).
This is the opposite of ``required`` (empty = invalid), so the two are mutually
exclusive.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    CfgSectionSpec,
    DirectValue,
    ScalarSpec,
    make_default_value,
)
from zcu_tools.gui.app.main.services.session_codec import (
    _section_value_from_raw,
    _section_value_to_raw,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        fields={
            "reps": ScalarSpec(label="Reps", type=int),
            "mixer_freq": ScalarSpec(label="Mixer freq", type=float, optional=True),
        }
    )


def test_required_and_optional_are_mutually_exclusive() -> None:
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        ScalarSpec(label="bad", type=float, required=True, optional=True)


def test_make_default_value_optional_scalar_is_unset_none() -> None:
    val = make_default_value(_spec())
    mf = val.fields["mixer_freq"]
    assert isinstance(mf, DirectValue)
    assert mf.value is None


def test_static_validate_allows_optional_none() -> None:
    val = make_default_value(_spec())
    # Must not raise: optional None is a legal final state.
    CfgSchema(spec=_spec(), value=val).validate(ModuleLibrary())


def test_lowering_omits_optional_unset_scalar() -> None:
    val = make_default_value(_spec())
    raw = CfgSchema(spec=_spec(), value=val).to_raw_dict(None, None)
    assert "mixer_freq" not in raw
    assert raw["reps"] == 0


def test_lowering_includes_optional_when_set() -> None:
    val = make_default_value(_spec()).with_field("mixer_freq", 5000.0)
    raw = CfgSchema(spec=_spec(), value=val).to_raw_dict(None, None)
    assert raw["mixer_freq"] == 5000.0


def test_lowering_non_optional_unset_still_raises() -> None:
    spec = CfgSectionSpec(fields={"x": ScalarSpec(label="X", type=float)})
    val = make_default_value(spec).with_field("x", DirectValue(value=None))
    with pytest.raises(RuntimeError, match="unset"):
        CfgSchema(spec=spec, value=val).to_raw_dict(None, None)


def test_validate_dynamic_allows_optional_none() -> None:
    val = make_default_value(_spec())
    # md present → dynamic validation runs; optional None must not raise.
    CfgSchema(spec=_spec(), value=val).validate_dynamic(MetaDict(), ModuleLibrary())


def test_validate_dynamic_non_optional_none_raises() -> None:
    spec = CfgSectionSpec(fields={"x": ScalarSpec(label="X", type=float)})
    val = make_default_value(spec).with_field("x", DirectValue(value=None))
    with pytest.raises(RuntimeError, match="unset"):
        CfgSchema(spec=spec, value=val).validate_dynamic(MetaDict(), ModuleLibrary())


def test_codec_round_trips_optional_none() -> None:
    spec = _spec()
    val = make_default_value(spec)
    raw = _section_value_to_raw(spec, val)
    back = _section_value_from_raw(spec, raw)
    mf = back.fields["mixer_freq"]
    assert isinstance(mf, DirectValue)
    assert mf.value is None


def test_codec_round_trips_optional_value() -> None:
    spec = _spec()
    val = make_default_value(spec).with_field("mixer_freq", 4200.0)
    raw = _section_value_to_raw(spec, val)
    back = _section_value_from_raw(spec, raw)
    assert back.fields["mixer_freq"].value == 4200.0  # type: ignore[union-attr]


def test_codec_missing_optional_key_falls_back_to_none() -> None:
    """An old session saved before the optional field existed restores to None."""
    spec = _spec()
    back = _section_value_from_raw(spec, {"reps": {"__kind": "direct", "value": 50}})
    mf = back.fields["mixer_freq"]
    assert isinstance(mf, DirectValue)
    assert mf.value is None


# --- pulse spec integration: mixer_freq is an optional "Advanced" field --------


def test_make_pulse_spec_mixer_freq_is_optional_advanced() -> None:
    from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec

    mf = make_pulse_spec().fields["mixer_freq"]
    assert isinstance(mf, ScalarSpec)
    assert mf.optional is True
    assert mf.group == "Advanced"


def test_pulse_default_has_unset_mixer_freq_and_lowers_omitted() -> None:
    from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec

    spec = make_pulse_spec()
    val = make_default_value(spec)
    mf = val.fields["mixer_freq"]
    assert isinstance(mf, DirectValue) and mf.value is None
    raw = CfgSchema(spec=spec, value=val).to_raw_dict(None, None)
    assert "mixer_freq" not in raw  # unset → omitted → PulseCfg default None


def test_pulse_with_mixer_freq_lowers_value() -> None:
    from zcu_tools.gui.app.main.specs.pulse import make_pulse_spec

    spec = make_pulse_spec()
    val = make_default_value(spec).with_field("mixer_freq", 500.0)
    raw = CfgSchema(spec=spec, value=val).to_raw_dict(None, None)
    assert raw["mixer_freq"] == 500.0


# --- direct readout spec integration: gen_ch is an optional "Advanced" field ---


def test_make_direct_readout_spec_gen_ch_is_optional_advanced() -> None:
    from zcu_tools.gui.app.main.specs.readout import make_direct_readout_spec

    gen_ch = make_direct_readout_spec().fields["gen_ch"]
    assert isinstance(gen_ch, ScalarSpec)
    assert gen_ch.type is int
    assert gen_ch.optional is True
    assert gen_ch.group == "Advanced"


def test_direct_readout_default_has_unset_gen_ch_and_lowers_omitted() -> None:
    from zcu_tools.gui.app.main.specs.readout import make_direct_readout_spec

    spec = make_direct_readout_spec()
    val = make_default_value(spec)
    gen_ch = val.fields["gen_ch"]
    assert isinstance(gen_ch, DirectValue) and gen_ch.value is None
    raw = CfgSchema(spec=spec, value=val).to_raw_dict(None, None)
    assert "gen_ch" not in raw


def test_direct_readout_with_gen_ch_lowers_value() -> None:
    from zcu_tools.gui.app.main.specs.readout import make_direct_readout_spec

    spec = make_direct_readout_spec()
    val = make_default_value(spec).with_field("gen_ch", 9)
    raw = CfgSchema(spec=spec, value=val).to_raw_dict(None, None)
    assert raw["gen_ch"] == 9

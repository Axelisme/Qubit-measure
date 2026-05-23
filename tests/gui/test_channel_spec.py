"""Tests for ChannelSpec / ChannelValue — schema_to_dict, make_default_value, inherit_from."""

from __future__ import annotations

from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    ChannelSpec,
    ChannelValue,
    ScalarSpec,
    DirectValue,
    ScalarValue,
    inherit_from,
    make_default_value,
    schema_to_dict,
)


def _section(fields: dict) -> CfgSectionSpec:
    return CfgSectionSpec(fields=fields)


def _val(fields: dict) -> CfgSectionValue:
    return CfgSectionValue(fields=fields)


# ---------------------------------------------------------------------------
# schema_to_dict
# ---------------------------------------------------------------------------


def test_channel_int_chosen_to_dict() -> None:
    spec = _section({"ch": ChannelSpec(label="Gen ch")})
    val = _val({"ch": ChannelValue(chosen=3, resolved=None)})
    result = schema_to_dict(CfgSchema(spec=spec, value=val), ml=None)
    assert result["ch"] == 3


def test_channel_str_chosen_to_dict_uses_resolved() -> None:
    spec = _section({"ch": ChannelSpec(label="Gen ch")})
    val = _val({"ch": ChannelValue(chosen="res_ch", resolved=5)})
    result = schema_to_dict(CfgSchema(spec=spec, value=val), ml=None)
    assert result["ch"] == 5


def test_channel_str_chosen_unresolved_raises() -> None:
    import pytest

    spec = _section({"ch": ChannelSpec(label="Gen ch")})
    val = _val({"ch": ChannelValue(chosen="unknown_key", resolved=None)})
    with pytest.raises(RuntimeError, match="unknown_key"):
        schema_to_dict(CfgSchema(spec=spec, value=val), ml=None)


# ---------------------------------------------------------------------------
# make_default_value
# ---------------------------------------------------------------------------


def test_channel_default_value() -> None:
    spec = _section({"ch": ChannelSpec(label="Gen ch")})
    val = make_default_value(spec)
    ch_val = val.fields["ch"]
    assert isinstance(ch_val, ChannelValue)
    assert ch_val.chosen == 0
    assert ch_val.resolved is None


# ---------------------------------------------------------------------------
# inherit_from
# ---------------------------------------------------------------------------


def test_channel_inherit_from_channel() -> None:
    old_spec = _section({"ch": ChannelSpec(label="Gen ch")})
    new_spec = _section({"ch": ChannelSpec(label="Gen ch")})
    old_val = _val({"ch": ChannelValue(chosen="res_ch", resolved=2)})

    result = inherit_from(old_val, old_spec, new_spec)
    ch = result.fields["ch"]
    assert isinstance(ch, ChannelValue)
    assert ch.chosen == "res_ch"
    assert ch.resolved == 2


def test_channel_inherit_from_non_channel_uses_default() -> None:
    old_spec = _section({"ch": ScalarSpec(label="ch", type=int)})
    new_spec = _section({"ch": ChannelSpec(label="Gen ch")})
    old_val = _val({"ch": DirectValue(7)})

    result = inherit_from(old_val, old_spec, new_spec)
    ch = result.fields["ch"]
    assert isinstance(ch, ChannelValue)
    assert ch.chosen == 0
    assert ch.resolved is None


def test_channel_missing_key_uses_default() -> None:
    old_spec = _section({})
    new_spec = _section({"ch": ChannelSpec(label="Gen ch")})
    old_val = _val({})

    result = inherit_from(old_val, old_spec, new_spec)
    ch = result.fields["ch"]
    assert isinstance(ch, ChannelValue)
    assert ch.chosen == 0

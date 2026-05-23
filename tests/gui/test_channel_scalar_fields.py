"""Tests for channel fields represented as scalar int values."""

from __future__ import annotations

import pytest
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ScalarSpec,
    inherit_from,
    make_default_value,
    schema_to_dict,
)


def _section(fields: dict) -> CfgSectionSpec:
    return CfgSectionSpec(fields=fields)


def _val(fields: dict) -> CfgSectionValue:
    return CfgSectionValue(fields=fields)


def test_channel_direct_scalar_to_dict() -> None:
    spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    val = _val({"ch": DirectValue(3)})
    result = schema_to_dict(CfgSchema(spec=spec, value=val), ml=None)
    assert result["ch"] == 3


def test_channel_eval_scalar_to_dict_uses_resolved_snapshot() -> None:
    spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    val = _val({"ch": EvalValue(expr="res_ch", resolved=5)})
    result = schema_to_dict(CfgSchema(spec=spec, value=val), ml=None)
    assert result["ch"] == 5


def test_channel_eval_scalar_unresolved_raises() -> None:
    spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    val = _val({"ch": EvalValue(expr="unknown_key", resolved=None)})
    with pytest.raises(RuntimeError, match="unknown_key"):
        schema_to_dict(CfgSchema(spec=spec, value=val), ml=None)


def test_channel_scalar_default_value() -> None:
    spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    val = make_default_value(spec)
    ch_val = val.fields["ch"]
    assert isinstance(ch_val, DirectValue)
    assert ch_val.value == 0


def test_channel_scalar_inherits_direct_value() -> None:
    old_spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    new_spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    old_val = _val({"ch": DirectValue(2)})

    result = inherit_from(old_val, old_spec, new_spec)
    ch = result.fields["ch"]
    assert isinstance(ch, DirectValue)
    assert ch.value == 2


def test_channel_scalar_inherits_eval_value() -> None:
    old_spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    new_spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    old_eval = EvalValue(expr="qub_ch", resolved=2)
    old_val = _val({"ch": old_eval})

    result = inherit_from(old_val, old_spec, new_spec)

    assert result.fields["ch"] is old_eval


def test_channel_missing_key_uses_scalar_default() -> None:
    old_spec = _section({})
    new_spec = _section({"ch": ScalarSpec(label="Gen ch", type=int)})
    old_val = _val({})

    result = inherit_from(old_val, old_spec, new_spec)
    ch = result.fields["ch"]
    assert isinstance(ch, DirectValue)
    assert ch.value == 0

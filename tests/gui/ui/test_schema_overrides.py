# pyright: reportAttributeAccessIssue=false
"""Tests for schema_overrides."""

from __future__ import annotations

from typing import cast

import pytest
from zcu_tools.gui.adapter import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
    SweepValue,
)
from zcu_tools.gui.schema_overrides import (
    apply_schema_overrides,
    hide_field,
    lock_field,
    set_default_value,
    set_field_choices,
    set_field_label,
)


def make_dummy_schema() -> CfgSchema:
    spec = CfgSectionSpec(
        fields={
            "a": ScalarSpec("A", type=int, editable=True, hidden=False),
            "b": CfgSectionSpec(fields={"c": ScalarSpec("C", type=float)}),
        }
    )
    val = CfgSectionValue(
        fields={
            "a": DirectValue(1),
            "b": CfgSectionValue(fields={"c": DirectValue(2.0)}),
        }
    )
    return CfgSchema(spec, val)


def test_lock_field():
    s = make_dummy_schema()
    s2 = lock_field(s, "a")
    assert getattr(s2.spec.fields["a"], "editable") is False

    s3 = lock_field(s, "b.c")
    assert (
        getattr(cast(CfgSectionSpec, s3.spec.fields["b"]).fields["c"], "editable")
        is False
    )

    with pytest.raises(RuntimeError, match="Path 'x' does not point to a ScalarSpec"):
        lock_field(s, "x")


def test_hide_field():
    s = make_dummy_schema()
    s2 = hide_field(s, "a")
    assert getattr(s2.spec.fields["a"], "hidden") is True


def test_set_default_value():
    s = make_dummy_schema()
    s2 = set_default_value(s, "a", 100)
    assert cast(DirectValue, s2.value.fields["a"]).value == 100

    s3 = set_default_value(s, "b.c", 3.14)
    assert cast(CfgSectionValue, s3.value.fields["b"]).fields["c"].value == 3.14

    with pytest.raises(RuntimeError, match="does not exist"):
        set_default_value(s, "x", 1)


def test_set_field_label():
    s = make_dummy_schema()
    s2 = set_field_label(s, "a", "New A")
    assert s2.spec.fields["a"].label == "New A"


def test_set_field_choices():
    s = make_dummy_schema()
    s2 = set_field_choices(s, "a", [1, 2, 3])
    assert cast(ScalarSpec, s2.spec.fields["a"]).choices == [1, 2, 3]


def test_apply_schema_overrides():
    s = make_dummy_schema()

    s2 = apply_schema_overrides(
        s,
        spec_overrides={
            "a": {"editable": False, "hidden": True, "label": "L", "choices": [1]},
            "b.c": {"label": "L2"},
        },
        value_overrides={"a": 5, "b.c": 6.0},
    )

    assert getattr(s2.spec.fields["a"], "editable") is False
    assert getattr(s2.spec.fields["a"], "hidden") is True
    assert s2.spec.fields["a"].label == "L"
    assert cast(ScalarSpec, s2.spec.fields["a"]).choices == [1]

    assert cast(DirectValue, s2.value.fields["a"]).value == 5
    assert cast(CfgSectionValue, s2.value.fields["b"]).fields["c"].value == 6.0


def test_apply_schema_overrides_errors():
    s = make_dummy_schema()

    with pytest.raises(RuntimeError, match="Unsupported editable"):
        apply_schema_overrides(s, spec_overrides={"a": {"editable": "invalid"}})

    with pytest.raises(RuntimeError, match="Unsupported hidden"):
        apply_schema_overrides(s, spec_overrides={"a": {"hidden": "invalid"}})

    with pytest.raises(RuntimeError, match="must be str"):
        apply_schema_overrides(s, spec_overrides={"a": {"label": 123}})

    with pytest.raises(RuntimeError, match="must be list"):
        apply_schema_overrides(s, spec_overrides={"a": {"choices": 123}})

    with pytest.raises(RuntimeError, match="Unsupported spec override key"):
        apply_schema_overrides(s, spec_overrides={"a": {"unknown": 123}})


def test_coerce_value_for_node():
    s = make_dummy_schema()

    # test SweepValue mismatch
    with pytest.raises(RuntimeError, match="requires SweepValue"):
        s.value.fields["a"] = SweepValue(start=0, stop=1, expts=2)
        set_default_value(s, "a", 10)

    # test section mismatch
    with pytest.raises(RuntimeError, match="requires CfgSectionValue"):
        set_default_value(s, "b", 10)

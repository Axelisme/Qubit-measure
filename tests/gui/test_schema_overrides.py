"""Tests for path-based schema override helpers."""

from __future__ import annotations

from typing import cast

import pytest
from zcu_tools.gui.adapter import CfgSchema, ScalarSpec, ScalarValue
from zcu_tools.gui.schema_overrides import (
    apply_schema_overrides,
    hide_field,
    lock_field,
    set_default_value,
)
from zcu_tools.gui.specs.readout import make_pulse_readout_spec


def _make_schema() -> CfgSchema:
    spec = make_pulse_readout_spec()
    from zcu_tools.gui.adapter import make_default_value

    value = make_default_value(spec)
    return CfgSchema(spec=spec, value=value)


def test_lock_field_marks_scalar_non_editable():
    schema = _make_schema()

    updated = lock_field(schema, "pulse_cfg.freq")

    assert schema.spec.fields["pulse_cfg"].fields["freq"].editable is True  # type: ignore[union-attr]
    assert updated.spec.fields["pulse_cfg"].fields["freq"].editable is False  # type: ignore[union-attr]


def test_hide_field_marks_scalar_hidden():
    schema = _make_schema()

    updated = hide_field(schema, "ro_cfg.ro_freq")

    assert schema.spec.fields["ro_cfg"].fields["ro_freq"].hidden is False  # type: ignore[union-attr]
    assert updated.spec.fields["ro_cfg"].fields["ro_freq"].hidden is True  # type: ignore[union-attr]


def test_set_default_value_updates_scalar_without_mutating_input():
    schema = _make_schema()

    updated = set_default_value(schema, "pulse_cfg.freq", 0.0)

    assert schema.value.fields["pulse_cfg"].fields["freq"] == ScalarValue(0.0)  # type: ignore[union-attr]
    assert updated.value.fields["pulse_cfg"].fields["freq"] == ScalarValue(0.0)  # type: ignore[union-attr]
    assert updated.value.fields["pulse_cfg"] is not schema.value.fields["pulse_cfg"]  # type: ignore[comparison-overlap]


def test_apply_schema_overrides_combines_spec_and_value_updates():
    schema = _make_schema()

    updated = apply_schema_overrides(
        schema,
        spec_overrides={"pulse_cfg.freq": {"editable": False, "hidden": True}},
        value_overrides={"pulse_cfg.freq": 123.4},
    )

    pulse_spec = cast(CfgSchema, updated).spec.fields["pulse_cfg"]
    freq_spec = cast(ScalarSpec, pulse_spec.fields["freq"])  # type: ignore[union-attr]
    assert freq_spec.editable is False
    assert freq_spec.hidden is True
    assert updated.value.fields["pulse_cfg"].fields["freq"] == ScalarValue(123.4)  # type: ignore[union-attr]


def test_lock_field_missing_path_fails_fast():
    schema = _make_schema()

    with pytest.raises(RuntimeError):
        lock_field(schema, "pulse_cfg.missing")


def test_hide_field_non_scalar_fails_fast():
    schema = _make_schema()

    with pytest.raises(RuntimeError):
        hide_field(schema, "pulse_cfg")

"""Unit tests for validate_schema — static value-tree validation.

Validates the *static* contract (structure complete, LiteralSpec == spec.value,
DirectValue scalar type/choices, None only on optional refs). The *dynamic*
contract (required has a value, EvalValue resolves) is enforced by lowering.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import (
    CenteredSweepSpec,
    CenteredSweepValue,
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    LiteralSpec,
    ReferenceSpec,
    ReferenceValue,
    ScalarSpec,
    SweepSpec,
    SweepValue,
)
from zcu_tools.gui.app.main.adapter.lowering import validate_schema


def _ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    return ml


def _inner_spec() -> CfgSectionSpec:
    return CfgSectionSpec(
        label="Pulse",
        fields={"type": LiteralSpec("pulse"), "gain": ScalarSpec("Gain", float)},
    )


# --- structure completeness -------------------------------------------------


def test_complete_value_passes():
    spec = CfgSectionSpec(fields={"reps": ScalarSpec("Reps", int)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"reps": DirectValue(5)})
    )
    validate_schema(schema, _ml())  # no raise


def test_missing_key_raises():
    spec = CfgSectionSpec(
        fields={"reps": ScalarSpec("Reps", int), "rounds": ScalarSpec("Rounds", int)}
    )
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"reps": DirectValue(5)})
    )
    with pytest.raises(RuntimeError, match="'rounds' is missing"):
        validate_schema(schema, _ml())


def test_missing_nested_key_raises():
    spec = CfgSectionSpec(
        fields={"sub": CfgSectionSpec(fields={"x": ScalarSpec("X", int)})}
    )
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"sub": CfgSectionValue(fields={})})
    )
    with pytest.raises(RuntimeError, match="'sub.x' is missing"):
        validate_schema(schema, _ml())


def test_unknown_field_raises():
    spec = CfgSectionSpec(fields={"reps": ScalarSpec("Reps", int)})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(fields={"reps": DirectValue(5), "extra": DirectValue(1)}),
    )
    with pytest.raises(RuntimeError, match="unknown fields"):
        validate_schema(schema, _ml())


# --- LiteralSpec ------------------------------------------------------------


def test_literal_equal_passes():
    spec = CfgSectionSpec(fields={"t": LiteralSpec(value=1)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"t": DirectValue(1)}))
    validate_schema(schema, _ml())


def test_literal_mismatch_raises():
    spec = CfgSectionSpec(fields={"t": LiteralSpec(value=1)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"t": DirectValue(2)}))
    with pytest.raises(RuntimeError, match="locked literal"):
        validate_schema(schema, _ml())


# --- scalar type (widen-only: int->float OK, float->int reject) -------------


def test_int_widens_to_float_field():
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", float)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"x": DirectValue(5)}))
    validate_schema(schema, _ml())  # int 5 acceptable for a float field


def test_float_does_not_narrow_to_int_field():
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", int)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"x": DirectValue(5.0)}))
    with pytest.raises(RuntimeError, match="not compatible with spec type int"):
        validate_schema(schema, _ml())


def test_bool_not_accepted_as_int():
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", int)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"x": DirectValue(True)})
    )
    with pytest.raises(RuntimeError, match="not compatible"):
        validate_schema(schema, _ml())


def test_str_type_mismatch_raises():
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", str)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"x": DirectValue(1)}))
    with pytest.raises(RuntimeError, match="not compatible"):
        validate_schema(schema, _ml())


def test_string_on_float_field_reports_not_coerced():
    # A string standing in for a numeric field is the un-coerced-MCP-value case:
    # the literal "0.2" is a valid float value, it just kept the wrong type. The
    # message must call out the missing coercion, not "incompatible value".
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", float)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"x": DirectValue("0.2")})
    )
    with pytest.raises(RuntimeError, match="received string '0.2' where a float"):
        validate_schema(schema, _ml())


def test_string_on_int_field_reports_not_coerced():
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", int)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"x": DirectValue("5")}))
    with pytest.raises(RuntimeError, match="not coerced"):
        validate_schema(schema, _ml())


def test_real_float_on_float_field_passes():
    # The counterpart to the above: a real float (the value an UNTYPED JSON schema
    # lets the MCP client send through unchanged) validates without complaint.
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", float)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"x": DirectValue(0.2)}))
    validate_schema(schema, _ml())


def test_unset_scalar_none_passes():
    # A None DirectValue is "unset" — legal static state (required-has-value is
    # a dynamic check, enforced by lowering, not validate).
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", int, required=True)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"x": DirectValue(None)})
    )
    validate_schema(schema, _ml())


def test_eval_value_skips_type_check():
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", int)})
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"x": EvalValue(expr="q_f")})
    )
    validate_schema(schema, _ml())  # EvalValue type fixed at resolve time, skipped


# --- choices ----------------------------------------------------------------


def test_choices_in_passes():
    spec = CfgSectionSpec(fields={"nqz": ScalarSpec("Nqz", int, choices=[1, 2])})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"nqz": DirectValue(2)}))
    validate_schema(schema, _ml())


def test_choices_out_raises():
    spec = CfgSectionSpec(fields={"nqz": ScalarSpec("Nqz", int, choices=[1, 2])})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"nqz": DirectValue(3)}))
    with pytest.raises(RuntimeError, match="not in allowed choices"):
        validate_schema(schema, _ml())


# --- None on refs -----------------------------------------------------------


def test_none_on_optional_ref_passes():
    spec = CfgSectionSpec(
        fields={
            "m": ReferenceSpec(kind="module", allowed=[_inner_spec()], optional=True)
        }
    )
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"m": None}))
    validate_schema(schema, _ml())


def test_none_on_required_ref_raises():
    spec = CfgSectionSpec(
        fields={
            "m": ReferenceSpec(kind="module", allowed=[_inner_spec()], optional=False)
        }
    )
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"m": None}))
    with pytest.raises(RuntimeError, match="not a disabled optional ref"):
        validate_schema(schema, _ml())


def test_none_on_scalar_raises():
    spec = CfgSectionSpec(fields={"x": ScalarSpec("X", int)})
    schema = CfgSchema(spec=spec, value=CfgSectionValue(fields={"x": None}))
    with pytest.raises(RuntimeError, match="not a disabled optional ref"):
        validate_schema(schema, _ml())


# --- ref recursion ----------------------------------------------------------


def test_enabled_ref_recurses_into_sub_value():
    inner = _inner_spec()
    spec = CfgSectionSpec(
        fields={"m": ReferenceSpec(kind="module", allowed=[inner], label="M")}
    )
    # the ref's sub-value omits 'gain' → incomplete → raises at m.gain
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={
                "m": ReferenceValue(
                    "<Custom:Pulse>",
                    CfgSectionValue(fields={"type": DirectValue("pulse")}),
                )
            }
        ),
    )
    with pytest.raises(RuntimeError, match="'m.gain' is missing"):
        validate_schema(schema, _ml())


def test_enabled_ref_complete_sub_value_passes():
    inner = _inner_spec()
    spec = CfgSectionSpec(
        fields={"m": ReferenceSpec(kind="module", allowed=[inner], label="M")}
    )
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(
            fields={
                "m": ReferenceValue(
                    "<Custom:Pulse>",
                    CfgSectionValue(
                        fields={
                            "type": DirectValue("pulse"),
                            "gain": DirectValue(0.5),
                        }
                    ),
                )
            }
        ),
    )
    validate_schema(schema, _ml())


# --- sweep ------------------------------------------------------------------


def test_sweep_value_passes():
    spec = CfgSectionSpec(fields={"s": SweepSpec(label="S")})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(fields={"s": SweepValue(0.0, 1.0, 11)}),
    )
    validate_schema(schema, _ml())


def test_centered_sweep_value_passes():
    spec = CfgSectionSpec(fields={"s": CenteredSweepSpec(label="S")})
    schema = CfgSchema(
        spec=spec,
        value=CfgSectionValue(fields={"s": CenteredSweepValue(0.0, 1.0, 11)}),
    )
    validate_schema(schema, _ml())

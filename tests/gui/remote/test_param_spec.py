"""Tests for the ParamSpec validation engine.

Semantics must match the legacy wire._require_* / _optional_* helpers exactly:
non-empty required strings, bool-rejecting integers/numbers, JSON-safe values.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.param_spec import (
    JsonType,
    ParamSpec,
    build_input_schema,
    schema_property,
    validate_params,
)


def _spec(json_type: JsonType, *, required=True, default=None) -> tuple[ParamSpec, ...]:
    return (ParamSpec("x", json_type, required=required, default=default),)


def test_number_pairs_decode_owned_finite_coordinates():
    from zcu_tools.gui.remote.param_spec import NumberPairs

    raw = [[1, 2.5], [-3.0, 4]]
    decoded = validate_params(
        (ParamSpec("vertices", JsonType.NUMBER_PAIRS),), {"vertices": raw}
    )
    pairs = decoded["vertices"]
    assert isinstance(pairs, NumberPairs)
    assert pairs.values == ((1.0, 2.5), (-3.0, 4.0))
    raw[0][0] = 99
    assert pairs.values[0] == (1.0, 2.5)
    again = validate_params(
        (ParamSpec("vertices", JsonType.NUMBER_PAIRS),), {"vertices": pairs}
    )
    assert again["vertices"] == pairs


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        "0,1",
        [[0]],
        [[0, 1, 2]],
        [(0, 1)],
        [[True, 1]],
        [["0", 1]],
        [[float("nan"), 1]],
        [[0, float("inf")]],
        [[10**1000, 1]],
        [0, 1],
    ],
)
def test_number_pairs_reject_invalid_wire_values(value):
    with pytest.raises(RemoteError) as error:
        validate_params(
            (ParamSpec("vertices", JsonType.NUMBER_PAIRS),), {"vertices": value}
        )
    assert error.value.code == ErrorCode.INVALID_PARAMS


def test_number_pairs_schema_is_numeric_nested_array():
    schema = build_input_schema((ParamSpec("vertices", JsonType.NUMBER_PAIRS),))
    assert schema["properties"] == {
        "vertices": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number"},
            },
        }
    }


def test_required_string_accepts_non_empty():
    assert validate_params(_spec(JsonType.STRING), {"x": "hi"}) == {"x": "hi"}


def test_required_string_rejects_empty():
    with pytest.raises(RemoteError) as e:
        validate_params(_spec(JsonType.STRING), {"x": ""})
    assert e.value.code is ErrorCode.INVALID_PARAMS


def test_required_string_rejects_missing():
    with pytest.raises(RemoteError, match="missing 'x'"):
        validate_params(_spec(JsonType.STRING), {})


def test_required_string_rejects_wrong_type():
    with pytest.raises(RemoteError, match="must be a string"):
        validate_params(_spec(JsonType.STRING), {"x": 5})


def test_optional_string_allows_empty_and_missing():
    assert validate_params(_spec(JsonType.STRING, required=False), {"x": ""}) == {
        "x": ""
    }
    assert validate_params(_spec(JsonType.STRING, required=False), {}) == {"x": None}


def test_string_enum_rejects_non_member_after_type_validation():
    specs = (ParamSpec("mode", JsonType.STRING, enum=("fast", "careful")),)
    assert validate_params(specs, {"mode": "fast"}) == {"mode": "fast"}
    with pytest.raises(RemoteError, match="mode") as error:
        validate_params(specs, {"mode": "other"})
    assert error.value.code is ErrorCode.INVALID_PARAMS


@pytest.mark.parametrize(
    "json_type,enum,default",
    [
        (JsonType.NUMBER, ("fast",), None),
        (JsonType.STRING, (), None),
        (JsonType.STRING, ("fast", 1), None),
        (JsonType.STRING, ("fast", "fast"), None),
        (JsonType.STRING, ("fast", "careful"), "other"),
    ],
)
def test_enum_declaration_rejects_invalid_contract(json_type, enum, default):
    with pytest.raises(ValueError, match="enum"):
        ParamSpec("mode", json_type, default=default, enum=enum)


def test_string_enum_is_projected_by_shared_schema():
    spec = ParamSpec("mode", JsonType.STRING, enum=("fast", "careful"))
    assert schema_property(spec) == {
        "type": "string",
        "enum": ["fast", "careful"],
    }


@pytest.mark.parametrize("default", [None, "fast"])
@pytest.mark.parametrize("params", [{}, {"mode": None}])
def test_optional_enum_preserves_omission_and_null_default(default, params):
    spec = ParamSpec(
        "mode",
        JsonType.STRING,
        required=False,
        default=default,
        enum=("fast", "careful"),
    )
    assert validate_params((spec,), params) == {"mode": default}


def test_integer_rejects_bool():
    with pytest.raises(RemoteError, match="must be an integer"):
        validate_params(_spec(JsonType.INTEGER), {"x": True})


def test_integer_accepts_int():
    assert validate_params(_spec(JsonType.INTEGER), {"x": 7}) == {"x": 7}


def test_number_rejects_bool_and_coerces_int():
    with pytest.raises(RemoteError, match="must be a number"):
        validate_params(_spec(JsonType.NUMBER), {"x": False})
    assert validate_params(_spec(JsonType.NUMBER), {"x": 3}) == {"x": 3.0}


@pytest.mark.parametrize("value", [10**1000, -(10**1000)], ids=["positive", "negative"])
def test_number_conversion_overflow_is_invalid_params(value):
    with pytest.raises(RemoteError, match="representable") as error:
        validate_params(_spec(JsonType.NUMBER), {"x": value})
    assert error.value.code == ErrorCode.INVALID_PARAMS
    assert isinstance(error.value.__cause__, OverflowError)


def test_boolean_requires_bool():
    assert validate_params(_spec(JsonType.BOOLEAN), {"x": True}) == {"x": True}
    with pytest.raises(RemoteError, match="must be a boolean"):
        validate_params(_spec(JsonType.BOOLEAN), {"x": 1})


def test_optional_with_default_returns_default_when_absent():
    assert validate_params(
        _spec(JsonType.BOOLEAN, required=False, default=True), {}
    ) == {"x": True}


def test_object_requires_dict():
    assert validate_params(_spec(JsonType.OBJECT), {"x": {"a": 1}}) == {"x": {"a": 1}}
    with pytest.raises(RemoteError, match="must be an object"):
        validate_params(_spec(JsonType.OBJECT), {"x": [1, 2]})


def test_json_accepts_serializable_rejects_not():
    assert validate_params(_spec(JsonType.JSON), {"x": [1, "a", None]}) == {
        "x": [1, "a", None]
    }
    with pytest.raises(RemoteError, match="JSON-serializable"):
        validate_params(_spec(JsonType.JSON), {"x": object()})


def test_extra_undeclared_params_ignored():
    out = validate_params(_spec(JsonType.STRING), {"x": "ok", "extra": 1})
    assert out == {"x": "ok"}


def test_build_input_schema_marks_required_and_types():
    specs = (
        ParamSpec("tab_id", JsonType.STRING, required=True),
        ParamSpec("flag", JsonType.BOOLEAN, required=False, default=False),
        ParamSpec("payload", JsonType.JSON, required=True),
    )
    schema = build_input_schema(specs)
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["tab_id"] == {"type": "string"}
    assert props["flag"] == {"type": "boolean"}
    # JSON => an UNTYPED schema (no "type" key) so the MCP client never coerces a
    # value against a string member (which would stringify a number e.g. 0.2).
    assert "type" not in props["payload"]
    assert set(schema.get("required", [])) == {"tab_id", "payload"}


def test_json_schema_property_is_untyped_but_keeps_description():
    # A JsonType.JSON property carries NO "type" key (untyped = any JSON value),
    # but a description, when present, is still rendered.
    prop = schema_property(ParamSpec("v", JsonType.JSON, description="any value"))
    assert "type" not in prop
    assert prop.get("description") == "any value"


def test_non_json_schema_property_keeps_its_type():
    # The other kinds still render a concrete "type" (only JSON goes untyped).
    assert schema_property(ParamSpec("n", JsonType.NUMBER)).get("type") == "number"
    assert schema_property(ParamSpec("b", JsonType.BOOLEAN)).get("type") == "boolean"


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"mode": "other"}, "missing 'first'"),
        ({"first": 7, "mode": "other"}, "'first' must be a string, got int"),
    ],
    ids=["missing", "wrong-type"],
)
def test_validation_keeps_first_spec_failure_before_later_enum(
    params: dict[str, object], message: str
):
    specs = (
        ParamSpec("first", JsonType.STRING),
        ParamSpec("mode", JsonType.STRING, enum=("fast", "careful")),
    )
    with pytest.raises(RemoteError) as error:
        validate_params(specs, params)
    assert error.value.code is ErrorCode.INVALID_PARAMS
    assert error.value.message == message
    assert error.value.__cause__ is None

"""Tests for the ParamSpec validation engine.

Semantics must match the legacy wire._require_* / _optional_* helpers exactly:
non-empty required strings, bool-rejecting integers/numbers, JSON-safe values.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.services.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.services.remote.param_spec import (
    JsonType,
    ParamSpec,
    build_input_schema,
    validate_params,
)


def _spec(json_type: JsonType, *, required=True, default=None) -> tuple[ParamSpec, ...]:
    return (ParamSpec("x", json_type, required=required, default=default),)


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


def test_integer_rejects_bool():
    with pytest.raises(RemoteError, match="must be an integer"):
        validate_params(_spec(JsonType.INTEGER), {"x": True})


def test_integer_accepts_int():
    assert validate_params(_spec(JsonType.INTEGER), {"x": 7}) == {"x": 7}


def test_number_rejects_bool_and_coerces_int():
    with pytest.raises(RemoteError, match="must be a number"):
        validate_params(_spec(JsonType.NUMBER), {"x": False})
    assert validate_params(_spec(JsonType.NUMBER), {"x": 3}) == {"x": 3.0}


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
    # JSON => a type-union covering any JSON value.
    assert props["payload"]["type"] == [
        "number",
        "string",
        "boolean",
        "object",
        "array",
        "null",
    ]
    assert set(schema["required"]) == {"tab_id", "payload"}

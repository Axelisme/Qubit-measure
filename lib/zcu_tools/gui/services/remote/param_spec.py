"""Single source of truth for a wire method's parameter contract.

A ``ParamSpec`` declares one parameter's name, JSON type, requiredness and
default. The dispatcher validates incoming params against a method's ParamSpec
tuple *before* calling the handler, so the handler receives already-typed
values. The same specs generate the MCP ``inputSchema`` (Step 7), so the wire
type contract and the runtime validation can never drift.

Validation semantics intentionally mirror the legacy ``wire._require_*`` /
``_optional_*`` helpers exactly:

- ``STRING`` required: must be a non-empty string.
- ``STRING`` optional: may be absent/None; if present must be a string (empty ok).
- ``INTEGER``: must be int, ``bool`` rejected (bool is an int subclass).
- ``NUMBER``: must be int/float, ``bool`` rejected; coerced to float.
- ``BOOLEAN``: must be bool.
- ``OBJECT``: must be a dict.
- ``JSON``: must be present and JSON-serializable (value passed through).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .errors import ErrorCode, RemoteError


class JsonType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    JSON = "json"  # any JSON-serializable value


@dataclass(frozen=True)
class ParamSpec:
    name: str
    json_type: JsonType
    required: bool = True
    default: object = None
    description: str = ""

    def _coerce(self, present: bool, value: object) -> object:
        if not present or value is None:
            if self.required:
                raise RemoteError(ErrorCode.INVALID_PARAMS, f"missing '{self.name}'")
            return self.default
        jt = self.json_type
        if jt is JsonType.STRING:
            if not isinstance(value, str):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"'{self.name}' must be a string, got {type(value).__name__}",
                )
            if self.required and not value:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS, f"'{self.name}' must be non-empty"
                )
            return value
        if jt is JsonType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"'{self.name}' must be an integer, got {type(value).__name__}",
                )
            return value
        if jt is JsonType.NUMBER:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"'{self.name}' must be a number, got {type(value).__name__}",
                )
            return float(value)
        if jt is JsonType.BOOLEAN:
            if not isinstance(value, bool):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"'{self.name}' must be a boolean, got {type(value).__name__}",
                )
            return value
        if jt is JsonType.OBJECT:
            if not isinstance(value, dict):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"'{self.name}' must be an object, got {type(value).__name__}",
                )
            return value
        if jt is JsonType.JSON:
            try:
                json.dumps(value)
            except (TypeError, ValueError) as exc:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"'{self.name}' must be JSON-serializable",
                ) from exc
            return value
        raise RemoteError(  # pragma: no cover - exhaustive guard
            ErrorCode.INTERNAL, f"unhandled json_type {jt!r}"
        )


def validate_params(
    specs: tuple[ParamSpec, ...], params: Mapping[str, object]
) -> dict[str, object]:
    """Validate ``params`` against ``specs``; return a name -> typed-value dict.

    Only declared params are surfaced. Required-but-missing or type-mismatched
    params raise ``RemoteError(INVALID_PARAMS)``. Extra undeclared params are
    ignored (the wire stays forward-compatible).
    """
    out: dict[str, object] = {}
    for spec in specs:
        present = spec.name in params
        out[spec.name] = spec._coerce(present, params.get(spec.name))
    return out


_ANY_JSON_TYPE = ["number", "string", "boolean", "object", "array", "null"]


def schema_property(spec: ParamSpec) -> dict[str, object]:
    """Render one ParamSpec as a JSON-schema property (for MCP inputSchema)."""
    json_schema_type: object = {
        JsonType.STRING: "string",
        JsonType.INTEGER: "integer",
        JsonType.NUMBER: "number",
        JsonType.BOOLEAN: "boolean",
        JsonType.OBJECT: "object",
        JsonType.JSON: _ANY_JSON_TYPE,  # any JSON value
    }[spec.json_type]
    prop: dict[str, object] = {"type": json_schema_type}
    if spec.description:
        prop["description"] = spec.description
    return prop


def build_input_schema(specs: tuple[ParamSpec, ...]) -> dict[str, object]:
    """Render a method's ParamSpec tuple as a JSON-schema object."""
    properties = {spec.name: schema_property(spec) for spec in specs}
    required = [spec.name for spec in specs if spec.required]
    schema: dict[str, object] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema

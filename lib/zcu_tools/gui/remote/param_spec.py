"""Single source of truth for a wire method's parameter contract.

A ``ParamSpec`` declares one parameter's name, JSON type, requiredness and
default, and optional string enum. Enum declarations fail fast on empty or
invalid choices and defaults. The dispatcher validates incoming params against a method's ParamSpec
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
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal, NotRequired, TypedDict

from .errors import ErrorCode, RemoteError

SchemaJsonType = Literal["string", "integer", "number", "boolean", "object", "array"]


class SchemaProperty(TypedDict, total=False):
    """One property schema; omitted type accepts any JSON value.

    type is the JSON primitive/container kind. items declares array element shape.
    minItems/maxItems are nonnegative array lengths. enum lists allowed strings.
    description is human-readable parameter guidance. Absent fields impose no
    corresponding constraint; defaults remain ParamSpec-owned, not schema fields.
    """

    type: SchemaJsonType
    items: SchemaProperty
    minItems: int
    maxItems: int
    enum: list[str]
    description: str


class InputSchema(TypedDict):
    """Method's agent-facing object schema.

    type is always object. properties maps visible parameter names to schemas.
    required lists mandatory visible names; omission means none are mandatory.
    """

    type: Literal["object"]
    properties: dict[str, SchemaProperty]
    required: NotRequired[list[str]]


class JsonType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    JSON = "json"  # any JSON-serializable value
    ARRAY = "array"  # homogeneous string list; emits {"type":"array","items":{"type":"string"}}
    NUMBER_PAIRS = "number_pairs"  # nonempty array of finite numeric pairs


@dataclass(frozen=True)
class NumberPairs:
    """Owned tuple of finite numeric pairs decoded from a nonempty JSON list.

    values contains (first, second) float pairs in caller-declared units.
    Construction validates/coerces these immutable pairs. from_wire also accepts
    an existing validated carrier for already-decoded command dispatch. Empty or
    malformed data, bool, nonfinite numbers and overflow raise
    RemoteError(INVALID_PARAMS), preserving conversion causes.
    """

    values: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _owned_number_pairs(self.values))

    @classmethod
    def from_wire(cls, value: object) -> NumberPairs:
        """Decode JSON list or retain validated carrier; INVALID_PARAMS rejects others."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, list) or not value:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "number pairs must be a nonempty JSON list"
            )
        pairs: list[tuple[float, float]] = []
        for pair in value:
            if not isinstance(pair, list) or len(pair) != 2:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    "each number pair must be a two-number JSON list",
                )
            pairs.append((_pair_number(pair[0]), _pair_number(pair[1])))
        return cls(tuple(pairs))


def _pair_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "number pairs require numeric values"
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "number pairs require finite values"
        ) from exc
    if not math.isfinite(number):
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "number pairs require finite values"
        )
    return number


def _owned_number_pairs(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, tuple) or not value:
        raise RemoteError(
            ErrorCode.INVALID_PARAMS, "number pairs must be a nonempty tuple"
        )
    pairs: list[tuple[float, float]] = []
    for pair in value:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS, "each number pair must have two values"
            )
        pairs.append((_pair_number(pair[0]), _pair_number(pair[1])))
    return tuple(pairs)


def _validate_string_enum(values: tuple[object, ...]) -> None:
    if not values or any(not isinstance(value, str) for value in values):
        raise ValueError("enum must be a non-empty tuple of strings")
    if len(set(values)) != len(values):
        raise ValueError("enum must not contain duplicates")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    json_type: JsonType
    required: bool = True
    default: object = None
    description: str = ""
    # When True the param is validated and reaches the handler as usual, but is
    # omitted from the MCP inputSchema — a wire-only param the mcp layer fills
    # (e.g. ``expected_versions``), never surfaced to the agent.
    mcp_hidden: bool = False
    enum: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.enum is None:
            return
        if self.json_type is not JsonType.STRING:
            raise ValueError("enum requires a STRING parameter")
        _validate_string_enum(self.enum)
        if self.default is not None and self.default not in self.enum:
            raise ValueError("default must belong to enum")

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
            try:
                return float(value)
            except OverflowError as exc:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"'{self.name}' must be representable as a float",
                ) from exc
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
        if jt is JsonType.ARRAY:
            if not isinstance(value, list):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    f"'{self.name}' must be a list, got {type(value).__name__}",
                )
            return value
        if jt is JsonType.NUMBER_PAIRS:
            return NumberPairs.from_wire(value)
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
    params raise ``RemoteError(INVALID_PARAMS)``. NUMBER values outside the
    float conversion range also raise INVALID_PARAMS, preserving the conversion
    error as the cause. Extra undeclared params are ignored (the wire stays
    forward-compatible).
    """
    out: dict[str, object] = {}
    for spec in specs:
        present = spec.name in params
        value = spec._coerce(present, params.get(spec.name))
        if spec.enum is not None and value is not None and value not in spec.enum:
            raise RemoteError(
                ErrorCode.INVALID_PARAMS,
                f"'{spec.name}' must be one of {spec.enum!r}",
            )
        out[spec.name] = value
    return out


def schema_property(spec: ParamSpec) -> SchemaProperty:
    """Render one ParamSpec as a JSON-schema property (for MCP inputSchema).

    ``JsonType.JSON`` renders with NO ``type`` key at all — an untyped schema is
    the correct JSON-schema spelling of "any JSON value". A ``type`` union that
    lists ``"string"`` lets the MCP client coerce a number (e.g. ``0.2``) against
    the string member and send ``"0.2"``, which then fails a downstream float
    field check. Omitting ``type`` means the client passes the value through
    untouched (a number stays a number), so a JSON param never gets stringified.
    """
    prop: SchemaProperty = {}
    if spec.json_type is JsonType.ARRAY:
        # Emit typed array schema with string items; all current ARRAY params are
        # string lists.  A concrete "type" is required so the MCP client does not
        # stringify the whole array (the failure mode of the old J.JSON spelling).
        prop["type"] = "array"
        prop["items"] = {"type": "string"}
    elif spec.json_type is JsonType.NUMBER_PAIRS:
        prop.update(
            type="array",
            minItems=1,
            items={
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "number"},
            },
        )
    elif spec.json_type is not JsonType.JSON:
        scalar_types: dict[JsonType, SchemaJsonType] = {
            JsonType.STRING: "string",
            JsonType.INTEGER: "integer",
            JsonType.NUMBER: "number",
            JsonType.BOOLEAN: "boolean",
            JsonType.OBJECT: "object",
        }
        prop["type"] = scalar_types[spec.json_type]
    if spec.enum is not None:
        prop["enum"] = list(spec.enum)
    if spec.description:
        prop["description"] = spec.description
    return prop


def build_input_schema(specs: tuple[ParamSpec, ...]) -> InputSchema:
    """Render a method's ParamSpec tuple as a JSON-schema object.

    ``mcp_hidden`` params are wire-only (mcp-filled) and excluded from the
    agent-facing schema.
    """
    visible = tuple(spec for spec in specs if not spec.mcp_hidden)
    properties = {spec.name: schema_property(spec) for spec in visible}
    required = [spec.name for spec in visible if spec.required]
    schema: InputSchema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema

"""Regression tests for ARRAY params in generate_tools / param_spec.

Root cause: J.JSON emits no "type" key in the MCP inputSchema, so MCP clients
stringify the whole array.  J.ARRAY must emit {"type": "array", "items":
{"type": "string"}} so the client passes the list through untouched.

Two invariants pinned here:
  1. build_input_schema emits {"type": "array"} for ARRAY params.
  2. A generate_tools forwarder passes a multi-element list through to the
     handler intact (no char-split).
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import (
    JsonType,
    ParamSpec,
    build_input_schema,
    schema_property,
    validate_params,
)

# ---------------------------------------------------------------------------
# Invariant 1: schema_property / build_input_schema for ARRAY
# ---------------------------------------------------------------------------


def test_array_schema_property_has_type_array():
    """ARRAY must emit {"type": "array"} so the MCP client does not stringify."""
    prop = schema_property(ParamSpec("paths", JsonType.ARRAY))
    assert prop["type"] == "array"


def test_array_schema_property_has_string_items():
    """items must be {"type": "string"} — all current ARRAY params are string lists."""
    prop = schema_property(ParamSpec("paths", JsonType.ARRAY))
    assert prop["items"] == {"type": "string"}


def test_array_schema_property_keeps_description():
    prop = schema_property(ParamSpec("ids", JsonType.ARRAY, description="some ids"))
    assert prop["description"] == "some ids"
    assert prop["type"] == "array"


def test_build_input_schema_marks_array_param():
    specs = (
        ParamSpec("paths", JsonType.ARRAY, required=True),
        ParamSpec("mode", JsonType.STRING, required=False, default="write"),
    )
    schema = build_input_schema(specs)
    props = schema["properties"]
    assert isinstance(props, dict)
    paths_prop = props["paths"]
    assert isinstance(paths_prop, dict)
    assert paths_prop["type"] == "array"
    assert paths_prop["items"] == {"type": "string"}
    assert schema["required"] == ["paths"]


# ---------------------------------------------------------------------------
# Invariant 2: generate_tools forwarder passes list through intact
# ---------------------------------------------------------------------------


def test_forwarder_passes_list_intact():
    """A multi-element list sent to a ARRAY-typed forwarder must reach the
    handler as-is — no char-split, no stringify."""
    from zcu_tools.mcp.core.bridge import make_forwarder

    received: list[tuple[object, float]] = []

    def _send(method: str, params: dict, *, timeout_seconds: float) -> dict:
        received.append((params.get("paths"), timeout_seconds))
        return {"ok": True}

    spec = MethodSpec(
        5.0,
        "dummy",
        params=(
            ParamSpec("paths", JsonType.ARRAY),
            ParamSpec("mode", JsonType.STRING, required=False, default="write"),
        ),
    )
    forwarder = make_forwarder("claim", spec, _send)

    paths = ["lib/foo", "lib/bar", "lib/baz"]
    forwarder({"paths": paths})

    assert len(received) == 1
    assert received[0][0] == paths, f"expected {paths!r}, got {received[0][0]!r}"
    assert received[0][1] == pytest.approx(6.0)


def test_forwarder_does_not_char_split_serialised_array_string():
    """If MCP somehow sends a stringified array (the pre-fix failure mode),
    validate_params with ARRAY must reject it, not silently char-split."""
    specs = (ParamSpec("paths", JsonType.ARRAY),)
    stringified = "['lib/foo', 'lib/bar']"
    from zcu_tools.gui.remote.errors import RemoteError

    with pytest.raises(RemoteError, match="must be a list"):
        validate_params(specs, {"paths": stringified})

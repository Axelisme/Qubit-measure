"""Guards for MCP tool generation from the wire-contract SSOT.

The MCP tool table is assembled from METHOD_SPECS (generated 1:1 RPC tools)
overlaid with a hand-written override subset. These tests pin the invariants
that keep the agent-facing tool contract stable and drift-free.
"""

from __future__ import annotations

from zcu_tools.gui.services.remote import mcp_server as m
from zcu_tools.gui.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.services.remote.method_specs import METHOD_SPECS


def test_handlers_and_specs_match():
    # dispatch binds exactly one handler per spec; the registry is built from it.
    assert set(METHOD_REGISTRY) == set(METHOD_SPECS)


def test_every_tool_has_callable_handler_and_object_schema():
    for name, tool in m.TOOLS.items():
        assert callable(tool["handler"]), name
        assert isinstance(tool["description"], str), name
        schema = tool["inputSchema"]
        assert schema["type"] == "object", name
        assert isinstance(schema["properties"], dict), name


def test_generated_and_override_sets_are_disjoint():
    generated = m._generate_tools()
    overrides = {n for n in m._OVERRIDE_TOOLS if n in m._OVERRIDE_NAMES}
    assert set(generated).isdisjoint(overrides)


def test_every_non_generated_method_is_covered_by_an_override():
    # Each method excluded from generation must surface via an override tool,
    # otherwise an RPC method would have no MCP entry point.
    generated_methods = set(METHOD_SPECS) - m._NON_GENERATED_METHODS
    generated_tool_names = {
        m._tool_name_for(meth, METHOD_SPECS[meth]) for meth in generated_methods
    }
    assert generated_tool_names.issubset(set(m.TOOLS))


def test_generated_required_params_appear_in_schema():
    # Spot-check that a required param shows up in the generated schema's
    # "required" list and a typed property.
    tool = m.TOOLS["gui_tab_new"]
    assert tool["inputSchema"]["required"] == ["adapter_name"]
    assert tool["inputSchema"]["properties"]["adapter_name"]["type"] == "string"


def test_generated_optional_param_not_required():
    tool = m.TOOLS["gui_tab_snapshot"]
    assert "required" not in tool["inputSchema"]
    assert "tab_id" in tool["inputSchema"]["properties"]

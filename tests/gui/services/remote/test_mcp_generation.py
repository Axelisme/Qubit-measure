"""Guards for MCP tool generation from the wire-contract SSOT.

The MCP tool table is assembled from METHOD_SPECS (generated 1:1 RPC tools)
overlaid with a hand-written override subset. These tests pin the invariants
that keep the agent-facing tool contract stable and drift-free.
"""

from __future__ import annotations

from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
from zcu_tools.mcp.core.bridge import generate_tools
from zcu_tools.mcp.measure import server as m


def _tool_name_for(method: str, spec) -> str:
    # The shared generate_tools derives a tool name as spec.tool_name or the
    # app's tool_prefix + method-with-dots-as-underscores. Mirror that here so
    # the coverage assertion below stays pinned to the same naming rule.
    return spec.tool_name or m._CONFIG.tool_prefix + method.replace(".", "_")


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
    generated = generate_tools(
        m._CONFIG, METHOD_SPECS, m._NON_GENERATED_METHODS, m.send_gui_rpc
    )
    overrides = {n for n in m._OVERRIDE_TOOLS if n in m._OVERRIDE_NAMES}
    assert set(generated).isdisjoint(overrides)


def test_every_non_generated_method_is_covered_by_an_override():
    # Each method excluded from generation must surface via an override tool,
    # otherwise an RPC method would have no MCP entry point.
    generated_methods = set(METHOD_SPECS) - m._NON_GENERATED_METHODS
    generated_tool_names = {
        _tool_name_for(meth, METHOD_SPECS[meth]) for meth in generated_methods
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


def test_cfg_editor_tools_generated():
    # editor.commit's MCP tool is named gui_editor_save_as_module (tool_name
    # override): its real semantics are "register the draft as an ml module/
    # waveform", not "apply a tab cfg edit" (those edits are already live).
    expected = {
        "gui_editor_open",
        "gui_editor_set_field",
        "gui_editor_get",
        "gui_editor_save_as_module",
        "gui_editor_discard",
    }
    assert expected.issubset(set(m.TOOLS))

    open_tool = m.TOOLS["gui_editor_open"]
    # editor.open is modify-only (from_name); the blank-by-discriminator surface
    # was removed (create a blank via ml.create_from_role(role_id='<disc>:blank')).
    assert set(open_tool["inputSchema"]["required"]) == {"item_kind", "from_name"}
    props = open_tool["inputSchema"]["properties"]
    assert "discriminator" not in props
    assert "from_name" in props

    # 'value' is a JSON kind (scalar OR the tagged eval object) — its schema
    # must not pin a single type.
    value_schema = m.TOOLS["gui_editor_set_field"]["inputSchema"]["properties"]["value"]
    assert "type" not in value_schema or isinstance(value_schema.get("type"), list)


def test_writeback_set_selected_is_boolean_schema():
    """``selected`` must render as a boolean schema so the client sends a real
    boolean. A JSON-any schema lets the client send the string "false", which
    ``bool("false")`` wrongly reads as True (selection never clears)."""
    props = m.TOOLS["gui_writeback_set"]["inputSchema"]["properties"]
    assert props["selected"]["type"] == "boolean"

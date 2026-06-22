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
    # editor.commit's MCP tool is named gui_editor_save (tool_name override):
    # its real semantics are "register the draft as an ml module/waveform",
    # not "apply a tab cfg edit" (those edits are already live).
    expected = {
        "gui_editor_new",
        "gui_editor_set_field",
        "gui_editor_get",
        "gui_editor_save",
        "gui_editor_discard",
    }
    assert expected.issubset(set(m.TOOLS))

    # Old tool names must be absent.
    assert "gui_editor_open" not in m.TOOLS
    assert "gui_editor_save_as_module" not in m.TOOLS

    open_tool = m.TOOLS["gui_editor_new"]
    # editor.new is modify-only (from_name); the blank-by-discriminator surface
    # was removed (create a blank via context.ml_create_from_role(role_id='<disc>:blank')).
    assert set(open_tool["inputSchema"]["required"]) == {"item_kind", "from_name"}
    props = open_tool["inputSchema"]["properties"]
    assert "discriminator" not in props
    assert "from_name" in props

    # editor.get is a tree view keyed by editor_id with an optional dotted prefix;
    # the old flat-serving knobs (verbosity / under) are gone.
    get_tool = m.TOOLS["gui_editor_get"]
    assert get_tool["inputSchema"]["required"] == ["editor_id"]
    get_props = get_tool["inputSchema"]["properties"]
    assert "prefix" in get_props
    assert "verbosity" not in get_props
    assert "under" not in get_props

    # 'value' is a JSON kind (scalar OR the tagged eval object): its schema is
    # UNTYPED (no "type" key) so the MCP client never coerces a number against a
    # "string" member and stringifies it (e.g. 0.2 -> "0.2", which then fails the
    # downstream float-field check).
    value_schema = m.TOOLS["gui_editor_set_field"]["inputSchema"]["properties"]["value"]
    assert "type" not in value_schema


def test_tab_new_is_a_pure_generated_forwarder():
    """gui_tab_new is the auto-generated tab.new forwarder (MCP 45): it returns
    just {tab_id}; the fan-out + guide fold moved to gui_tab_stage1. So tab.new
    must NOT be excluded from generation nor served by an override."""
    assert "tab.new" not in m._NON_GENERATED_METHODS
    assert "gui_tab_new" not in m._OVERRIDE_NAMES
    # The generated schema keeps the single required 'adapter_name' string param.
    schema = m.TOOLS["gui_tab_new"]["inputSchema"]
    assert schema["required"] == ["adapter_name"]
    assert schema["properties"]["adapter_name"]["type"] == "string"


def test_writeback_apply_is_a_pure_generated_forwarder():
    """gui_tab_writeback_apply is the auto-generated tab.writeback_apply forwarder
    (MCP 45 / Phase 170c): it returns just {applied_ids}; the save_data chaining
    moved to gui_tab_stage4. So tab.writeback_apply must NOT be excluded from
    generation nor served by an override, and the agent schema exposes only 'tab_id'."""
    assert "tab.writeback_apply" not in m._NON_GENERATED_METHODS
    assert "gui_tab_writeback_apply" not in m._OVERRIDE_NAMES
    schema = m.TOOLS["gui_tab_writeback_apply"]["inputSchema"]
    # expected_versions is mcp_hidden; save_data is gone.
    assert set(schema["properties"]) == {"tab_id"}


def test_writeback_set_selected_is_boolean_schema():
    """``selected`` must render as a boolean schema so the client sends a real
    boolean. A JSON-any schema lets the client send the string "false", which
    ``bool("false")`` wrongly reads as True (selection never clears)."""
    props = m.TOOLS["gui_tab_writeback_set"]["inputSchema"]["properties"]
    assert props["selected"]["type"] == "boolean"


def test_view_screenshot_not_generated():
    """view.screenshot returns raw base64 PNG — it must NOT be auto-generated into
    a gui_view_screenshot agent tool. The only entry point is gui_debug_screenshot
    (which decodes + writes a file). Mirrors the dialog.screenshot invariant."""
    # Must be in the exclusion set so the generator skips it.
    assert "view.screenshot" in m._NON_GENERATED_METHODS
    # The leaked tool must not appear in the assembled tool table.
    assert "gui_view_screenshot" not in m.TOOLS
    # dialog.screenshot is the existing precedent — verify it is still excluded too.
    assert "dialog.screenshot" in m._NON_GENERATED_METHODS
    assert "gui_dialog_screenshot" not in m.TOOLS


def test_phase169_removed_tools_absent():
    """Phase 169 dropped 9 redundant agent tools. The 7 A-class wire methods are
    gone from the contract entirely; app.shutdown / view.snapshot are kept as
    internal-only wire methods (gui_stop / overview consume them) but exposed as
    NO agent tool — so none of the 9 tools appears in the assembled table."""
    removed_tools = {
        "gui_tab_get_cfg_summary",
        "gui_adapter_cfg_spec",
        "gui_adapter_analyze_spec",
        "gui_tab_update_cfg",
        "gui_dialog_open",
        "gui_dialog_close",
        "gui_dialog_list_open",
        "gui_app_shutdown",
        "gui_view_snapshot",
    }
    assert removed_tools.isdisjoint(set(m.TOOLS))

    # A-class: the wire method is gone from the contract entirely.
    a_class = {
        "tab.get_cfg_summary",
        "adapter.cfg_spec",
        "adapter.analyze_spec",
        "tab.update_cfg",
        "dialog.open",
        "dialog.close",
        "dialog.list_open",
    }
    assert a_class.isdisjoint(set(METHOD_SPECS))

    # B-class: the wire method + handler stay (the dispatch key-match needs the
    # spec), but generation is suppressed via _NON_GENERATED_METHODS.
    assert "app.shutdown" in METHOD_SPECS
    assert "view.snapshot" in METHOD_SPECS
    assert "app.shutdown" in m._NON_GENERATED_METHODS
    assert "view.snapshot" in m._NON_GENERATED_METHODS


def test_phase170a_tab_cfg_io_tools():
    """Phase 170a tab cfg I/O normalization:
    - OLD raw tab.get_cfg wire method is removed entirely (A-class removal).
    - tab.list_paths is renamed to tab.get_cfg (now the value tree).
    - tab.set_cfg is a new wire method (edits param = array-of-objects, so
      it is non-generated; the override is gui_tab_set_cfg).
    - gui_tab_get_cfg is auto-generated from the renamed wire method.
    - gui_tab_list_paths no longer exists (the wire method is gone).
    """
    # Old raw tab.get_cfg wire method is gone from the contract.
    # (It was replaced by the renamed tree-returning method.)
    # The old raw shape {"raw": ...} is no longer a wire method at all.

    # tab.get_cfg now exists and maps to the value-tree handler.
    assert "tab.get_cfg" in METHOD_SPECS
    assert "tab.get_cfg" not in m._NON_GENERATED_METHODS
    # Auto-generated tool is present.
    assert "gui_tab_get_cfg" in m.TOOLS
    assert "gui_tab_get_cfg" not in m._OVERRIDE_NAMES

    # tab.list_paths is gone from the wire contract.
    assert "tab.list_paths" not in METHOD_SPECS
    assert "gui_tab_list_paths" not in m.TOOLS

    # tab.set_cfg is a new wire method (non-generated: edits is array-of-objects).
    assert "tab.set_cfg" in METHOD_SPECS
    assert "tab.set_cfg" in m._NON_GENERATED_METHODS
    # Hand-written override is present.
    assert "gui_tab_set_cfg" in m.TOOLS
    assert "gui_tab_set_cfg" in m._OVERRIDE_NAMES

    # Editor tools now require editor_id only (no tab_id branch).
    for tool_name in ("gui_editor_set_field", "gui_editor_set_fields"):
        props = m.TOOLS[tool_name]["inputSchema"]["properties"]
        assert "tab_id" not in props, f"{tool_name} must not expose 'tab_id' anymore"
        assert "editor_id" in props


def test_phase170c_save_writeback_tools():
    """Phase 170c save + writeback under tab.* normalization:
    - save.{data,image,post_image,result,set_paths} renamed to tab.save_*.
    - writeback.{preview,set,apply} renamed to tab.writeback_*.
    - Old MCP tool names (gui_save_*, gui_writeback_*) are absent.
    - New MCP tool names (gui_tab_save_*, gui_tab_writeback_*) are present.
    - All new methods are auto-generated (no overrides, no exclusions).
    """
    # New wire methods are present in the contract.
    new_wire_methods = {
        "tab.save_data",
        "tab.save_image",
        "tab.save_post_image",
        "tab.save_result",
        "tab.save_set_paths",
        "tab.writeback_preview",
        "tab.writeback_set",
        "tab.writeback_apply",
    }
    assert new_wire_methods.issubset(set(METHOD_SPECS))

    # Old wire methods are gone from the contract entirely.
    old_wire_methods = {
        "save.data",
        "save.image",
        "save.post_image",
        "save.result",
        "save.set_paths",
        "writeback.preview",
        "writeback.set",
        "writeback.apply",
    }
    assert old_wire_methods.isdisjoint(set(METHOD_SPECS))

    # New MCP tool names are present in the assembled table.
    new_tool_names = {
        "gui_tab_save_data",
        "gui_tab_save_image",
        "gui_tab_save_post_image",
        "gui_tab_save_result",
        "gui_tab_save_set_paths",
        "gui_tab_writeback_preview",
        "gui_tab_writeback_set",
        "gui_tab_writeback_apply",
    }
    assert new_tool_names.issubset(set(m.TOOLS))

    # Old MCP tool names are absent from the assembled table.
    old_tool_names = {
        "gui_save_data",
        "gui_save_image",
        "gui_save_post_image",
        "gui_save_result",
        "gui_save_set_paths",
        "gui_writeback_preview",
        "gui_writeback_set",
        "gui_writeback_apply",
    }
    assert old_tool_names.isdisjoint(set(m.TOOLS))

    # All new methods are auto-generated (no override, no exclusion).
    for method in new_wire_methods:
        assert method not in m._NON_GENERATED_METHODS, (
            f"{method} must not be excluded from generation"
        )
    for tool_name in new_tool_names:
        assert tool_name not in m._OVERRIDE_NAMES, (
            f"{tool_name} must not be an override"
        )


def test_phase170b_tab_run_analyze_tools():
    """Phase 170b tab listing/run/analyze normalization:
    - tab.list renamed to tab.list_all (auto-generated as gui_tab_list_all).
    - run.start renamed to tab.run_start; tab.run_cancel auto-generated.
    - analyze.start renamed to tab.analyze; post_analyze.start renamed to
      tab.post_analyze. Both are non-generated (short-wait degrade overrides).
    - Stage bundles renamed: gui_run_stage* -> gui_tab_stage*.
    - run/analyze/post_analyze tools renamed accordingly.
    - run.running_tab stays as internal-only (no agent tool generated).
    - Old names (gui_run_start, gui_run_stage1, gui_analyze, ...) are absent.
    """
    # tab.list_all: auto-generated (no special exclusion, no override).
    assert "tab.list_all" in METHOD_SPECS
    assert "tab.list_all" not in m._NON_GENERATED_METHODS
    assert "gui_tab_list_all" in m.TOOLS
    assert "gui_tab_list_all" not in m._OVERRIDE_NAMES

    # tab.list is gone from the wire contract entirely.
    assert "tab.list" not in METHOD_SPECS
    assert "gui_tab_list" not in m.TOOLS

    # run.running_tab: internal-only — wire method + spec stay but no agent tool.
    assert "run.running_tab" in METHOD_SPECS
    assert "run.running_tab" in m._NON_GENERATED_METHODS
    assert "gui_run_running_tab" not in m.TOOLS

    # tab.run_start: non-generated hand-written override.
    assert "tab.run_start" in METHOD_SPECS
    assert "tab.run_start" in m._NON_GENERATED_METHODS
    assert "gui_tab_run_start" in m.TOOLS
    assert "gui_tab_run_start" in m._OVERRIDE_NAMES

    # tab.run_cancel: auto-generated (no override needed).
    assert "tab.run_cancel" in METHOD_SPECS
    assert "tab.run_cancel" not in m._NON_GENERATED_METHODS
    assert "gui_tab_run_cancel" in m.TOOLS
    assert "gui_tab_run_cancel" not in m._OVERRIDE_NAMES

    # tab.analyze and tab.post_analyze: non-generated hand-written overrides.
    assert "tab.analyze" in METHOD_SPECS
    assert "tab.analyze" in m._NON_GENERATED_METHODS
    assert "gui_tab_analyze" in m.TOOLS
    assert "gui_tab_analyze" in m._OVERRIDE_NAMES

    assert "tab.post_analyze" in METHOD_SPECS
    assert "tab.post_analyze" in m._NON_GENERATED_METHODS
    assert "gui_tab_post_analyze" in m.TOOLS
    assert "gui_tab_post_analyze" in m._OVERRIDE_NAMES

    # Stage bundles: all present under new names, all old names absent.
    for stage in ("1", "2", "3", "4"):
        assert f"gui_tab_stage{stage}" in m.TOOLS, f"gui_tab_stage{stage} missing"
        assert f"gui_run_stage{stage}" not in m.TOOLS, (
            f"old gui_run_stage{stage} leaked"
        )

    # Wait/poll tools present under new names.
    for tool in (
        "gui_tab_run_wait",
        "gui_tab_run_poll",
        "gui_tab_analyze_wait",
        "gui_tab_analyze_poll",
        "gui_tab_post_analyze_wait",
        "gui_tab_post_analyze_poll",
    ):
        assert tool in m.TOOLS, f"{tool} missing"

    # Old names must be absent from the assembled tool table.
    old_names = {
        "gui_run_start",
        "gui_run_wait",
        "gui_run_poll",
        "gui_run_running_tab",
        "gui_analyze",
        "gui_analyze_wait",
        "gui_analyze_poll",
        "gui_post_analyze",
        "gui_post_analyze_wait",
        "gui_post_analyze_poll",
    }
    assert old_names.isdisjoint(set(m.TOOLS)), (
        f"old tool names leaked: {old_names & set(m.TOOLS)}"
    )


def test_phase170d_context_md_ml_prefix_editor_rename():
    """Phase 170d context md/ml prefix + editor open->new/save_as_module->save:
    - context.get_md*/set_md*/del_md* renamed to context.md_* wire methods.
    - context.get_ml*/del_ml*/rename_ml* renamed to context.ml_* wire methods.
    - ml.list_roles/create_from_role moved to context.ml_* wire methods.
    - editor.open renamed to editor.new; editor.commit tool_name ->  gui_editor_save.
    - New MCP tool names (gui_context_md_*/gui_context_ml_*/gui_editor_new/
      gui_editor_save) are present in the assembled table.
    - Old MCP tool names (gui_context_get_md*/gui_context_get_ml*,
      gui_context_del_ml*/gui_context_rename_ml*, gui_ml_*,
      gui_editor_open/gui_editor_save_as_module) are absent.
    """
    # New wire methods are present in the contract.
    new_wire_methods = {
        "context.md_get",
        "context.md_get_attr",
        "context.ml_get",
        "context.md_set_attr",
        "context.md_del_attr",
        "context.ml_del_module",
        "context.ml_del_waveform",
        "context.ml_rename_module",
        "context.ml_rename_waveform",
        "context.ml_list_roles",
        "context.ml_create_from_role",
        "editor.new",
    }
    assert new_wire_methods.issubset(set(METHOD_SPECS))

    # Old wire methods are gone from the contract entirely.
    old_wire_methods = {
        "context.get_md",
        "context.get_md_attr",
        "context.get_ml",
        "context.set_md_attr",
        "context.del_md_attr",
        "context.del_ml_module",
        "context.del_ml_waveform",
        "context.rename_ml_module",
        "context.rename_ml_waveform",
        "ml.list_roles",
        "ml.create_from_role",
        "editor.open",
    }
    assert old_wire_methods.isdisjoint(set(METHOD_SPECS))

    # New MCP tool names are present in the assembled table.
    new_tool_names = {
        "gui_context_md_get",
        "gui_context_md_get_attr",
        "gui_context_ml_get",
        "gui_context_md_set_attr",
        "gui_context_md_del_attr",
        "gui_context_ml_del_module",
        "gui_context_ml_del_waveform",
        "gui_context_ml_rename_module",
        "gui_context_ml_rename_waveform",
        "gui_context_ml_list_roles",
        "gui_context_ml_create_from_role",
        "gui_editor_new",
        "gui_editor_save",
    }
    assert new_tool_names.issubset(set(m.TOOLS))

    # Old MCP tool names are absent from the assembled table.
    # Note: gui_context_get_md_attrs / gui_context_set_md_attrs are MCP-side batch
    # convenience tools (not wire methods); their names are UNCHANGED — excluded here.
    old_tool_names = {
        "gui_context_get_md",
        "gui_context_get_md_attr",
        "gui_context_get_ml",
        "gui_context_set_md_attr",
        "gui_context_del_md_attr",
        "gui_context_del_ml_module",
        "gui_context_del_ml_waveform",
        "gui_context_rename_ml_module",
        "gui_context_rename_ml_waveform",
        "gui_ml_list_roles",
        "gui_ml_create_from_role",
        "gui_editor_open",
        "gui_editor_save_as_module",
    }
    assert old_tool_names.isdisjoint(set(m.TOOLS))

    # All new context md/ml + editor methods are auto-generated (no override, no exclusion).
    auto_generated_methods = new_wire_methods - {
        "editor.new"
    }  # editor.commit uses tool_name
    for method in auto_generated_methods:
        assert method not in m._NON_GENERATED_METHODS, (
            f"{method} must not be excluded from generation"
        )
    for tool_name in new_tool_names - {"gui_editor_save"}:
        assert tool_name not in m._OVERRIDE_NAMES, (
            f"{tool_name} must not be an override"
        )
    # editor.new (mapped from editor.new wire method) is also auto-generated.
    assert "editor.new" not in m._NON_GENERATED_METHODS
    assert "gui_editor_new" not in m._OVERRIDE_NAMES
    # editor.commit's tool_name is gui_editor_save (not an override either).
    assert "gui_editor_save" not in m._OVERRIDE_NAMES

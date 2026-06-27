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
    # P1 renamed the editor surface: editor.new -> gui_editor_open (override folds
    # the {tree} reply key to {cfg}); editor.get -> gui_editor_get_cfg; the
    # single-field gui_editor_set_field is retired (E5 batch-only) in favour of
    # gui_editor_set; editor.commit's tool stays gui_editor_save (tool_name
    # override); editor.discard stays gui_editor_discard.
    expected = {
        "gui_editor_open",
        "gui_editor_get_cfg",
        "gui_editor_set",
        "gui_editor_save",
        "gui_editor_discard",
    }
    assert expected.issubset(set(m.TOOLS))

    # Old tool names must be absent (retired / renamed in P1).
    assert "gui_editor_new" not in m.TOOLS
    assert "gui_editor_get" not in m.TOOLS
    assert "gui_editor_set_field" not in m.TOOLS
    assert "gui_editor_save_as_module" not in m.TOOLS

    open_tool = m.TOOLS["gui_editor_open"]
    # editor.open is modify-only (from_name); the blank-by-discriminator surface
    # was removed (create a blank via context.ml_create_from_role(role_id='<disc>:blank')).
    assert set(open_tool["inputSchema"]["required"]) == {"item_kind", "from_name"}
    props = open_tool["inputSchema"]["properties"]
    assert "discriminator" not in props
    assert "from_name" in props

    # gui_editor_get_cfg is a tree view keyed by editor_id with an optional dotted
    # prefix; the old flat-serving knobs (verbosity / under) are gone.
    get_tool = m.TOOLS["gui_editor_get_cfg"]
    assert get_tool["inputSchema"]["required"] == ["editor_id"]
    get_props = get_tool["inputSchema"]["properties"]
    assert "prefix" in get_props
    assert "verbosity" not in get_props
    assert "under" not in get_props

    # gui_editor_set is batch-only (edits = list of {path, value}). Each edit's
    # 'value' is a JSON kind (scalar OR tagged eval/value_ref object): its schema
    # is UNTYPED (no "type" key) so the MCP client never coerces a number against
    # a "string" member and stringifies it (e.g. 0.2 -> "0.2", which then fails
    # the downstream float-field check).
    set_props = m.TOOLS["gui_editor_set"]["inputSchema"]["properties"]
    assert "edits" in set_props
    value_schema = set_props["edits"]["items"]["properties"]["value"]
    assert "type" not in value_schema
    assert "value_ref" in m.TOOLS["gui_editor_set"]["description"]
    assert "gui_value_list" in m.TOOLS["gui_editor_set"]["description"]
    assert "value_ref" in value_schema["description"]


def test_cfg_set_tools_document_value_refs():
    assert "value_ref" in METHOD_SPECS["tab.set_cfg"].description
    assert "value_ref" in METHOD_SPECS["editor.set_field"].description
    assert "value_ref" in m.TOOLS["gui_tab_set_cfg"]["description"]
    assert "gui_value_list" in m.TOOLS["gui_tab_set_cfg"]["description"]

    tab_value_schema = m.TOOLS["gui_tab_set_cfg"]["inputSchema"]["properties"][
        "edits"
    ]["items"]["properties"]["value"]
    assert "type" not in tab_value_schema
    assert "value_ref" in tab_value_schema["description"]


def test_tab_new_is_a_pure_generated_forwarder():
    """gui_tab_new is the auto-generated tab.new forwarder (MCP 45): it returns
    just {tab_id}; the fan-out + guide fold moved to gui_tab_open. So tab.new
    must NOT be excluded from generation nor served by an override."""
    assert "tab.new" not in m._NON_GENERATED_METHODS
    assert "gui_tab_new" not in m._OVERRIDE_NAMES
    # The generated schema keeps the single required 'adapter_name' string param.
    schema = m.TOOLS["gui_tab_new"]["inputSchema"]
    assert schema["required"] == ["adapter_name"]
    assert schema["properties"]["adapter_name"]["type"] == "string"


def test_writeback_apply_is_a_pure_generated_forwarder():
    """gui_tab_writeback_apply is the auto-generated tab.writeback_apply forwarder:
    its reply is enriched in P3 ({applied_ids, written, context_version}) but the
    save chaining still lives in gui_tab_commit. So tab.writeback_apply must
    NOT be excluded from generation nor served by an override, and the agent schema
    exposes only 'tab_id'."""
    assert "tab.writeback_apply" not in m._NON_GENERATED_METHODS
    assert "gui_tab_writeback_apply" not in m._OVERRIDE_NAMES
    schema = m.TOOLS["gui_tab_writeback_apply"]["inputSchema"]
    # expected_versions is mcp_hidden; save_data is gone.
    assert set(schema["properties"]) == {"tab_id"}


def test_load_data_is_generated_with_hidden_expected_versions():
    assert "tab.load_data" in METHOD_SPECS
    assert "tab.load_data" not in m._NON_GENERATED_METHODS
    assert "gui_tab_load_data" in m.TOOLS

    schema = m.TOOLS["gui_tab_load_data"]["inputSchema"]
    assert set(schema["required"]) == {"tab_id", "data_path"}
    assert set(schema["properties"]) == {"tab_id", "data_path"}


def test_value_source_tools_are_generated_read_only_forwards():
    assert "value.list" in METHOD_SPECS
    assert "value.read" in METHOD_SPECS
    assert "value.list" not in m._NON_GENERATED_METHODS
    assert "value.read" not in m._NON_GENERATED_METHODS
    assert "gui_value_list" in m.TOOLS
    assert "gui_value_read" in m.TOOLS
    assert "gui_value_list" not in m._OVERRIDE_NAMES
    assert "gui_value_read" not in m._OVERRIDE_NAMES

    read_schema = m.TOOLS["gui_value_read"]["inputSchema"]
    assert read_schema["required"] == ["key"]
    assert set(read_schema["properties"]) == {"key", "type"}
    assert read_schema["properties"]["key"]["type"] == "string"
    assert read_schema["properties"]["type"]["type"] == "string"


def test_writeback_set_item_selected_is_boolean_schema():
    """``selected`` must render as a boolean schema so the client sends a real
    boolean. A JSON-any schema lets the client send the string "false", which
    ``bool("false")`` wrongly reads as True (selection never clears). The tool is
    renamed gui_tab_writeback_set_item in P3 (E6 editing-surface unification)."""
    props = m.TOOLS["gui_tab_writeback_set_item"]["inputSchema"]["properties"]
    assert props["selected"]["type"] == "boolean"
    # The unified surface carries both facet params (mutually exclusive at runtime).
    assert "proposed_value" in props
    assert "edits" in props


def test_view_screenshot_not_generated():
    """view.screenshot returns raw base64 PNG — it must NOT be auto-generated into
    a gui_view_screenshot agent tool. The only entry point is gui_screenshot
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

    # Editor tools require editor_id only (no tab_id branch). P1 retired the
    # single-field gui_editor_set_field; gui_editor_set (batch) + gui_editor_get_cfg
    # are the editor surface now.
    for tool_name in ("gui_editor_set", "gui_editor_get_cfg"):
        props = m.TOOLS[tool_name]["inputSchema"]["properties"]
        assert "tab_id" not in props, f"{tool_name} must not expose 'tab_id' anymore"
        assert "editor_id" in props


def test_phase170c_save_writeback_tools():
    """Save + writeback wire methods after P1.

    P1 folds the four save wire methods (tab.save_{data,image,post_image,result})
    into the single gui_tab_save override (artifact + figure selectors), so they
    move into _NON_GENERATED_METHODS and lose their per-method agent tool. The wire
    methods themselves stay (gui_tab_save and the gui_tab_commit bundle call them directly).
    tab.save_set_paths is renamed to gui_tab_set_save_paths (tool_name override,
    still generated). The writeback methods are untouched in P1 (writeback is P3) —
    still auto-generated as gui_tab_writeback_*.
    """
    # All save + writeback wire methods are present in the contract.
    save_wire_methods = {
        "tab.save_data",
        "tab.save_image",
        "tab.save_post_image",
        "tab.save_result",
    }
    other_wire_methods = {
        "tab.save_set_paths",
        "tab.writeback_preview",
        "tab.writeback_set",
        "tab.writeback_apply",
    }
    assert (save_wire_methods | other_wire_methods).issubset(set(METHOD_SPECS))

    # The four save wire methods are now excluded from generation (folded into
    # gui_tab_save); the single merged save tool is present as an override.
    for method in save_wire_methods:
        assert method in m._NON_GENERATED_METHODS, (
            f"{method} must be excluded from generation (folded into gui_tab_save)"
        )
    assert "gui_tab_save" in m.TOOLS
    assert "gui_tab_save" in m._OVERRIDE_NAMES

    # The per-method save tools no longer exist (merged into gui_tab_save).
    merged_away_tools = {
        "gui_tab_save_data",
        "gui_tab_save_image",
        "gui_tab_save_post_image",
        "gui_tab_save_result",
    }
    assert merged_away_tools.isdisjoint(set(m.TOOLS))

    # tab.save_set_paths is renamed to gui_tab_set_save_paths (still generated).
    assert "tab.save_set_paths" not in m._NON_GENERATED_METHODS
    assert "gui_tab_set_save_paths" in m.TOOLS
    assert "gui_tab_save_set_paths" not in m.TOOLS

    # Writeback tools after P3 (E6 editing-surface unification): preview->list,
    # set->set_item via tool_name overrides; wire methods unchanged, still
    # auto-generated (not in _NON_GENERATED_METHODS).
    writeback_tools = {
        "gui_tab_writeback_list",
        "gui_tab_writeback_set_item",
        "gui_tab_writeback_apply",
    }
    assert writeback_tools.issubset(set(m.TOOLS))
    # The pre-P3 writeback tool names are gone.
    assert {"gui_tab_writeback_preview", "gui_tab_writeback_set"}.isdisjoint(
        set(m.TOOLS)
    )
    for method in ("tab.writeback_preview", "tab.writeback_set", "tab.writeback_apply"):
        assert method not in m._NON_GENERATED_METHODS

    # The pre-170c MCP tool names stay absent.
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
    # tab.list_all: still the wire method (unchanged), but P1 exposes it as the
    # MCP tool gui_tab_list (tool_name override) with the named reply shape; it is
    # NOT an _OVERRIDE_NAMES entry (the rename is via the spec's tool_name, the
    # shape change lives in the dispatch handler).
    assert "tab.list_all" in METHOD_SPECS
    assert "tab.list_all" not in m._NON_GENERATED_METHODS
    assert "gui_tab_list" in m.TOOLS
    assert "gui_tab_list" not in m._OVERRIDE_NAMES

    # The old MCP tool name gui_tab_list_all is gone (renamed to gui_tab_list).
    assert "gui_tab_list_all" not in m.TOOLS
    # tab.list is gone from the wire contract entirely.
    assert "tab.list" not in METHOD_SPECS

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

    # tab.analyze and tab.post_analyze: non-generated hand-written overrides. P2
    # appends the _start suffix to the START tool names (the wait/poll halves are
    # retired in favour of the generic gui_op_wait / gui_op_poll, ADR-0026 §8).
    assert "tab.analyze" in METHOD_SPECS
    assert "tab.analyze" in m._NON_GENERATED_METHODS
    assert "gui_tab_analyze_start" in m.TOOLS
    assert "gui_tab_analyze_start" in m._OVERRIDE_NAMES

    assert "tab.post_analyze" in METHOD_SPECS
    assert "tab.post_analyze" in m._NON_GENERATED_METHODS
    assert "gui_tab_post_analyze_start" in m.TOOLS
    assert "gui_tab_post_analyze_start" in m._OVERRIDE_NAMES

    # Bundle tools: present under the P4 semantic names (open -> run ->
    # analyze_review -> commit); the old gui_tab_stage{1..4} (and the even older
    # gui_run_stage{1..4}) names are gone.
    for bundle in (
        "gui_tab_open",
        "gui_tab_run",
        "gui_tab_analyze_review",
        "gui_tab_commit",
    ):
        assert bundle in m.TOOLS, f"{bundle} missing"
        assert bundle in m._OVERRIDE_NAMES, f"{bundle} not an override"
    for stage in ("1", "2", "3", "4"):
        assert f"gui_tab_stage{stage}" not in m.TOOLS, (
            f"old gui_tab_stage{stage} leaked"
        )
        assert f"gui_run_stage{stage}" not in m.TOOLS, (
            f"old gui_run_stage{stage} leaked"
        )

    # Per-op wait/poll tools are RETIRED (P2 / ADR-0026 §8): the generic
    # gui_op_wait / gui_op_poll drive the handle a START reply folds.
    for tool in ("gui_op_wait", "gui_op_poll"):
        assert tool in m.TOOLS, f"{tool} missing"
    for retired in (
        "gui_tab_run_wait",
        "gui_tab_run_poll",
        "gui_tab_analyze_wait",
        "gui_tab_analyze_poll",
        "gui_tab_post_analyze_wait",
        "gui_tab_post_analyze_poll",
    ):
        assert retired not in m.TOOLS, f"{retired} should be retired"

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
        # P2 renamed the analyze/post_analyze START tools.
        "gui_tab_analyze",
        "gui_tab_post_analyze",
    }
    assert old_names.isdisjoint(set(m.TOOLS)), (
        f"old tool names leaked: {old_names & set(m.TOOLS)}"
    )


def test_phase170d_context_md_ml_prefix_editor_rename():
    """context md/ml + editor wire methods after P1.

    The context.md_*/ml_* and editor.* WIRE methods are unchanged by P1 (still the
    Phase 170d names), but the agent-facing MCP TOOL surface is reshaped:
    - context.md_get / md_get_attr / md_set_attr / md_del_attr move into
      _NON_GENERATED_METHODS — they feed the merged gui_context_md_read /
      gui_context_md_write / gui_context_md_delete overrides, so the single-attr /
      list-keys per-method tools are retired.
    - context.ml_get -> gui_context_ml_list (tool_name); ml_del_module/_waveform ->
      gui_context_ml_delete_module/_waveform (tool_name); ml_rename_*,
      ml_list_roles, ml_create_from_role keep their auto-generated names.
    - editor.new -> gui_editor_open (override); editor.commit -> gui_editor_save.
    """
    # The Phase 170d wire methods are still present in the contract.
    wire_methods = {
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
    assert wire_methods.issubset(set(METHOD_SPECS))

    # The pre-170d wire method names stay gone from the contract.
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

    # P1 MCP tool names present in the assembled table.
    new_tool_names = {
        "gui_context_md_read",
        "gui_context_md_write",
        "gui_context_md_delete",
        "gui_context_ml_list",
        "gui_context_ml_inspect",
        "gui_context_ml_delete_module",
        "gui_context_ml_delete_waveform",
        "gui_context_ml_rename_module",
        "gui_context_ml_rename_waveform",
        "gui_context_ml_list_roles",
        "gui_context_ml_create_from_role",
        "gui_editor_open",
        "gui_editor_save",
    }
    assert new_tool_names.issubset(set(m.TOOLS))

    # The merged-away per-attr md tools + the pre-P1 names are absent.
    old_tool_names = {
        # P1 merged these into gui_context_md_read/_write/_delete.
        "gui_context_md_get",
        "gui_context_md_get_attr",
        "gui_context_md_set_attr",
        "gui_context_md_del_attr",
        "gui_context_md_get_attrs",
        "gui_context_md_set_attrs",
        # P1 renamed these.
        "gui_context_ml_get",
        "gui_context_ml_del_module",
        "gui_context_ml_del_waveform",
        "gui_editor_new",
        # pre-170d names.
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
        "gui_editor_save_as_module",
    }
    assert old_tool_names.isdisjoint(set(m.TOOLS))

    # The md read/write/delete wire methods feed the merged overrides — excluded
    # from generation.
    for method in (
        "context.md_get",
        "context.md_get_attr",
        "context.md_set_attr",
        "context.md_del_attr",
    ):
        assert method in m._NON_GENERATED_METHODS, (
            f"{method} must be excluded (feeds a merged md override)"
        )
    for tool_name in (
        "gui_context_md_read",
        "gui_context_md_write",
        "gui_context_md_delete",
        "gui_context_ml_inspect",
    ):
        assert tool_name in m._OVERRIDE_NAMES, f"{tool_name} must be an override"

    # The ml rename/list-roles/create-from-role + editor.new wire methods stay
    # auto-generated (tool_name renames are NOT overrides).
    for method in (
        "context.ml_get",
        "context.ml_del_module",
        "context.ml_del_waveform",
        "context.ml_rename_module",
        "context.ml_rename_waveform",
        "context.ml_list_roles",
        "context.ml_create_from_role",
    ):
        assert method not in m._NON_GENERATED_METHODS, (
            f"{method} must not be excluded from generation"
        )
    for tool_name in (
        "gui_context_ml_list",
        "gui_context_ml_delete_module",
        "gui_context_ml_delete_waveform",
        "gui_context_ml_rename_module",
        "gui_context_ml_rename_waveform",
        "gui_context_ml_list_roles",
        "gui_context_ml_create_from_role",
    ):
        assert tool_name not in m._OVERRIDE_NAMES, (
            f"{tool_name} must not be an override"
        )

    # editor.new is served by the gui_editor_open OVERRIDE (folds {tree}->{cfg}),
    # so it IS excluded from generation and gui_editor_open is an override entry.
    assert "editor.new" in m._NON_GENERATED_METHODS
    assert "gui_editor_open" in m._OVERRIDE_NAMES
    # editor.commit's tool_name is gui_editor_save (not an override either).
    assert "gui_editor_save" not in m._OVERRIDE_NAMES
